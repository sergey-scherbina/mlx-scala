package mlx.llm

import mlx.*
import mlx.Mlx.*

/** A BATCHED DECODE STEP against each row alone: `stepMany`'s logits close to `logits` per row; three sessions
 * decoded in one batch give each one's tokens alone (greedy; a bf16 near tie may part them late — reported); and
 * the throughput of the batch against the three one after another */
class BatchSuite extends munit.FunSuite:
  override val munitTimeout = scala.concurrent.duration.Duration(30, "min")
  private val questions = Seq(
    "Name three rivers of Europe and say in one sentence each where they flow.",
    "Write a Scala function that reverses a list without using reverse.",
    "Explain in two sentences why the sky is blue.")

  for repo <- Seq("mlx-community/Qwen3-0.6B-4bit", "mlx-community/Qwen3.5-0.8B-MLX-4bit",
      "mlx-community/Qwen3.6-35B-A3B-4bit") do
    test(s"$repo: a batch of three — each row's logits, each session's tokens"):
      val dir = Models.snapshot(repo)
      assume(dir.isDefined, s"$repo is not in the HuggingFace cache")
      val chat = Chat.load(dir.get)
      val model = Model.load(dir.get, mapped = true)
      val prompts = questions.map(q => chat.ask(q, Map("enable_thinking" -> false)))

      // the step: three states fed their prompts but the last token, twice — one set stepped together, one apart
      val s = new Scope
      given Scope = s
      def fed() = prompts.map { p =>
        val st = model.newState()
        model.logits(ints(p.init, 1, p.length - 1), st).eval()
        st
      }
      val (together, apart) = (fed(), fed())
      val lasts = ints(prompts.map(_.last).toArray, 1, prompts.length)
      val many = astype(model.stepMany(lasts, together), Dtype.Float32)
      val v = many.dim(-1)
      val worst = prompts.indices.map { r =>
        val one = astype(model.logits(ints(Array(prompts(r).last), 1, 1), apart(r)), Dtype.Float32)
        val got = slice(many, Seq(0, r, 0), Seq(1, r + 1, v))
        val (a, b) = (got.toFloats, one.toFloats)
        a.indices.map(i => math.abs(a(i) - b(i))).max / b.map(math.abs).max
      }.max
      // the noise the code has without a batch: the last token's logits from the whole prompt in one pass (a product
      // over n rows) against the prompt but the last, then the last alone (over one row) — both used today
      val noise = prompts.map { p =>
        val (a1, b1) = (model.newState(), model.newState())
        val whole = astype(model.logits(ints(p, 1, p.length), a1), Dtype.Float32).toFloats
        model.logits(ints(p.init, 1, p.length - 1), b1).eval()
        val step = astype(model.logits(ints(Array(p.last), 1, 1), b1), Dtype.Float32).toFloats
        a1.close(); b1.close()
        whole.indices.map(i => math.abs(whole(i) - step(i))).max / step.map(math.abs).max
      }.max
      // three copies of one row in a batch: the same logits each (a row's own work is its own)
      val copies = Seq.fill(3) {
        val st = model.newState()
        model.logits(ints(prompts(0).init, 1, prompts(0).length - 1), st).eval()
        st
      }
      val same = astype(model.stepMany(ints(Array.fill(3)(prompts(0).last), 1, 3), copies), Dtype.Float32)
      val rows = (0 until 3).map(r => slice(same, Seq(0, r, 0), Seq(1, r + 1, v)).toFloats)
      val spread = rows.tail.map(o => o.indices.map(i => math.abs(o(i) - rows(0)(i))).max).max
      copies.foreach(_.close())
      println(f"$repo: the noise without a batch ${noise * 100}%.3f%%; three copies batched differ by $spread%.5f")
      (together ++ apart).foreach(_.close())
      s.close()
      println(f"$repo: stepMany against each row, the largest difference ${worst * 100}%.3f%% of the logits' range")
      // a batch's rows within the code's own noise (a product over n rows against over one), and no row touches another
      assert(worst <= 1.5 * noise + 0.005, s"rows differ by $worst, the noise $noise")
      assert(spread == 0f, s"copies differ by $spread")

      // decoding: three sessions one after another, then the three in one batch, called in turn
      val n = 64
      def drain(ss: Seq[Session]): Seq[Vector[Int]] =
        val out = ss.map(_ => Vector.newBuilder[Int])
        var open = ss.indices.toSet
        while open.nonEmpty do
          for i <- open.toSeq.sorted do
            ss(i).next() match
              case Some(t) => out(i) += t
              case None => open -= i
        out.map(_.result())
      val t0 = System.nanoTime()
      val alone = prompts.map(p => drain(Seq(Session(model, p, n, chat.stops))).head)
      val t1 = System.nanoTime()
      val batch = Batch(model)
      val batched = drain(prompts.map(p => Session(model, p, n, chat.stops, batch = batch)))
      val t2 = System.nanoTime()
      val total = alone.map(_.length).sum
      val agree = alone.zip(batched).map((a, b) => a.zip(b).takeWhile(_ == _).length)
      println(f"$repo: tokens alone ${alone.map(_.length)}, the same prefix batched $agree; " +
        f"${total / ((t1 - t0) / 1e9)}%.1f tok/s one after another, " +
        f"${batched.map(_.length).sum / ((t2 - t1) / 1e9)}%.1f tok/s batched " +
        f"(${batch.stepped.toDouble / batch.rounds}%.2f members a round)")
      // the first 8 tokens of each equal; the rest reported (a bf16 near tie parts them, as it parts a prompt read
      // whole from one read but its last token — the noise above)
      agree.zip(alone).foreach((k, a) => assert(k >= math.min(8, a.length), s"parted at $k of ${a.length}"))
      model.close()

/** A SESSION STARTED ALONE with guesses or its own drafts turns plain when a second starts, and joins the rounds:
 * its tokens are its tokens alone (greedy; a bf16 near tie may part them late — reported), the rounds are two wide */
class BatchJoinSuite extends munit.FunSuite:
  override val munitTimeout = scala.concurrent.duration.Duration(30, "min")
  for (repo, guesses, drafts) <- Seq(("mlx-community/Qwen3-0.6B-4bit", 6, 0),
      ("mlx-community/Qwen3.6-35B-A3B-4bit", 0, 2)) do
    test(s"$repo: alone with guesses $guesses / drafts $drafts, then batched"):
      val dir = Models.snapshot(repo)
      assume(dir.isDefined, s"$repo is not in the HuggingFace cache")
      val chat = Chat.load(dir.get)
      val model = Model.load(dir.get, mapped = true)
      val first = chat.ask("Write a numbered list of ten facts about the Moon.", Map("enable_thinking" -> false))
      val second = chat.ask("Write a numbered list of ten facts about Mars.", Map("enable_thinking" -> false))
      val n = 80
      def all(s: Session) = Iterator.continually(s.next()).takeWhile(_.isDefined).flatten.toVector
      val aloneA = all(Session(model, first, n, chat.stops, guesses = guesses, drafts = drafts))
      val aloneB = all(Session(model, second, n, chat.stops))
      val batch = Batch(model)
      val a = Session(model, first, n, chat.stops, guesses = guesses, drafts = drafts, batch = batch)
      val got = Vector.newBuilder[Int]
      // a alone for its first 20 tokens (its guesses or drafts working), then b starts: a turns plain
      for _ <- 0 until 20 do a.next().foreach(got += _)
      val b = Session(model, second, n, chat.stops, batch = batch)
      val gotB = Vector.newBuilder[Int]
      var open = Set("a", "b")
      while open.nonEmpty do
        if open("a") then a.next() match { case Some(t) => got += t; case None => open -= "a" }
        if open("b") then b.next() match { case Some(t) => gotB += t; case None => open -= "b" }
      val (ra, rb) = (got.result(), gotB.result())
      val agreeA = aloneA.zip(ra).takeWhile(_ == _).length
      val agreeB = aloneB.zip(rb).takeWhile(_ == _).length
      println(f"$repo: a ${ra.length} tokens (alone ${aloneA.length}), the same prefix $agreeA; b ${rb.length} " +
        f"(alone ${aloneB.length}), the same prefix $agreeB; ${batch.stepped.toDouble / batch.rounds}%.2f a round " +
        f"over ${batch.rounds} rounds")
      assert(agreeA >= 20 && agreeB >= math.min(8, aloneB.length))
      assert(batch.rounds > 0 && batch.stepped.toDouble / batch.rounds > 1.5)
      model.close()

/** THE PROMPT IN CHUNKS (`prefillStep`): fed 61 tokens a call, the same greedy tokens as fed whole; and fed while
 * another session decodes in the same batch — its chunks between the other's steps — both sessions still their own
 * tokens (to the noise of a batched step) */
class PrefillStepSuite extends munit.FunSuite:
  override val munitTimeout = scala.concurrent.duration.Duration(10, "min")
  test("Qwen3-0.6B: chunked prefill = whole; interleaved with another's decode"):
    val dir = Models.snapshot("mlx-community/Qwen3-0.6B-4bit")
    assume(dir.isDefined, "mlx-community/Qwen3-0.6B-4bit is not in the HuggingFace cache")
    val chat = Chat.load(dir.get)
    val model = Model.load(dir.get, mapped = true)
    val long = chat.ask("Summarize in one sentence: " + (1 to 120).map(i => s"Fact $i: the number $i is ${
      if i % 2 == 0 then "even" else "odd"}.").mkString(" "), Map("enable_thinking" -> false))
    val short = chat.ask("Write a numbered list of ten facts about the Moon.", Map("enable_thinking" -> false))
    assert(long.length > 600, long.length)
    val n = 48
    def all(s: Session) = Iterator.continually(s.next()).takeWhile(_.isDefined).flatten.toVector
    val whole = all(Session(model, long, n, chat.stops))
    val chunked = Session(model, long, n, chat.stops)
    var calls = 0
    while chunked.prefillStep(61) do calls += 1
    assert(calls >= long.length / 61 - 1, calls)
    assertEquals(all(chunked), whole)
    // interleaved: a decodes; b's prompt fed a chunk between each of a's tokens
    val aloneA = all(Session(model, short, n, chat.stops))
    val batch = Batch(model)
    val a = Session(model, short, n, chat.stops, batch = batch)
    val b = Session(model, long, n, chat.stops, batch = batch)
    val gotA = Vector.newBuilder[Int]
    for _ <- 0 until 5 do a.next().foreach(gotA += _)
    var steps = 0
    while b.prefillStep(61) do
      a.next().foreach(gotA += _)
      steps += 1
    assert(steps > 5, steps)
    val gotB = Vector.newBuilder[Int]
    var open = Set("a", "b")
    while open.nonEmpty do
      if open("a") then a.next() match { case Some(t) => gotA += t; case None => open -= "a" }
      if open("b") then b.next() match { case Some(t) => gotB += t; case None => open -= "b" }
    val (ra, rb) = (gotA.result(), gotB.result())
    val agreeA = aloneA.zip(ra).takeWhile(_ == _).length
    val agreeB = whole.zip(rb).takeWhile(_ == _).length
    println(s"chunked prefill: $calls calls; interleaved: a agrees $agreeA of ${aloneA.length}, b $agreeB of ${whole.length}")
    assert(agreeA >= 20 && agreeB >= 8, s"a $agreeA, b $agreeB")
    model.close()
