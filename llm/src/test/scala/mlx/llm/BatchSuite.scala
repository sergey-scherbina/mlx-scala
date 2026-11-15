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
