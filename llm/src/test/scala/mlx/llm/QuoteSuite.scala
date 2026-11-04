package mlx.llm

/** EXACT QUOTES (okay-inlay specs/quotes.md): the port's laws on token ids, then on a model — every quote a span of
 * the source; outside the markers nothing changed */
class QuoteSuite extends munit.FunSuite:
  override val munitTimeout = scala.concurrent.duration.Duration(10, "min")

  /** whether `q` is a contiguous span of `source` */
  private def spanOf(q: Seq[Int], source: Seq[Int]) = q.isEmpty || source.indexOfSlice(q) >= 0

  test("allowed: none outside; after the open marker the source's tokens and the close; narrowing; closing forced"):
    val q = new Quote(Array(10, 11, 12, 10, 13), open = Array(1), close = Array(2, 3))
    assertEquals(q.allowed, None)
    q.took(5)
    assertEquals(q.allowed, None)
    q.took(1)
    assertEquals(q.allowed.map(_.toSet), Some(Set(10, 11, 12, 13, 2)))
    q.took(10) // at 0 and 3: next 11 or 13
    assertEquals(q.allowed.map(_.toSet), Some(Set(11, 13, 2)))
    q.took(13) // the end of the source: only the close
    assertEquals(q.allowed.map(_.toSet), Some(Set(2)))
    q.took(2)
    assertEquals(q.allowed.map(_.toSeq), Some(Seq(3)))
    q.took(3)
    assertEquals(q.allowed, None)
    assertEquals(q.quotes.toVector, Vector(Vector(10, 13)))

  test("a multi-token open marker recognized at its last token; any path the port allows is a span"):
    val source = Array.tabulate(200)(i => (i * 7919) % 23)
    val r = scala.util.Random(1)
    for _ <- 1 to 200 do
      val q = new Quote(source, open = Array(100, 101), close = Array(102))
      q.took(100); q.took(101)
      var steps = 0
      while q.inside && steps < 30 do
        val ok = q.allowed.get
        assert(ok.contains(102), "the close always allowed")
        val t = if r.nextInt(10) == 0 then 102 else ok(r.nextInt(ok.length))
        q.took(t)
        steps += 1
      if q.inside then q.took(102)
      q.quotes.foreach(qt => assert(spanOf(qt, source.toSeq), s"$qt not a span"))

  test("a quote may end inside a source token: its prefix allowed, then only the close"):
    val q = new Quote(Array(10, 340, 11), Array(1), Array(2), prefixes = { case 340 => Array(8); case _ => Array.empty })
    q.took(1); q.took(10)
    assertEquals(q.allowed.map(_.toSet), Some(Set(340, 8, 2)))
    q.took(8)
    assertEquals(q.allowed.map(_.toSet), Some(Set(2)))

  private val log =
    """2026-11-04 14:01:58 INFO  checkout: order 81731 paid (PLN 129.00)
      |2026-11-04 14:02:03 WARN  payments: slow response 4200 ms (order 81733)
      |2026-11-04 14:02:07 ERROR payments: timeout after 30000 ms (order 81734)
      |2026-11-04 14:02:09 INFO  checkout: order 81735 paid (PLN 54.50)
      |2026-11-04 14:02:15 ERROR inventory: stock lock lost for sku ZX-4471-B (order 81736)""".stripMargin

  test("Qwen3-4B: with the port every quote is in the log; outside quotes the same text"):
    val dir = Models.snapshot("mlx-community/Qwen3-4B-4bit")
    assume(dir.isDefined, "Qwen3-4B is not in the HuggingFace cache")
    val chat = Chat.load(dir.get)
    val model = Model.load(dir.get, mapped = true)
    val tok = chat.tokenizer
    val (open, close) = (tok.encode("⟦"), tok.encode("⟧"))
    val prompt = chat.ask("Answer from this log only. Put every log line, order number and amount you mention " +
      s"inside ⟦ ⟧, copied exactly.\n\n$log\n\nWhich orders failed, and why? Answer in two sentences.",
      Map("enable_thinking" -> false))
    val source = tok.encode(log)
    def run(port: Port | Null) =
      val s = Session(model, prompt, 160, chat.stops, port = port)
      val out = Iterator.continually(s.next()).takeWhile(_.isDefined).flatten.toVector
      s.close()
      out
    val free = run(null)
    val quote = Quote(tok, source)
    val held = run(quote)
    println(s"markers: open ${open.mkString(",")} close ${close.mkString(",")}")
    println(s"free: ${tok.decode(free)}")
    println(s"held: ${tok.decode(held)}")
    val d = free.indices.find(i => i >= held.length || free(i) != held(i)).getOrElse(-1)
    println(s"first difference at $d: free ${free.slice(d - 2, d + 4)} held ${held.slice(d - 2, d + 4)}")
    assert(quote.quotes.nonEmpty, "the model quoted nothing")
    // by text: a quote ending inside a source token is a prefix of it
    quote.quotes.foreach(q => assert(log.contains(tok.decode(q)), s"not in the log: ${tok.decode(q)}"))
    // the same up to the first quote's first token (the port free until the open marker)
    val firstOpen = free.indexOfSlice(open)
    if firstOpen >= 0 then assertEquals(held.take(firstOpen + open.length), free.take(firstOpen + open.length))
    model.close()

  // 400 commits (~20k tokens a prompt, ~10 min each): QUOTE_LONG=1. Measured 2026-11-04 on Qwen3-4B: 12 commits, free
  // 0 of 6 quotes not in the log; 400 greedy, free 1 of 6, held 0; 400 drawn (temperature 1), 0 and 0
  private val long = sys.env.contains("QUOTE_LONG")
  for (n, sampling) <- Seq(12 -> Sampling.Greedy) ++ (if long then Seq(400 -> Sampling.Greedy,
      400 -> Sampling(temperature = 1.0, seed = Some(3))) else Nil)
  do test(s"Qwen3-4B, full commit hashes of $n commits, ${if sampling.greedy then "greedy" else "drawn"}"):
    val dir = Models.snapshot("mlx-community/Qwen3-4B-4bit")
    assume(dir.isDefined, "Qwen3-4B is not in the HuggingFace cache")
    val chat = Chat.load(dir.get)
    val model = Model.load(dir.get, mapped = true)
    val tok = chat.tokenizer
    val r = scala.util.Random(7)
    val subjects = Vector("Fix the timeout in payments", "Add a cache to inventory", "Rename Order to Purchase",
      "Bump Scala to 3.9.0", "Remove the old checkout", "Log slow requests", "Retry failed webhooks",
      "Index orders by date", "Validate SKUs on import", "Split the reports module", "Escape quotes in CSV",
      "Round amounts half-up")
    val all = Vector.tabulate(n)(i => if i < subjects.length then subjects(i) else s"Refactor module $i")
    val shuffled = r.shuffle(all)
    val log = shuffled.map(s => Seq.fill(40)("0123456789abcdef"(r.nextInt(16))).mkString + " " + s).mkString("\n")
    val questions = Vector("Which commit changed rounding?", "Which commit touched webhooks?",
      "Which commit renamed a class?", "Which commit is about CSV?", "Which commit added a cache?",
      "Which commit changed the Scala version?")
    var freeBad, heldBad, freeQuotes, heldQuotes = 0
    for q <- questions do
      val prompt = chat.ask(s"$log\n\n$q Give its full hash inside ⟦ ⟧, copied exactly, and nothing else.",
        Map("enable_thinking" -> false))
      def run(port: Port | Null) =
        val s = Session(model, prompt, 80, chat.stops, sampling = sampling, port = port)
        val out = Iterator.continually(s.next()).takeWhile(_.isDefined).flatten.toVector
        s.close()
        tok.decode(out)
      val quoted = "⟦([^⟧]*)⟧?".r
      val free = run(null)
      val held = run(Quote(tok, tok.encode(log)))
      val fq = quoted.findAllMatchIn(free).map(_.group(1)).toVector
      val hq = quoted.findAllMatchIn(held).map(_.group(1)).toVector
      freeQuotes += fq.size; heldQuotes += hq.size
      freeBad += fq.count(x => !log.contains(x)); heldBad += hq.count(x => !log.contains(x))
      if n == 12 then println(s"$q\n  free: $free\n  held: $held")
    println(s"$n commits, ${if sampling.greedy then "greedy" else "drawn"} — hashes: free $freeBad of $freeQuotes quotes not in the log; held $heldBad of $heldQuotes")
    assertEquals(heldBad, 0)
    model.close()
