package mlx.llm

/** LoRA's batched pass (`Model.hiddenBatch`, `Lora.stepBatch`): a batch's loss is the mean of its examples' losses
 * taken one at a time — examples of different lengths, padded, the adapters not zero; and a batched step's time */
class LoraBatchSuite extends munit.FunSuite:
  override val munitTimeout = scala.concurrent.duration.Duration(20, "min")

  test("Qwen3-0.6B: a padded batch's loss is the mean of its examples' own; a batched step against single ones"):
    val dir = Models.snapshot("mlx-community/Qwen3-0.6B-4bit")
    assume(dir.isDefined, "Qwen3-0.6B-4bit is not in the HuggingFace cache")
    val chat = Chat.load(dir.get)
    val model = Model.load(dir.get, mapped = true).asInstanceOf[Qwen3]
    def example(q: String, a: String) =
      val p = chat.ask(q, Map("enable_thinking" -> false))
      (p ++ chat.tokenizer.encode(a), p.length)
    val examples = Vector(
      example("How many orders are unpaid?", "answer: count\nwhere: status = unpaid\nend"),
      example("Total amount by city, largest first, in the table of orders placed this year by our customers.",
        "answer: sum amount\ngroup: city\norder: answer desc\nend"),
      example("Hi", "rows"),
      example("Which customer spent the most?", "answer: sum amount\ngroup: customer\norder: answer desc\nlimit: 1"))
    val lora = Lora(model, rank = 8, projs = Seq("q", "k", "v", "o", "gate", "up", "down"), lr = 1e-3f)
    // the adapters made non-zero: a few steps
    (1 to 3).foreach(_ => lora.step(examples(0)._1, examples(0)._2))
    val one = examples.map((ids, from) => lora.evaluate(ids, from))
    val batch = lora.evaluateBatch(examples)
    println(f"each: ${one.map(x => f"$x%.4f").mkString(" ")}; mean ${one.sum / one.size}%.4f; batch $batch%.4f")
    assertEqualsDouble(batch.toDouble, one.sum / one.size.toDouble, 0.02 * math.max(1.0, one.sum / one.size))
    // THE TRIE: the same law — and the short examples share the chat template's beginning
    val tree = lora.evaluateTree(examples)
    println(f"tree $tree%.4f")
    assertEqualsDouble(tree.toDouble, one.sum / one.size.toDouble, 0.02 * math.max(1.0, one.sum / one.size))
    // THE CHOICES: an option's score the sum of its tokens' log-probabilities after the context (each from `evaluate`,
    // a mean over its tokens), the loss −log softmax at the meant one, averaged over the choices
    val tok = chat.tokenizer
    val ctx = chat.ask("How many orders are unpaid?", Map("enable_thinking" -> false))
    val opts = Vector("answer: count\n", "answer: rows\n", "answer: sum amount\n").map(o => tok.encode(o))
    val ctx2 = ctx ++ tok.encode("answer: count\n")
    val opts2 = Vector("where: status = unpaid\n", "where: status = paid\n", "end").map(o => tok.encode(o))
    val choices = Vector(Lora.Choice(ctx, opts, 1), Lora.Choice(ctx2, opts2, 2))   // the unlikely ones: a loss of nats
    def scoreOf(c: Array[Int], o: Array[Int]) = -lora.evaluate(c ++ o, c.length).toDouble * o.length
    def nll(c: Lora.Choice) =
      val s = c.options.map(o => scoreOf(c.context, o))
      val mx = s.max
      mx + math.log(s.map(x => math.exp(x - mx)).sum) - s(c.meant)
    val want = choices.map(nll).sum / choices.size
    val got = lora.evaluateChoices(choices)
    println(f"choices: by single passes $want%.4f, over the trie $got%.4f")
    assertEqualsDouble(got.toDouble, want, 0.02 * math.max(1.0, want))
    // the time: 8 examples, one at a time against one batch
    val eight = Vector.fill(2)(examples).flatten
    def secs(f: => Unit) = { val t = System.nanoTime; f; (System.nanoTime - t) / 1e9 }
    eight.foreach((ids, from) => lora.step(ids, from))   // warm
    lora.stepBatch(eight)
    val single = secs(eight.foreach((ids, from) => lora.step(ids, from)))
    val batched = secs(lora.stepBatch(eight))
    println(f"8 examples: one at a time $single%.2f s, one batch $batched%.2f s")
    // LONG ONES (a frame's prompt: ~450 tokens, the answer ~30), alternated three times: the same load on the GPU
    val table = (1 to 40).map(i => s"order $i, city ${Seq("Kyiv", "Lviv", "Odesa")(i % 3)}, amount ${i * 7}")
      .mkString("; ")
    val long = Vector.tabulate(8)(k => example(s"$table. Question $k: total amount by city?",
      "answer: sum amount\ngroup: city\nend"))
    // as the frames' data is: three tables, a few questions each (the rule and a table shared)
    def tableOf(t: Int) = (1 to 30).map(i => s"row $i, ${Seq("Kyiv", "Lviv", "Odesa")((i + t) % 3)}, ${i * (t + 3)}")
      .mkString("; ")
    val mixed = Vector.tabulate(8)(k => example(s"${tableOf(k % 3)}. Question $k: total by city?",
      "answer: sum amount\ngroup: city\nend"))
    println(s"long: ${long.map(_._1.length).mkString(" ")} tokens; mixed ${mixed.map(_._1.length).mkString(" ")}")
    for (name, set) <- Seq("same table" -> long, "three tables" -> mixed) do
      lora.stepTree(set); lora.stepBatch(set)
      val times = (1 to 3).map(_ => (secs(set.foreach((ids, from) => lora.step(ids, from))),
        secs(lora.stepBatch(set)), secs(lora.stepTree(set))))
      println(f"$name, 8: one at a time ${times.map(_._1).min}%.2f s, a batch ${times.map(_._2).min}%.2f s, " +
        f"a trie ${times.map(_._3).min}%.2f s")
      // the forward alone: where a step's time goes
      val ahead = (1 to 3).map(_ => (secs(set.foreach((ids, from) => lora.evaluate(ids, from))),
        secs(lora.evaluateBatch(set)), secs(lora.evaluateTree(set))))
      println(f"$name, 8, forward: one at a time ${ahead.map(_._1).min}%.2f s, a batch ${ahead.map(_._2).min}%.2f s, " +
        f"a trie ${ahead.map(_._3).min}%.2f s")
    lora.close()
    model.close()
