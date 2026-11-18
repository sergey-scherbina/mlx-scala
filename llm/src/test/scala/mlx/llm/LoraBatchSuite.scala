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
    // the time: 8 examples, one at a time against one batch
    val eight = Vector.fill(2)(examples).flatten
    def secs(f: => Unit) = { val t = System.nanoTime; f; (System.nanoTime - t) / 1e9 }
    eight.foreach((ids, from) => lora.step(ids, from))   // warm
    lora.stepBatch(eight)
    val single = secs(eight.foreach((ids, from) => lora.step(ids, from)))
    val batched = secs(lora.stepBatch(eight))
    println(f"8 examples: one at a time $single%.2f s, one batch $batched%.2f s")
    lora.close()
    model.close()
