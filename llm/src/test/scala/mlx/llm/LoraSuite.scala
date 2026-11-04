package mlx.llm

/** LoRA on Qwen3-4B: the adapters start as nothing (the loss the model's own), and steps on an example lower it */
class LoraSuite extends munit.FunSuite:
  override val munitTimeout = scala.concurrent.duration.Duration(20, "min")

  test("Qwen3-4B: twenty AdamW steps on one example lower its loss"):
    val dir = Models.snapshot("mlx-community/Qwen3-4B-4bit")
    assume(dir.isDefined, "Qwen3-4B is not in the HuggingFace cache")
    val chat = Chat.load(dir.get)
    val model = Model.load(dir.get, mapped = true).asInstanceOf[Qwen3]
    val prompt = chat.ask("Write `def isImpure(symbol: String): Boolean` with symdex's `Equivalence.impure`.",
      Map("enable_thinking" -> false))
    val answer = chat.tokenizer.encode(
      "```scala\ndef isImpure(symbol: String): Boolean = Equivalence.impure.exists(symbol.startsWith)\n```")
    val ids = prompt ++ answer
    val lora = Lora(model, rank = 8, lr = 1e-3f)
    println(s"adapters: ${lora.names.size}, ${lora.size} parameters")
    val first = lora.evaluate(ids, prompt.length)
    val t0 = System.nanoTime
    val losses = (1 to 20).map(_ => lora.step(ids, prompt.length))
    val after = lora.evaluate(ids, prompt.length)
    println(f"loss: before $first%.4f, steps ${losses.map(l => f"$l%.3f").mkString(" ")}, after $after%.4f " +
      f"(${(System.nanoTime - t0) / 1e9 / 20}%.2f s a step)")
    assertEqualsFloat(losses.head, first, 1e-3f) // B = 0: the first step's loss is the model's own
    assert(after < first / 2, s"$first → $after")
    lora.close()
    model.close()
