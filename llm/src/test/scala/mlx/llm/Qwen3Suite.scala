package mlx.llm

/** Qwen3-4B (MLX, 4-bit) answering, greedy — when it is in the HuggingFace cache */
class Qwen3Suite extends munit.FunSuite:
  override val munitTimeout = scala.concurrent.duration.Duration(5, "min")
  private lazy val dir = Models.snapshot("mlx-community/Qwen3-4B-4bit")

  test("the capital of France"):
    assume(dir.isDefined, "mlx-community/Qwen3-4B-4bit is not in the HuggingFace cache")
    val chat = Chat.load(dir.get)
    val model = Qwen3.load(dir.get)
    val prompt = chat.ask("What is the capital of France? Answer in one word.", Map("enable_thinking" -> false))
    val out = Vector.newBuilder[Int]
    val t = Generate.greedy(model, prompt, 16, chat.stops)(out += _)
    val text = chat.tokenizer.decode(out.result())
    println(s"answer: ${text.trim} — $t")
    assert(text.contains("Paris"), text)
    model.close()
