package okay.mlx.llm

import java.nio.file.Files

/** Qwen3-4B (MLX, 4-bit) answering, greedy — when it is in the HuggingFace cache */
class Qwen3Suite extends munit.FunSuite:
  override val munitTimeout = scala.concurrent.duration.Duration(5, "min")
  private lazy val dir = Models.snapshot("mlx-community/Qwen3-4B-4bit")

  test("the capital of France"):
    assume(dir.isDefined, "mlx-community/Qwen3-4B-4bit is not in the HuggingFace cache")
    val tokenizer = Tokenizer.fromJson(Json.parse(Files.readString(dir.get.resolve("tokenizer.json"))))
    val model = Qwen3.load(dir.get)
    val prompt = tokenizer.encode(Generate.chat("What is the capital of France? Answer in one word."))
    val out = Vector.newBuilder[Int]
    val t = Generate.greedy(model, prompt, 16, id => id == model.config.eos || id == tokenizer.id("<|im_end|>"))(
      out += _)
    val text = tokenizer.decode(out.result())
    println(s"answer: ${text.trim} — $t")
    assert(text.contains("Paris"), text)
    model.close()
