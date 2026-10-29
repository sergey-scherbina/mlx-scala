package mlx.llm

import okay.effects.json.Json

/** Qwen3.5-9B and Qwen3.6-35B-A3B (MLX, 4-bit): greedy tokens equal to mlx_lm's on the same prompt ids
 * (resources/qwen35-9b-greedy.json, qwen36-35b-greedy.json, by mlx_lm 0.31) — each when it is in the HuggingFace
 * cache. MLX_SKIP_BIG=1 leaves the 35B out */
class Qwen35Suite extends munit.FunSuite:
  override val munitTimeout = scala.concurrent.duration.Duration(10, "min")

  for (repo, file) <- Seq("mlx-community/Qwen3.5-9B-MLX-4bit" -> "qwen35-9b-greedy.json",
      "mlx-community/Qwen3.6-35B-A3B-4bit" -> "qwen36-35b-greedy.json") do
    test(s"$repo: mlx_lm's tokens, greedy")(check(repo, file))

  private def check(repo: String, file: String): Unit =
    val dir = Models.snapshot(repo)
    assume(dir.isDefined, s"$repo is not in the HuggingFace cache")
    assume(!(repo.contains("35B") && sys.env.contains("MLX_SKIP_BIG")), "MLX_SKIP_BIG")
    val ref = Json.fromText(scala.io.Source.fromInputStream(getClass.getResourceAsStream(s"/$file"),
      "UTF-8").mkString)
    val prompt = ref("prompt").list.map(_.int).toArray
    val expected = ref("tokens").list.map(_.int)
    val model = Model.load(dir.get)
    val out = Vector.newBuilder[Int]
    val t = Generate.greedy(model, prompt, expected.length, _ => false)(out += _)
    val chat = Chat.load(dir.get)
    println(s"${chat.tokenizer.decode(out.result())} — $t")
    assertEquals(out.result(), expected)
    assertEquals(chat.tokenizer.encode(ref("text").string).toVector.take(5), expected.take(5))
    model.close()
