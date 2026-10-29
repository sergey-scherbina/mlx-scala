package mlx.llm

import okay.effects.json.Json

/** the tokenizer against HuggingFace `tokenizers` on Qwen3's tokenizer.json (ids from it, 0.22) — when the model is
 * in the HuggingFace cache */
class TokenizerSuite extends munit.FunSuite:
  private val expected = Seq(
    "Hello, world!" ->
      Vector(9707, 11, 1879, 0),
    "I'm sure they'll say it's fine — don't you think?" ->
      Vector(40, 2776, 2704, 807, 3278, 1977, 432, 594, 6915, 1959, 1513, 944, 498, 1744, 30),
    "def f(x):\n    return x * 2\n\n\n  # done  " ->
      Vector(750, 282, 2075, 982, 262, 470, 856, 353, 220, 17, 1406, 220, 671, 2814, 256),
    "Числа: 12345, π≈3.14159 and émigré café" ->
      Vector(72819, 13502, 30343, 25, 220, 16, 17, 18, 19, 20, 11, 51745, 144540, 18, 13, 16, 19, 16, 20, 24, 323,
        3958, 76, 5233, 963, 51950),
    "<|im_start|>user\nExplain Rust's ownership.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n" ->
      Vector(151644, 872, 198, 840, 20772, 33789, 594, 15278, 13, 151645, 198, 151644, 77091, 198, 151667, 271, 151668,
        271),
    "tabs\tand  double  spaces   end" ->
      Vector(30993, 52477, 220, 1990, 220, 12621, 256, 835),
    "emoji 🙂🚀 and 中文测试" ->
      Vector(37523, 27484, 145836, 323, 72858, 16744, 81705))

  private lazy val tokenizer = Models.snapshot("mlx-community/Qwen3-4B-4bit").map(d =>
    Tokenizer.fromJson(Json.fromText(String(java.nio.file.Files.readAllBytes(d.resolve("tokenizer.json")), "UTF-8"))))

  test("the ids HuggingFace gives, and the text back"):
    assume(tokenizer.isDefined, "mlx-community/Qwen3-4B-4bit is not in the HuggingFace cache")
    val t = tokenizer.get
    for (text, ids) <- expected do
      assertEquals(t.encode(text).toVector, ids, text)
      assertEquals(t.decode(ids), text)

  test("a stream gives text out only as far as it is whole UTF-8"):
    assume(tokenizer.isDefined)
    val t = tokenizer.get
    val ids = t.encode("emoji 🙂🚀 and 中文")
    val s = t.Stream()
    assertEquals(ids.map(s.push).mkString, "emoji 🙂🚀 and 中文")
