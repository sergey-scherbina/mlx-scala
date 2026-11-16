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

/** the tokenizer read without its JSON tree (`fromFile`) is the one read through it (`fromJson`): the same tables, the
 * same ids — on every tokenizer in the HuggingFace cache that the models here use */
class TokenizerFileSuite extends munit.FunSuite:
  override val munitTimeout = scala.concurrent.duration.Duration(5, "min")
  private val texts = Seq("Hello, world!", "Числа: 12345, π≈3.14159 and émigré café", "emoji 🙂🚀 and 中文测试",
    "<|im_start|>user\nhi<|im_end|>\n", "def f(x):\n    return x * 2\n", "\\u0041 \"quoted\" back\\slash\ttab")

  for repo <- Seq("mlx-community/Qwen3-4B-4bit", "mlx-community/Qwen3.5-4B-MLX-4bit",
      "mlx-community/Qwen3.6-35B-A3B-4bit", "mlx-community/Qwen3-0.6B-4bit") do
    test(s"$repo: fromFile = fromJson"):
      val dir = Models.snapshot(repo)
      assume(dir.isDefined, s"$repo is not in the HuggingFace cache")
      val file = dir.get.resolve("tokenizer.json")
      val a = Tokenizer.fromFile(file)
      val b = Tokenizer.fromJson(Json.fromText(String(java.nio.file.Files.readAllBytes(file), "UTF-8")))
      val (va, ra, pa, aa, sa, na) = a.tables
      val (vb, rb, pb, ab, sb, nb) = b.tables
      assertEquals(va, vb)
      assertEquals(ra, rb)
      assertEquals(pa, pb)
      assertEquals(aa, ab)
      assertEquals(sa, sb)
      assertEquals(na, nb)
      for t <- texts do assertEquals(a.encode(t).toVector, b.encode(t).toVector, t)

  test("escapes, surrogate pairs and nesting, as JSON reads them"):
    val text = """{"a": "x\"y\\zé🙂", "skip": {"n": [1, {"m": "]"}], "t": true}, "b": [["p", "q"], "r s"]}"""
    val s = JsonScan(text.getBytes("UTF-8"))
    var got = Vector.empty[String]
    s.obj {
      case "a" => got :+= s.str()
      case "b" => s.arr(() => if s.peek == '[' then s.arr(() => got :+= s.str()) else got :+= s.str())
      case _ => s.skip()
    }
    assertEquals(got, Vector("x\"y\\zé🙂", "p", "q", "r s"))
