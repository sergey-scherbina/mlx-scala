package mlx.llm

import java.nio.file.Files
import mlx.*
import mlx.Mlx.*

/** weights mapped: an unaligned file's aligned copy, the same values; a model mapped, the same tokens */
class SafeTensorsSuite extends munit.FunSuite:
  override val munitTimeout = scala.concurrent.duration.Duration(10, "min")
  Mlx.init()

  test("an unaligned file: its aligned copy maps, the values as read"):
    val dir = Files.createTempDirectory("safetensors")
    given s: Scope = new Scope
    val a = astype(ints(Array.range(0, 1000), 10, 100), Dtype.UInt32)
    val b = astype(ints(Array.range(0, 30), 3, 10), Dtype.BFloat16)
    // a key whose length leaves the header's end off a multiple of 4
    var key = "weight"
    val file = dir.resolve("m.safetensors")
    while { saveSafetensors(file.toString, Seq(key -> a, "b" -> b)); SafeTensors.header(file).aligned } do
      key = key + "x"
      Files.delete(file)
    val copy = SafeTensors.alignedOf(file, dir.resolve("aligned"))
    assert(copy != file && SafeTensors.header(copy).aligned)
    assertEquals(SafeTensors.alignedOf(copy, dir.resolve("aligned")), copy)
    val read = loadSafetensors(file.toString)
    val mapped = SafeTensors.map(copy)
    for k <- Seq(key, "b") do
      assertEquals(mapped(k).shape, read(k).shape)
      assertEquals(mapped(k).dtype, read(k).dtype)
      val diff = subtract(astype(mapped(k), Dtype.Float32), astype(read(k), Dtype.Float32))
      assertEquals(sum(reshape(multiply(diff, diff), -1), 0).eval().itemFloat, 0f)
    s.close()

  test("an unaligned file aligned in place, through its link: the values as read"):
    val dir = Files.createTempDirectory("safetensors")
    given s: Scope = new Scope
    val a = astype(ints(Array.range(0, 1000), 10, 100), Dtype.UInt32)
    var key = "w"
    val blob = dir.resolve("blob.safetensors")   // MLX names a file it writes so
    while { saveSafetensors(blob.toString, Seq(key -> a)); SafeTensors.header(blob).aligned } do key = key + "x"
    val link = Files.createSymbolicLink(dir.resolve("m.safetensors"), blob)
    val before = loadSafetensors(link.toString)(key)
    assertEquals(SafeTensors.alignInPlace(link), link)
    assert(Files.isSymbolicLink(link) && SafeTensors.header(blob).aligned)
    assert(Files.list(dir).toArray.length == 2, "nothing left beside it")
    val mapped = SafeTensors.map(link)(key)
    val diff = subtract(astype(mapped, Dtype.Float32), astype(before, Dtype.Float32))
    assertEquals(sum(reshape(multiply(diff, diff), -1), 0).eval().itemFloat, 0f)
    s.close()

  test("a model mapped: the tokens as read"):
    val dir = Models.snapshot("mlx-community/Qwen3-4B-4bit")
    assume(dir.isDefined, "mlx-community/Qwen3-4B-4bit is not in the HuggingFace cache")
    val chat = Chat.load(dir.get)
    val prompt = chat.ask("Name three colors.", Map("enable_thinking" -> false))
    def run(mapped: Boolean) =
      val model = Model.load(dir.get, mapped)
      val session = Session(model, prompt, 24, chat.stops)
      val out = Iterator.continually(session.next()).takeWhile(_.isDefined).flatten.toVector
      session.close()
      model.close()
      out
    assertEquals(run(mapped = true), run(mapped = false))
