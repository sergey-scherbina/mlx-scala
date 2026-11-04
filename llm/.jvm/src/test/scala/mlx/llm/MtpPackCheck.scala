package mlx.llm

import java.nio.file.Files
import mlx.*
import mlx.Mlx.*

/** the pack's tensors, dequantized, against the     val zero = multiply(a, scalar(0f))
    println(f"$k ${a.shape.mkString("x")}: |orig−pack|² ${d2(a, b)}%.3e, |orig|² ${d2(a, zero)}%.3e")inal's */
@main def mtpPackCheck(repo: String, original: String): Unit =
  val dir = Models.snapshot(repo).get
  val src = Files.createTempFile("mtp-pack", ".safetensors")
  HfRanges.tensors(original, src, k => k.startsWith("mtp.") && !k.contains("experts"), identity)
  Mlx.init()
  given Scope = new Scope
  val orig = Mlx.loadSafetensors(src.toString)
  val pack = Mlx.loadSafetensors(Mtp.path(dir).toString)
  def d2(x: Arr, y: Arr) =
    val d = subtract(astype(x, Dtype.Float32), astype(y, Dtype.Float32))
    sum(reshape(multiply(d, d), -1), 0).eval().itemFloat / x.shape.product
  for (k, a) <- orig.toSeq.sortBy(_._1) do
    val p = k.stripSuffix(".weight")
    val b =
      if pack.contains(s"$p.scales") then dequantize(pack(k), pack(s"$p.scales"), pack(s"$p.biases"), 64, 4)
      else subtract(astype(pack(k), Dtype.Float32), scalar(1f))
    println(f"$k ${a.shape.mkString("x")}: |orig−pack|² ${d2(a, b)}%.3e, |orig|² ${d2(a, multiply(a, scalar(0f)))}%.3e")
  Files.delete(src)
