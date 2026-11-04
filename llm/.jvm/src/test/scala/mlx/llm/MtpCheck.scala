package mlx.llm

import java.nio.file.Files
import mlx.*
import mlx.Mlx.*

/** the fetch and the quantization checked on a tensor both checkpoints have: the original's (fetched, quantized as
 * `Mtp.convert` does) against mlx-community's, both dequantized */
@main def mtpCheck(repo: String, original: String, layer: Int): Unit =
  val dir = Models.snapshot(repo).get
  val src = Files.createTempFile("mtp-check", ".safetensors")
  val names =
    Seq(s"layers.$layer.self_attn.q_proj.weight", s"layers.$layer.mlp.down_proj.weight", "embed_tokens.weight")
  HfRanges.tensors(original, src, k => names.exists(n => k == s"model.language_model.$n"), identity)
  Mlx.init()
  given Scope = new Scope
  val ours = Mlx.loadSafetensors(src.toString)
  val theirs = SafeTensors.shards(dir).flatMap(p => Mlx.loadSafetensors(p.toString)).toMap
  for n <- names do
    val a = ours(s"model.language_model.$n")
    val p = s"language_model.model.${n.stripSuffix(".weight")}"
    val b = dequantize(theirs(s"$p.weight"), theirs(s"$p.scales"), theirs(s"$p.biases"), 64, 4)
    val (q, s, bb) = quantize(astype(a, b.dtype), 64, 4)
    val c = dequantize(q, s, bb, 64, 4)
    def mean(x: Arr) = sum(reshape(astype(x, Dtype.Float32), -1), 0).eval().itemFloat / x.shape.product
    def d2(x: Arr, y: Arr) =
      val d = subtract(astype(x, Dtype.Float32), astype(y, Dtype.Float32))
      mean(multiply(d, d))
    println(f"$n ${a.shape.mkString("x")}: |orig−theirs|² ${d2(a, b)}%.3e, |ours−theirs|² ${d2(c, b)}%.3e, " +
      f"|orig|² ${d2(a, multiply(a, scalar(0f)))}%.3e")
  Files.delete(src)
