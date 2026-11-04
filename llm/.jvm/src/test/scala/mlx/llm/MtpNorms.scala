package mlx.llm

import java.nio.file.Files
import mlx.*
import mlx.Mlx.*

/** which norms mlx-community's conversion shifted (+1): a layer's norms, the original's against the model's */
@main def mtpNorms(repo: String, original: String, layer: Int): Unit =
  val dir = Models.snapshot(repo).get
  val src = Files.createTempFile("mtp-norms", ".safetensors")
  val names = Seq("input_layernorm", "post_attention_layernorm", "self_attn.q_norm", "self_attn.k_norm")
    .map(n => s"layers.$layer.$n.weight") :+ "norm.weight"
  HfRanges.tensors(original, src, k => names.exists(n => k == s"model.language_model.$n"), identity)
  Mlx.init()
  given Scope = new Scope
  val ours = Mlx.loadSafetensors(src.toString)
  val theirs = SafeTensors.shards(dir).flatMap(p => Mlx.loadSafetensors(p.toString)).toMap
  def d(x: Arr, y: Arr) =
    val e = subtract(astype(x, Dtype.Float32), astype(y, Dtype.Float32))
    max(reshape(maximum(e, multiply(e, scalar(-1f))), -1), 0).eval().itemFloat
  for n <- names do
    val a = ours(s"model.language_model.$n")
    val b = theirs(s"language_model.model.$n")
    println(f"$n: as is ${d(a, b)}%.4f, +1 ${d(add(astype(a, Dtype.Float32), scalar(1f)), b)}%.4f")
  Files.delete(src)
