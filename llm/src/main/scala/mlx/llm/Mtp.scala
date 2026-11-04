package mlx.llm

import okay.effects.json.Json

import java.nio.file.{Files, Path, Paths}
import mlx.*
import mlx.Mlx.*

/** MULTI-TOKEN PREDICTION, Qwen3.5 and Qwen3.6's: a draft layer of the checkpoint's own (`mtp.*`) that guesses the
 * token after the next from the hidden state that chose the next and the next's embedding. mlx-community's
 * conversions leave it out; it is taken from the original checkpoint (bf16, the `mtp.*` tensors alone — `MtpFetch`
 * reads them by their byte ranges), quantized as the model's layers are, and kept in a file of its own (`path`):
 * the model's directory is HuggingFace's, not written to.
 *
 * In the pack, the names as the model's: `mtp.fc`, `mtp.pre_fc_norm_embedding`, `mtp.pre_fc_norm_hidden`,
 * `mtp.norm`, `mtp.layers.0.*` (a full attention layer; its experts as `mlp.switch_mlp.{gate,up,down}_proj`) */
object Mtp:
  /** the pack of the model in `dir`: MLX_SCALA_MTP, else ~/.cache/mlx-scala/mtp/<model>/mtp.safetensors (<model>:
   * HuggingFace's repository folder, `models--org--name`, when `dir` is a snapshot in its cache) */
  def path(dir: Path): Path =
    sys.env.get("MLX_SCALA_MTP").map(Paths.get(_)).getOrElse(
      Paths.get(sys.props("user.home"), ".cache", "mlx-scala", "mtp", name(dir), "mtp.safetensors"))

  private def name(dir: Path): String =
    val abs = dir.toAbsolutePath.normalize
    val parts = (0 until abs.getNameCount).map(abs.getName(_).toString)
    parts.lastIndexWhere(_.startsWith("models--")) match
      case -1 => parts.lastOption.getOrElse("model")
      case i => parts(i)

  /** a pack tensor's quantization: as the model's layer 0 has it (its router and shared expert's gate at 8 bits in
   * mlx-community's), else the model's */
  def quantOf(prefix: String, config: Qwen35Config): (Int, Int) =
    config.quantized.getOrElse(prefix.replace("mtp.layers.0.", "language_model.model.layers.0."),
      (config.groupSize, config.bits))

  /** the overrides of the model's quantization for the pack's tensors, to load them by */
  def overrides(names: Iterable[String], config: Qwen35Config): Map[String, (Int, Int)] =
    names.filter(_.endsWith(".scales")).map(_.stripSuffix(".scales")).map(p => p -> quantOf(p, config)).toMap

  /** the original checkpoint's norms are zero-centred (weight 1 + w); MLX's conversion adds the 1 to the model's —
   * `check`, the original's final norm, says whether: its value as the model has it, or 1 below */
  private def shifted(check: Arr, model: Arr)(using Scope): Boolean =
    def diff(a: Arr) =
      val d = subtract(astype(a, Dtype.Float32), astype(model, Dtype.Float32))
      maxAll(maximum(d, multiply(d, scalar(-1f))))
    diff(add(astype(check, Dtype.Float32), scalar(1f))) < diff(check)

  private def maxAll(a: Arr)(using Scope): Float = max(reshape(a, -1), 0).eval().itemFloat

  /** the pack from the original's `mtp.*` tensors (`source`: bf16, and the original's final norm as `check.norm`),
   * for the model in `modelDir`, written to `out` */
  def convert(source: Path, modelDir: Path, out: Path): Unit =
    Mlx.init()
    val config = Qwen35Config(Json.fromText(Files.readString(modelDir.resolve("config.json"))))
    val sc = new Scope
    given Scope = sc
    try
      val src = Mlx.loadSafetensors(source.toString)
      val modelNorm = SafeTensors.shards(modelDir).iterator
        .map(p => SafeTensors.locate(p, inPlace = false).get("language_model.model.norm.weight").map(_ => p))
        .collectFirst { case Some(p) => Mlx.loadSafetensors(p.toString)("language_model.model.norm.weight") }
        .getOrElse(sys.error(s"no language_model.model.norm.weight in $modelDir"))
      val shift = src.get("check.norm").forall(shifted(_, modelNorm))
      val dtype = modelNorm.dtype
      def norm(a: Arr): Arr = if shift then astype(add(astype(a, Dtype.Float32), scalar(1f)), dtype) else astype(a,
        dtype)
      def quantized(prefix: String, a: Arr): Seq[(String, Arr)] =
        val (gs, bits) = quantOf(prefix, config)
        val (q, s, b) = quantize(astype(a, dtype), gs, bits)
        Seq(s"$prefix.weight" -> q, s"$prefix.scales" -> s, s"$prefix.biases" -> b)
      val tensors = src.toSeq.filter(_._1.startsWith("mtp.")).flatMap { (k, a) =>
        // the norms: each layer's, the last, and the two before `fc` (pre_fc_norm_embedding, pre_fc_norm_hidden)
        if a.ndim == 1 && k.contains("norm") then Seq(k -> norm(a))
        else if k.endsWith(".mlp.experts.gate_up_proj") then
          // [experts, 2·inner, hidden]: the gate's rows, then the up's
          val Seq(e, two, h) = a.shape: @unchecked
          val p = k.stripSuffix("experts.gate_up_proj") + "switch_mlp"
          quantized(s"$p.gate_proj", slice(a, Seq(0, 0, 0), Seq(e, two / 2, h))) ++
            quantized(s"$p.up_proj", slice(a, Seq(0, two / 2, 0), Seq(e, two, h)))
        else if k.endsWith(".mlp.experts.down_proj") then
          quantized(k.stripSuffix("experts.down_proj") + "switch_mlp.down_proj", a)
        else if k.endsWith(".weight") && a.ndim == 2 then quantized(k.stripSuffix(".weight"), a)
        else Seq(k -> astype(a, dtype))
      }
      require(tensors.exists(_._1 == "mtp.fc.weight"), s"no mtp.fc in $source")
      evalAll(tensors.map(_._2))
      Files.createDirectories(out.toAbsolutePath.getParent)
      // MLX adds .safetensors to a name without it
      val tmp = out.resolveSibling(out.getFileName.toString.stripSuffix(".safetensors") + ".new.safetensors")
      Mlx.saveSafetensors(tmp.toString, tensors, Map("format" -> "mlx", "norms_shifted" -> shift.toString))
      Files.move(tmp, out, java.nio.file.StandardCopyOption.REPLACE_EXISTING)
    finally sc.close()
