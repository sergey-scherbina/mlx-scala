package mlx.llm

import okay.effects.json.Json

import java.nio.file.{Files, Path, StandardCopyOption}
import mlx.*
import mlx.Mlx.*

/** A LORA ADAPTER MERGED INTO A QWEN3 MODEL'S WEIGHTS: each adapted projection W + scale · (A B)ᵀ, then every linear
 * layer quantized again at one width (`bits`, group 64 — one quantization for all, as any MLX loader reads it) — a
 * model directory of its own, the tokenizer's files copied: for an engine that knows no adapters (rozum's gateway
 * loads its embedding model by name, `ROZUM_EMBED_MODEL`) */
object LoraMerge:
  private val projs = Map("q" -> "self_attn.q_proj", "k" -> "self_attn.k_proj", "v" -> "self_attn.v_proj",
    "o" -> "self_attn.o_proj", "gate" -> "mlp.gate_proj", "up" -> "mlp.up_proj", "down" -> "mlp.down_proj")

  def qwen3(dir: Path, adapter: Path, out: Path, bits: Int = 4, groupSize: Int = 64): Unit =
    Mlx.init()
    val sc = new Scope
    given Scope = sc
    try
      val config = Json.fromText(Files.readString(dir.resolve("config.json")))
      val q = config("quantization")
      val (gs0, bits0) = (q("group_size").int, q("bits").int)
      val w = SafeTensors.shards(dir).flatMap(p => Mlx.loadSafetensors(p.toString)).toMap
      val a = Mlx.loadSafetensors(adapter.toString)
      val scale = a("lora.scale").itemFloat
      // the adapted projections' deltas, by weight name
      val deltas = a.keys.filter(_.endsWith(".a")).map(_.stripSuffix(".a")).map { n =>
        val Array(i, p) = n.split('.')
        val delta = transpose(multiply(matmul(a(s"$n.a"), a(s"$n.b")), scalar(scale)), 1, 0) // [out, in]
        s"model.layers.$i.${projs(p)}" -> delta
      }.toMap
      val prefixes = w.keys.filter(_.endsWith(".scales")).map(_.stripSuffix(".scales")).toVector.sorted
      val requantized = prefixes.flatMap { p =>
        val dense0 = astype(dequantize(w(s"$p.weight"), w(s"$p.scales"), w(s"$p.biases"), gs0, bits0), Dtype.Float32)
        val dense = deltas.get(p).fold(dense0)(d => add(dense0, astype(d, Dtype.Float32)))
        val (qw, s, b) = quantize(astype(dense, Dtype.BFloat16), groupSize, bits)
        val r = Seq(s"$p.weight" -> qw, s"$p.scales" -> s, s"$p.biases" -> b)
        evalAll(r.map(_._2))
        r
      }
      val rest = w.filterNot((k, _) => prefixes.exists(p => k.startsWith(p + "."))).toSeq
      require(deltas.keys.forall(prefixes.contains), s"adapted projections not in the model: ${deltas.keys}")
      Files.createDirectories(out)
      for f <- Files.list(dir).toArray.map(_.asInstanceOf[Path]) if !f.toString.endsWith(".safetensors") &&
          f.getFileName.toString != "config.json" && f.getFileName.toString != "model.safetensors.index.json" do
        Files.copy(f, out.resolve(f.getFileName.toString), StandardCopyOption.REPLACE_EXISTING)
      val quant = Json.obj("group_size" -> Json.Num(groupSize), "bits" -> Json.Num(bits))
      val cfg = config.record.map {
        case ("quantization", _) => "quantization" -> quant
        case ("quantization_config", _) => "quantization_config" -> quant
        case kv => kv
      }
      Files.writeString(out.resolve("config.json"), Json.Obj(cfg).render)
      Mlx.saveSafetensors(out.resolve("model.safetensors").toString, requantized ++ rest, Map("format" -> "mlx"))
    finally sc.close()
