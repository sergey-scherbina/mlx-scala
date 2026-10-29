package mlx.llm

import java.nio.file.{Files, Path}
import mlx.*
import mlx.Mlx.*

/** Qwen3's shape, from its config.json */
final case class Qwen3Config(
    hidden: Int, layers: Int, heads: Int, kvHeads: Int, headDim: Int, ropeTheta: Float, eps: Float,
    groupSize: Int, bits: Int, eos: Int)

object Qwen3Config:
  def apply(json: Json): Qwen3Config =
    require(json("model_type").str == "qwen3", s"a qwen3 model expected: ${json("model_type")}")
    val q = json("quantization")
    Qwen3Config(json("hidden_size").int, json("num_hidden_layers").int, json("num_attention_heads").int,
      json("num_key_value_heads").int, json("head_dim").int, json("rope_theta").double.toFloat,
      json("rms_norm_eps").double.toFloat, q("group_size").int, q("bits").int, json("eos_token_id").int)

/** a linear layer with quantized weights: x · wᵀ */
final class QLinear(w: Arr, scales: Arr, biases: Arr, groupSize: Int, bits: Int):
  def apply(x: Arr)(using Scope): Arr = quantizedMatmul(x, w, scales, biases, transpose = true, groupSize, bits)
  /** rows of the weights, dequantized: an embedding's lookup */
  def rows(ids: Arr)(using Scope): Arr =
    dequantize(take(w, ids, 0), take(scales, ids, 0), take(biases, ids, 0), groupSize, bits)

/** A LAYER'S KV CACHE: keys and values of the positions so far, in buffers grown 256 positions at a time; the new
 * positions written in place (`slice_update`), the positions so far a view */
final class KvCache(kvHeads: Int, headDim: Int):
  private val own = new Scope
  private var keys: Arr | Null = null
  private var values: Arr | Null = null
  private var capacity = 0
  var offset = 0

  def update(k: Arr, v: Arr)(using s: Scope): (Arr, Arr) =
    val n = k.dim(2)
    if offset + n > capacity then grow(offset + n, k.dtype)
    val end = offset + n
    val start = Seq(0, 0, offset, 0)
    val stop = Seq(1, kvHeads, end, headDim)
    val (oldK, oldV) = (keys.nn, values.nn)
    keys = sliceUpdate(oldK, k, start, stop).moveTo(own)
    values = sliceUpdate(oldV, v, start, stop).moveTo(own)
    own.release(oldK)
    own.release(oldV)
    offset = end
    val all = Seq(1, kvHeads, end, headDim)
    (slice(keys.nn, Seq(0, 0, 0, 0), all), slice(values.nn, Seq(0, 0, 0, 0), all))

  private def grow(need: Int, dtype: Dtype)(using s: Scope): Unit =
    val cap = (need + 255) / 256 * 256
    val shape = Seq(1, kvHeads, cap, headDim)
    def bigger(old: Arr | Null): Arr =
      val z = zeros(shape, dtype)
      val a = if old == null then z
        else sliceUpdate(z, slice(old.nn, Seq(0, 0, 0, 0), Seq(1, kvHeads, offset, headDim)), Seq(0, 0, 0, 0),
          Seq(1, kvHeads, offset, headDim))
      a.moveTo(own)
      if old != null then own.release(old.nn)
      a
    keys = bigger(keys)
    values = bigger(values)
    capacity = cap

  def close(): Unit = own.close()

/** QWEN3: a decoder of `layers` blocks — attention with GQA and q/k RMS norms, RoPE; a SwiGLU MLP — over quantized
 * weights; the embedding is the output layer too (tied) */
final class Qwen3 private (val config: Qwen3Config, weights: Map[String, Arr], own: Scope):
  import config.*
  private def w(name: String): Arr = weights.getOrElse(name, sys.error(s"no weight $name"))
  private def lin(prefix: String) =
    QLinear(w(s"$prefix.weight"), w(s"$prefix.scales"), w(s"$prefix.biases"), groupSize, bits)

  private final class Layer(i: Int):
    private val p = s"model.layers.$i"
    val inNorm: Arr = w(s"$p.input_layernorm.weight")
    val postNorm: Arr = w(s"$p.post_attention_layernorm.weight")
    val q: QLinear = lin(s"$p.self_attn.q_proj")
    val k: QLinear = lin(s"$p.self_attn.k_proj")
    val v: QLinear = lin(s"$p.self_attn.v_proj")
    val o: QLinear = lin(s"$p.self_attn.o_proj")
    val qNorm: Arr = w(s"$p.self_attn.q_norm.weight")
    val kNorm: Arr = w(s"$p.self_attn.k_norm.weight")
    val gate: QLinear = lin(s"$p.mlp.gate_proj")
    val up: QLinear = lin(s"$p.mlp.up_proj")
    val down: QLinear = lin(s"$p.mlp.down_proj")

  private val blocks = Vector.tabulate(layers)(Layer(_))
  private val embed = lin("model.embed_tokens")
  private val norm = w("model.norm.weight")
  private val scale = (1.0 / math.sqrt(headDim)).toFloat

  def newCache(): Vector[KvCache] = Vector.fill(layers)(KvCache(kvHeads, headDim))

  private def attention(l: Layer, x: Arr, n: Int, cache: KvCache)(using Scope): Arr =
    val offset = cache.offset
    def split(t: Arr, count: Int) = transpose(reshape(t, 1, n, count, headDim), 0, 2, 1, 3)
    val q = rope(transpose(rmsNorm(reshape(l.q(x), 1, n, config.heads, headDim), l.qNorm, eps), 0, 2, 1, 3),
      headDim, false, ropeTheta, 1f, offset)
    val k = rope(transpose(rmsNorm(reshape(l.k(x), 1, n, kvHeads, headDim), l.kNorm, eps), 0, 2, 1, 3),
      headDim, false, ropeTheta, 1f, offset)
    val (keys, values) = cache.update(k, split(l.v(x), kvHeads))
    val a = Mlx.attention(q, keys, values, scale, causal = n > 1)
    l.o(reshape(transpose(a, 0, 2, 1, 3), 1, n, config.heads * headDim))

  /** the next token's logits after `ids` [1, n], the cache moved on by n: the last position's alone.
   *
   * Each layer in a scope of its own, its output alone moved out: a handle held keeps MLX from freeing its buffer
   * once used — with every intermediate of a pass held, a 4096-token prompt peaked at 48.7 GB */
  def logits(ids: Arr, cache: Vector[KvCache])(using s: Scope): Arr =
    val n = ids.dim(1)
    var h = Scope(s)(reshape(embed.rows(reshape(ids, n)), 1, n, hidden))
    var i = 0
    while i < layers do
      val l = blocks(i)
      val x = h
      h = Scope(s) {
        val a = add(x, attention(l, rmsNorm(x, l.inNorm, eps), n, cache(i)))
        val m = rmsNorm(a, l.postNorm, eps)
        add(a, l.down(multiply(silu(l.gate(m)), l.up(m))))
      }
      s.release(x)
      i += 1
    Scope(s) {
      val last = if n == 1 then h else slice(h, Seq(0, n - 1, 0), Seq(1, n, hidden))
      embed(rmsNorm(last, norm, eps))
    }

  def close(): Unit = own.close()

object Qwen3:
  /** the model in a directory: config.json and its .safetensors shards */
  def load(dir: Path): Qwen3 =
    Mlx.init()
    val config = Qwen3Config(Json.parse(Files.readString(dir.resolve("config.json"))))
    val own = new Scope
    given Scope = own
    val shards = Files.list(dir).toArray.map(_.asInstanceOf[Path]).filter(_.toString.endsWith(".safetensors"))
    val weights = shards.sorted.flatMap(p => loadSafetensors(p.toString)).toMap
    evalAll(weights.values.toSeq)
    new Qwen3(config, weights, own)
