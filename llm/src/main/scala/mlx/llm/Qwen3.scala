package mlx.llm

import okay.effects.json.Json

import java.nio.file.{Files, Path}
import mlx.*
import mlx.Mlx.*

/** Qwen3's shape, from its config.json */
final case class Qwen3Config(
    hidden: Int, layers: Int, heads: Int, kvHeads: Int, headDim: Int, ropeTheta: Float, eps: Float,
    groupSize: Int, bits: Int, eos: Int)

object Qwen3Config:
  def apply(json: Json): Qwen3Config =
    require(json("model_type").string == "qwen3", s"a qwen3 model expected: ${json("model_type")}")
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

  /** the cache as it is: handles of its own on the same buffers (immutable: the next update copies, not this) */
  def snapshot(): KvCache.Snap =
    val sc = new Scope
    KvCache.Snap(sc, if keys == null then null else keys.nn.retain(sc), if values == null then null else
      values.nn.retain(sc), capacity, offset, kvHeads, headDim)
  /** the cache a snapshot was */
  def restore(snap: KvCache.Snap): Unit =
    keys = if snap.keys == null then null else snap.keys.nn.retain(own)
    values = if snap.values == null then null else snap.values.nn.retain(own)
    capacity = snap.capacity
    offset = snap.offset

  def close(): Unit = own.close()

object KvCache:
  /** a cache kept: its buffers (bf16), how far they hold */
  final class Snap(private[llm] val scope: Scope, private[llm] val keys: Arr | Null,
      private[llm] val values: Arr | Null, private[llm] val capacity: Int, private[llm] val offset: Int, kvHeads: Int,
      headDim: Int):
    def bytes: Long = 2L * capacity * kvHeads * headDim * 2
    def close(): Unit = scope.close()
    /** the buffers cut to what they hold, named after layer `i` */
    def tensors(i: Int)(using Scope): Vector[(String, Arr)] =
      if keys == null then Vector.empty
      else
        val stop = Seq(1, kvHeads, offset, headDim)
        Vector(s"$i.k" -> slice(keys.nn, Seq(0, 0, 0, 0), stop), s"$i.v" -> slice(values.nn, Seq(0, 0, 0, 0), stop))
    def meta(i: Int): Map[String, String] = Map(s"$i.offset" -> offset.toString)

  /** a snapshot of layer `i` from a file's tensors (they hold `offset` positions, and no more room) */
  def snapOf(i: Int, t: Map[String, Arr], meta: Map[String, String], kvHeads: Int, headDim: Int): Snap =
    val offset = meta.get(s"$i.offset").map(_.toInt).getOrElse(0)
    Snap(new Scope, t.get(s"$i.k").orNull, t.get(s"$i.v").orNull, offset, offset, kvHeads, headDim)

/** QWEN3: a decoder of `layers` blocks — attention with GQA and q/k RMS norms, RoPE; a SwiGLU MLP — over quantized
 * weights; the embedding is the output layer too (tied) */
final class Qwen3 private (val config: Qwen3Config, weights: Map[String, Arr], own: Scope) extends Model:
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

  /** a KV cache a layer */
  final class State(val caches: Vector[KvCache]) extends Model.State:
    def close(): Unit = caches.foreach(_.close())
    def snapshot(): Model.Snapshot = Snapshot(caches.map(_.snapshot()))
  private final class Snapshot(val snaps: Vector[KvCache.Snap], owner: Scope | Null = null) extends Model.Snapshot:
    def bytes: Long = snaps.map(_.bytes).sum
    def close(): Unit = { snaps.foreach(_.close()); if owner != null then owner.nn.close() }
    def tensors(using Scope): Vector[(String, Arr)] = snaps.zipWithIndex.flatMap((sn, i) => sn.tensors(i))
    def meta: Map[String, String] = snaps.zipWithIndex.flatMap((sn, i) => sn.meta(i)).toMap
  def snapshotOf(tensors: Map[String, Arr], meta: Map[String, String], owner: Scope): Model.Snapshot =
    Snapshot(Vector.tabulate(layers)(i => KvCache.snapOf(i, tensors, meta, kvHeads, headDim)), owner)
  def newState(): State = State(Vector.fill(layers)(KvCache(kvHeads, headDim)))
  def restore(snapshot: Model.Snapshot): State =
    val st = newState()
    st.caches.zip(snapshot.asInstanceOf[Snapshot].snaps).foreach((c, sn) => c.restore(sn))
    st
  def logits(ids: Arr, state: Model.State)(using Scope): Arr = logits(ids, state.asInstanceOf[State].caches)

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
  /** the model in a directory: config.json and its .safetensors shards — read, or mapped (`SafeTensors`) */
  def load(dir: Path, mapped: Boolean = SafeTensors.mappedByDefault): Qwen3 =
    Mlx.init()
    val config = Qwen3Config(Json.fromText(Files.readString(dir.resolve("config.json"))))
    val own = new Scope
    given Scope = own
    val weights = SafeTensors.shards(dir).flatMap(SafeTensors.load(_, mapped)).toMap
    evalAll(weights.values.toSeq)
    new Qwen3(config, weights, own)
