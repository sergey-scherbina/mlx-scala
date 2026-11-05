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
  // DENSE (`densify`): the weights unpacked once [out, in] — on the CPU, MLX has no fast kernel for quantized
  // products (Qwen3-Embedding-0.6B: minutes a text), a plain matmul is BLAS's
  private var dense: Arr | Null = null
  def apply(x: Arr)(using Scope): Arr =
    if dense == null then quantizedMatmul(x, w, scales, biases, transpose = true, groupSize, bits)
    else matmul(astype(x, dense.nn.dtype), transpose(dense.nn, 1, 0))
  /** its input and output widths (the weights packed `32 / bits` to a 32-bit word) */
  def inDim: Int = w.dim(1) * (32 / bits)
  def outDim: Int = w.dim(0)
  /** rows of the weights, dequantized: an embedding's lookup */
  def rows(ids: Arr)(using Scope): Arr =
    if dense != null then take(dense.nn, ids, 0)
    else dequantize(take(w, ids, 0), take(scales, ids, 0), take(biases, ids, 0), groupSize, bits)
  /** the weights unpacked to `dtype`, kept in `owner`: products by a plain matmul from now on */
  def densify(dtype: Dtype, owner: Scope): Unit =
    if dense == null then
      dense = Scope(owner)(astype(dequantize(w, scales, biases, groupSize, bits), dtype).eval())

/** A LAYER'S KV CACHE: keys and values of the positions so far, in buffers grown 256 positions at a time; the new
 * positions written in place (`slice_update`), the positions so far a view */
final class KvCache(kvHeads: Int, headDim: Int) extends Model.Draft:
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
final class Qwen3 private (val config: Qwen3Config, weights: Map[String, Arr], own: Scope) extends Adaptable:
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
  /** after each layer, the residual stream [1, n, hidden] to a hook: read (a direction measured) or changed (steered);
   * none by default */
  var hook: Qwen3.Hook | Null = null
  /** LoRA adapters: `"<layer>.<proj>"` (proj q, k, v, o, gate, up, down) → (A [in, r], B [r, out]); a projection
   * with one gives `W x + scale · (x A) B`. Set by training (the adapters its inputs), or loaded */
  var lora: Map[String, (Arr, Arr)] = Map.empty
  var loraScale: Float = 1f
  private def adapted(i: Int, proj: String, base: QLinear, x: Arr)(using Scope): Arr =
    lora.get(s"$i.$proj") match
      case None => base(x)
      case Some((a, b)) =>
        val t = x.dtype
        add(base(x), multiply(matmul(matmul(x, astype(a, t)), astype(b, t)), astype(scalar(loraScale), t)))
  /** a projection's input and output widths: what an adapter on it is shaped by */
  def layerCount: Int = layers
  def widths(i: Int, proj: String): Option[(Int, Int)] =
    val l = blocks(i)
    val p: QLinear | Null = proj match
      case "q" => l.q
      case "k" => l.k
      case "v" => l.v
      case "o" => l.o
      case "gate" => l.gate
      case "up" => l.up
      case "down" => l.down
      case _ => null
    if p == null then None else Some((p.nn.inDim, p.nn.outDim))
  private val embed = lin("model.embed_tokens")
  private val norm = w("model.norm.weight")
  private val scale = (1.0 / math.sqrt(headDim)).toFloat

  /** a KV cache a layer */
  final class State(val caches: Vector[KvCache]) extends Model.State:
    def close(): Unit = caches.foreach(_.close())
    def snapshot(): Model.Snapshot = Snapshot(caches.map(_.snapshot()))
    // a cache of keys and values goes back to any place: its offset
    def checkpoint(): Model.Checkpoint = Qwen3.Offsets(caches.map(_.offset))
    def rewind(c: Model.Checkpoint, keep: Int): Boolean =
      caches.zip(c.asInstanceOf[Qwen3.Offsets].at).foreach((cache, o) => cache.offset = o + keep)
      true
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
  def logitsAll(ids: Arr, state: Model.State)(using Scope): Arr =
    logits(ids, state.asInstanceOf[State].caches, all = true)

  /** while training, attention by ops: MLX's fused kernel has no gradient for a short query ("the vector kernels do
   * not produce a logsumexp") — a query head's group of `heads / kvHeads` on its key head, the future masked */
  private var differentiable = false
  override def training(on: Boolean): Unit = differentiable = on
  private def attentionByOps(q: Arr, keys: Arr, values: Arr, n: Int, offset: Int)(using Scope): Arr =
    val rep = config.heads / kvHeads
    val t = keys.dim(2)
    val qg = reshape(q, 1, kvHeads, rep * n, headDim)
    val s = astype(multiply(matmul(qg, transpose(keys, 0, 1, 3, 2)), astype(scalar(scale), q.dtype)), Dtype.Float32)
    val mask = Array.tabulate(rep * n * t) { k =>
      val (row, j) = (k / t, k % t)
      if j <= offset + row % n then 0f else -1e9f
    }
    val p = astype(softmax(add(s, floats(mask, rep * n, t)), -1), q.dtype)
    reshape(matmul(p, values), 1, config.heads, n, headDim)

  private def attention(i: Int, l: Layer, x: Arr, n: Int, cache: KvCache)(using Scope): Arr =
    val offset = cache.offset
    def split(t: Arr, count: Int) = transpose(reshape(t, 1, n, count, headDim), 0, 2, 1, 3)
    val qx = adapted(i, "q", l.q, x)
    val kx = adapted(i, "k", l.k, x)
    val q = rope(transpose(rmsNorm(reshape(qx, 1, n, config.heads, headDim), l.qNorm, eps), 0, 2, 1, 3),
      headDim, false, ropeTheta, 1f, offset)
    val k = rope(transpose(rmsNorm(reshape(kx, 1, n, kvHeads, headDim), l.kNorm, eps), 0, 2, 1, 3),
      headDim, false, ropeTheta, 1f, offset)
    val (keys, values) = cache.update(k, split(adapted(i, "v", l.v, x), kvHeads))
    val a = if differentiable then attentionByOps(q, keys, values, n, offset)
      else Mlx.attention(q, keys, values, scale, causal = n > 1)
    adapted(i, "o", l.o, reshape(transpose(a, 0, 2, 1, 3), 1, n, config.heads * headDim))

  /** the next token's logits after `ids` [1, n], the cache moved on by n: the last position's alone.
   *
   * Each layer in a scope of its own, its output alone moved out: a handle held keeps MLX from freeing its buffer
   * once used — with every intermediate of a pass held, a 4096-token prompt peaked at 48.7 GB */
  def logits(ids: Arr, cache: Vector[KvCache], all: Boolean = false)(using s: Scope): Arr =
    val h = forward(ids, cache, all)
    val r = Scope(s)(embed(h))
    s.release(h)
    r

  override def hiddenStates(ids: Arr, state: Model.State, all: Boolean)(using Scope): Arr =
    forward(ids, state.asInstanceOf[State].caches, all)
  override def head(h: Arr)(using Scope): Arr = embed(h)

  /** the hidden states after the last norm, of the last of `ids` [1, n] or of each */
  private def forward(ids: Arr, cache: Vector[KvCache], all: Boolean)(using s: Scope): Arr =
    val n = ids.dim(1)
    var h = Scope(s)(reshape(embed.rows(reshape(ids, n)), 1, n, hidden))
    var i = 0
    while i < layers do
      val l = blocks(i)
      val x = h
      h = Scope(s) {
        val a = add(x, attention(i, l, rmsNorm(x, l.inNorm, eps), n, cache(i)))
        val m = rmsNorm(a, l.postNorm, eps)
        val out = add(a, adapted(i, "down", l.down,
          multiply(silu(adapted(i, "gate", l.gate, m)), adapted(i, "up", l.up, m))))
        if hook == null then out else hook.nn.after(i, out)
      }
      s.release(x)
      i += 1
    Scope(s) {
      val last = if n == 1 || all then h else slice(h, Seq(0, n - 1, 0), Seq(1, n, hidden))
      rmsNorm(last, norm, eps)
    }

  /** every projection unpacked once (`QLinear.densify`): for the CPU, where quantized products are slow */
  def densify(dtype: Dtype = Dtype.Float32): Unit =
    (embed +: blocks.flatMap(l => Seq(l.q, l.k, l.v, l.o, l.gate, l.up, l.down))).foreach(_.densify(dtype, own))

  /** its weights freed, and MLX's buffers kept for reuse given back: the memory free for another model */
  def close(): Unit =
    own.close()
    Mlx.clearCache()

object Qwen3:
  /** a hook on the residual stream: after layer `layer`, `h` [1, n, hidden] — what to go on with */
  trait Hook:
    def after(layer: Int, h: Arr)(using Scope): Arr
  private final case class Offsets(at: Vector[Int]) extends Model.Checkpoint:
    def close(): Unit = ()
  /** the model in a directory: config.json and its .safetensors shards — read, or mapped (`SafeTensors`) */
  def load(dir: Path, mapped: Boolean = SafeTensors.mappedByDefault,
      inPlace: Boolean = SafeTensors.alignInPlace): Qwen3 =
    Mlx.init()
    val config = Qwen3Config(Json.fromText(Files.readString(dir.resolve("config.json"))))
    val own = new Scope
    given Scope = own
    val weights = SafeTensors.shards(dir).flatMap(SafeTensors.load(_, mapped, inPlace)).toMap
    evalAll(weights.values.toSeq)
    new Qwen3(config, weights, own)
