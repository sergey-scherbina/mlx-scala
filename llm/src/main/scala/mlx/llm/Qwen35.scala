package mlx.llm

import okay.effects.json.Json

import java.nio.file.{Files, Path}
import mlx.*
import mlx.Mlx.*

/** Qwen3.5 / Qwen3.6's shape (config.json's text_config): which layers are linear, the MoE when there is one */
final case class Qwen35Config(
    hidden: Int, layers: Int, heads: Int, kvHeads: Int, headDim: Int, ropeDims: Int, ropeTheta: Float, eps: Float,
    interval: Int, linearKeyHeads: Int, linearValueHeads: Int, linearKeyDim: Int, linearValueDim: Int,
    convKernel: Int, experts: Int, expertsPerToken: Int, normTopK: Boolean,
    groupSize: Int, bits: Int, quantized: Map[String, (Int, Int)])

object Qwen35Config:
  def apply(json: Json): Qwen35Config =
    val t = json.field("text_config").getOrElse(json)
    val rope = t.field("rope_parameters").getOrElse(t)
    val q = json("quantization")
    val overrides = q.record.collect { case (k, o: Json.Obj) => k -> (o("group_size").int, o("bits").int) }.toMap
    val headDim = t("head_dim").int
    val partial = rope.field("partial_rotary_factor").orElse(t.field("partial_rotary_factor")).map(_.double)
      .getOrElse(0.25)
    Qwen35Config(t("hidden_size").int, t("num_hidden_layers").int, t("num_attention_heads").int,
      t("num_key_value_heads").int, headDim, (headDim * partial).toInt,
      rope.field("rope_theta").orElse(t.field("rope_theta")).map(_.double).getOrElse(1e7).toFloat,
      t("rms_norm_eps").double.toFloat, t.field("full_attention_interval").map(_.int).getOrElse(4),
      t("linear_num_key_heads").int, t("linear_num_value_heads").int, t("linear_key_head_dim").int,
      t("linear_value_head_dim").int, t.field("linear_conv_kernel_dim").map(_.int).getOrElse(4),
      t.field("num_experts").map(_.int).getOrElse(0), t.field("num_experts_per_tok").map(_.int).getOrElse(0),
      t.field("norm_topk_prob").forall(_.boolean), q("group_size").int, q("bits").int, overrides)

/** A LINEAR LAYER'S RECURRENT STATE: the convolution's last inputs, the delta rule's state */
final class LinearState:
  private val own = new Scope
  var conv: Arr | Null = null
  var state: Arr | Null = null
  def keep(conv: Arr, state: Arr): Unit =
    val (c0, s0) = (this.conv, this.state)
    this.conv = conv.moveTo(own)
    this.state = state.moveTo(own)
    if c0 != null then own.release(c0.nn)
    if s0 != null then own.release(s0.nn)
  def snapshot(): LinearState.Snap =
    val sc = new Scope
    LinearState.Snap(sc, if conv == null then null else conv.nn.retain(sc),
      if state == null then null else state.nn.retain(sc))
  def restore(snap: LinearState.Snap): Unit =
    conv = if snap.conv == null then null else snap.conv.nn.retain(own)
    state = if snap.state == null then null else snap.state.nn.retain(own)
  def close(): Unit = own.close()

object LinearState:
  final class Snap(private[llm] val scope: Scope, private[llm] val conv: Arr | Null,
      private[llm] val state: Arr | Null, val bytes: Long = 0L):
    def close(): Unit = scope.close()
    def tensors(i: Int): Vector[(String, Arr)] =
      (if conv == null then Vector.empty else Vector(s"$i.conv" -> conv.nn)) ++
        (if state == null then Vector.empty else Vector(s"$i.state" -> state.nn))

/** QWEN3.5 AND QWEN3.6: a hybrid decoder — three layers in four linear attention (Gated DeltaNet: a causal
 * convolution, then the gated delta rule's recurrence, a Metal kernel), the fourth full attention with an output
 * gate and partial RoPE; its MLP dense, or a mixture of experts (top-k of many, and a shared expert) — over quantized
 * weights. As MLX's mlx_lm (qwen3_5.py, qwen3_next.py, gated_delta.py) computes it */
final class Qwen35 private (val config: Qwen35Config, weights: Map[String, Arr], own: Scope,
    hotExperts: Map[String, ExpertCache]) extends Model:
  import config.*
  private val P = "language_model.model."
  private def w(name: String): Arr = weights.getOrElse(name, sys.error(s"no weight $name"))
  private def lin(prefix: String): QLinear =
    val (gs, b) = quantized.getOrElse(prefix, (groupSize, bits))
    QLinear(w(s"$prefix.weight"), w(s"$prefix.scales"), w(s"$prefix.biases"), gs, b)

  /** a layer's experts: all of them, or a cache of the hot ones (`ExpertCache`: its buffers in their place) */
  private final class Experts(prefix: String):
    private val cache = hotExperts.get(prefix).orNull
    private type Q = (Arr, Arr, Arr, Int, Int)
    private def q(n: String) =
      val p = s"$prefix.$n"
      val (gs, b) = quantized.getOrElse(p, (groupSize, bits))
      (w(s"$p.weight"), w(s"$p.scales"), w(s"$p.biases"), gs, b)
    private val all: (Q, Q, Q) = (q("gate_proj"), q("up_proj"), q("down_proj"))
    // the cache's slots in their place: the same quantization, `slots` experts
    private val hot: (Q, Q, Q) | Null =
      if cache == null then null
      else
        def s(t: Q, at: Int): Q = (cache.buffers(at), cache.buffers(at + 1), cache.buffers(at + 2), t._4, t._5)
        (s(all._1, 0), s(all._2, 3), s(all._3, 6))
    private def glu(x: Arr, idx: Arr, sorted: Boolean, on: (Q, Q, Q) = all)(using Scope) =
      def at(t: Q, v: Arr) = gatherQmm(v, t._1, t._2, t._3, idx, transpose = true, t._4, t._5, sorted)
      at(on._3, swiglu(at(on._1, x), at(on._2, x)))
    /** x [1, n, 1, 1, D], idx [1, n, k]: [1, n, k, D]. With many rows (n·k ≥ 64) they go sorted by expert, each
     * expert's weights read once — mlx_lm's `_gather_sort` */
    /** a step's few picks (n·k < 64) on the hot experts, when there is a cache: read to the CPU, the missing ones
     * read in, the picks mapped to their slots. A prompt's many on all of them (mapped: the GPU reads the file's
     * pages, the OS's cache — a prompt uses nearly every expert, too many to copy) */
    def run(x: Arr, idx: Arr)(using Scope): Arr =
      val n = idx.dim(1)
      val k = idx.dim(2)
      if n * k < 64 then
        if hot == null then squeeze(glu(x, idx, sorted = false), -2)
        else
          val slots = cache.slotsOf(idx.toInts)
          squeeze(glu(x, astype(ints(slots, 1, n, k), Dtype.UInt32), sorted = false, hot.nn), -2)
      else
        val flat = reshape(idx, n * k)
        val order = argsort(flat, 0)
        val inverse = argsort(order, 0)
        val rows = take(reshape(x, n, 1, x.dim(-1)), floorDivide(order, astype(int(k), order.dtype)), 0)
        val y = glu(rows, take(flat, order, 0), sorted = true)
        squeeze(reshape(take(y, inverse, 0), 1, n, k, 1, y.dim(-1)), -2)

  private final class Layer(val i: Int):
    private val p = s"${P}layers.$i"
    val linear: Boolean = (i + 1) % interval != 0
    val inNorm: Arr = w(s"$p.input_layernorm.weight")
    val postNorm: Arr = w(s"$p.post_attention_layernorm.weight")
    // full attention
    lazy val q: QLinear = lin(s"$p.self_attn.q_proj")
    lazy val k: QLinear = lin(s"$p.self_attn.k_proj")
    lazy val v: QLinear = lin(s"$p.self_attn.v_proj")
    lazy val o: QLinear = lin(s"$p.self_attn.o_proj")
    lazy val qNorm: Arr = w(s"$p.self_attn.q_norm.weight")
    lazy val kNorm: Arr = w(s"$p.self_attn.k_norm.weight")
    // linear attention
    lazy val qkv: QLinear = lin(s"$p.linear_attn.in_proj_qkv")
    lazy val z: QLinear = lin(s"$p.linear_attn.in_proj_z")
    lazy val a: QLinear = lin(s"$p.linear_attn.in_proj_a")
    lazy val b: QLinear = lin(s"$p.linear_attn.in_proj_b")
    lazy val out: QLinear = lin(s"$p.linear_attn.out_proj")
    lazy val conv: Arr = w(s"$p.linear_attn.conv1d.weight")
    lazy val aLog: Arr = w(s"$p.linear_attn.A_log")
    lazy val dtBias: Arr = w(s"$p.linear_attn.dt_bias")
    lazy val gNorm: Arr = w(s"$p.linear_attn.norm.weight")
    // MLP
    lazy val gate: QLinear = lin(s"$p.mlp.gate_proj")
    lazy val up: QLinear = lin(s"$p.mlp.up_proj")
    lazy val down: QLinear = lin(s"$p.mlp.down_proj")
    lazy val router: QLinear = lin(s"$p.mlp.gate")
    lazy val experts: Experts = Experts(s"$p.mlp.switch_mlp")
    lazy val sharedGate: QLinear = lin(s"$p.mlp.shared_expert.gate_proj")
    lazy val sharedUp: QLinear = lin(s"$p.mlp.shared_expert.up_proj")
    lazy val sharedDown: QLinear = lin(s"$p.mlp.shared_expert.down_proj")
    lazy val sharedRouter: QLinear = lin(s"$p.mlp.shared_expert_gate")

  private val blocks = Vector.tabulate(layers)(Layer(_))
  private val embed = lin(s"${P}embed_tokens")
  private val head = if weights.contains("language_model.lm_head.weight") then lin("language_model.lm_head") else embed
  private val norm = w(s"${P}norm.weight")
  private val keyDim = linearKeyHeads * linearKeyDim
  private val valueDim = linearValueHeads * linearValueDim
  private val convDim = keyDim * 2 + valueDim

  /** a KV cache for a full attention layer, a recurrent state for a linear one */
  final class State(val caches: Vector[KvCache | LinearState]) extends Model.State:
    def close(): Unit = caches.foreach { case c: KvCache => c.close(); case l: LinearState => l.close() }
    def snapshot(): Model.Snapshot = Snapshot(caches.map { case c: KvCache => c.snapshot(); case l: LinearState =>
      l.snapshot() })
  private val linearBytes = 2L * (convKernel - 1) * (2 * linearKeyHeads * linearKeyDim + linearValueHeads *
    linearValueDim) + 4L * linearValueHeads * linearValueDim * linearKeyDim
  private final class Snapshot(val snaps: Vector[KvCache.Snap | LinearState.Snap], owner: Scope | Null = null)
      extends Model.Snapshot:
    def bytes: Long = snaps.map { case k: KvCache.Snap => k.bytes; case _: LinearState.Snap => linearBytes }.sum
    def close(): Unit =
      snaps.foreach { case k: KvCache.Snap => k.close(); case l: LinearState.Snap => l.close() }
      if owner != null then owner.nn.close()
    def tensors(using Scope): Vector[(String, Arr)] = snaps.zipWithIndex.flatMap {
      case (k: KvCache.Snap, i) => k.tensors(i)
      case (l: LinearState.Snap, i) => l.tensors(i)
    }
    def meta: Map[String, String] = snaps.zipWithIndex.collect { case (k: KvCache.Snap, i) => k.meta(i) }.flatten.toMap
  def snapshotOf(tensors: Map[String, Arr], meta: Map[String, String], owner: Scope): Model.Snapshot =
    Snapshot(blocks.zipWithIndex.map((l, i) =>
      if l.linear then LinearState.Snap(new Scope, tensors.get(s"$i.conv").orNull, tensors.get(s"$i.state").orNull)
      else KvCache.snapOf(i, tensors, meta, kvHeads, headDim)), owner)
  def restore(snapshot: Model.Snapshot): State =
    val st = newState()
    st.caches.zip(snapshot.asInstanceOf[Snapshot].snaps).foreach {
      case (c: KvCache, k: KvCache.Snap) => c.restore(k)
      case (l: LinearState, k: LinearState.Snap) => l.restore(k)
      case _ => sys.error("a snapshot of another model")
    }
    st
  def newState(): State = State(blocks.map(l =>
    if l.linear then LinearState() else KvCache(kvHeads, headDim)))

  private def attention(l: Layer, x: Arr, n: Int, cache: KvCache)(using Scope): Arr =
    val offset = cache.offset
    val qg = reshape(l.q(x), 1, n, heads, 2 * headDim)
    val q = slice(qg, Seq(0, 0, 0, 0), Seq(1, n, heads, headDim))
    val gate = reshape(slice(qg, Seq(0, 0, 0, headDim), Seq(1, n, heads, 2 * headDim)), 1, n, heads * headDim)
    val qs = rope(transpose(rmsNorm(q, l.qNorm, eps), 0, 2, 1, 3), ropeDims, false, ropeTheta, 1f, offset)
    val ks = rope(transpose(rmsNorm(reshape(l.k(x), 1, n, kvHeads, headDim), l.kNorm, eps), 0, 2, 1, 3),
      ropeDims, false, ropeTheta, 1f, offset)
    val vs = transpose(reshape(l.v(x), 1, n, kvHeads, headDim), 0, 2, 1, 3)
    val (keys, values) = cache.update(ks, vs)
    val a = Mlx.attention(qs, keys, values, (1.0 / math.sqrt(headDim)).toFloat, causal = n > 1)
    l.o(multiply(reshape(transpose(a, 0, 2, 1, 3), 1, n, heads * headDim), sigmoid(gate)))

  private val inv = (1.0 / math.sqrt(linearKeyDim)).toFloat
  /** the readout scales inv² and inv at an array type, made once: an `astype` a step is a kernel a step */
  private val scaleCache = scala.collection.mutable.HashMap.empty[Dtype, (Arr, Arr)]
  private def scales(dt: Dtype): (Arr, Arr) = scaleCache.getOrElseUpdate(dt, {
    given Scope = own
    val pair = (astype(scalar(inv * inv), dt).eval(), astype(scalar(inv), dt).eval())
    pair
  })

  private def linearAttention(l: Layer, x: Arr, n: Int, st: LinearState)(using Scope): Arr =
    val qkv = l.qkv(x)
    val z = reshape(l.z(x), 1, n, linearValueHeads, linearValueDim)
    val beta = sigmoid(l.b(x))
    // g = exp(-exp(A_log) · softplus(a + dt_bias)), in float32
    val g = computeG(l.aLog, l.a(x), l.dtBias)
    val past = if st.conv == null then zeros(Seq(1, convKernel - 1, convDim), x.dtype) else st.conv.nn
    val input = concatenate(Seq(past, qkv), 1)
    val keep = contiguous(slice(input, Seq(0, n, 0), Seq(1, n + convKernel - 1, convDim)))
    val conv = silu(conv1d(input, l.conv, groups = convDim))
    def part(from: Int, to: Int, h: Int, d: Int) = reshape(slice(conv, Seq(0, 0, from), Seq(1, n, to)), 1, n, h, d)
    // q and k L2-normalized, the readout scale folded into q; a scale is of the array's type (as Python's float is
    // to MLX: 128^-0.5 in bf16 is 0.0883789, not 0.0883883)
    val (invSq, invOne) = scales(x.dtype)
    def scaled(t: Arr, c: Arr) = multiply(rmsNorm(t, Mlx.none, 1e-6f * inv * inv), c)
    val q = scaled(part(0, keyDim, linearKeyHeads, linearKeyDim), invSq)
    val k = scaled(part(keyDim, 2 * keyDim, linearKeyHeads, linearKeyDim), invOne)
    val v = part(2 * keyDim, convDim, linearValueHeads, linearValueDim)
    val state = if st.state == null then zeros(Seq(1, linearValueHeads, linearValueDim, linearKeyDim), Dtype.Float32)
      else st.state.nn
    // the packed kernel where it fits (Dk 128: eight value rows a SIMD-group), as mlx_lm; else the generic one
    val packed = Qwen35.packedAllowed && linearKeyDim == 128 && linearValueDim % 8 == 0
    val (kernel, grid, group) =
      if packed then (Qwen35.deltaRulePacked, (32, linearValueDim / 8, linearValueHeads), (32, 2, 1))
      else (Qwen35.deltaRule, (32, linearValueDim, linearValueHeads), (32, 4, 1))
    val Vector(y, next) = kernel(Seq(q, k, v, g, beta, state, int(n)), Seq(Seq(1, n, linearValueHeads,
      linearValueDim), Seq(1, linearValueHeads, linearValueDim, linearKeyDim)), Seq(x.dtype, Dtype.Float32),
      grid, group, Seq("InT" -> x.dtype, "StT" -> Dtype.Float32, "Dk" -> linearKeyDim, "Dv" -> linearValueDim,
      "Hk" -> linearKeyHeads, "Hv" -> linearValueHeads)): @unchecked
    st.keep(keep, next)
    // the gated RMS norm: silu(z) · norm(y), in float32
    val gated = preciseSwiglu(y, z, rmsNorm(y, l.gNorm, eps))
    l.out(reshape(gated, 1, n, valueDim))

  private def mlp(l: Layer, x: Arr, n: Int)(using Scope): Arr =
    if experts == 0 then l.down(swiglu(l.gate(x), l.up(x)))
    else
      val gates = softmax(l.router(x), -1)
      val inds = slice(argpartition(gates, experts - expertsPerToken, -1), Seq(0, 0, experts - expertsPerToken),
        Seq(1, n, experts))
      val s0 = takeAlong(gates, inds, -1)
      val scores = if normTopK then divide(s0, sum(s0, -1, keepdims = true)) else s0
      val seen = Qwen35.routes
      if seen != null then seen(l.i, inds.toInts)
      val y = l.experts.run(expandDims(expandDims(x, -2), -2), inds)
      val mixed = sum(multiply(y, expandDims(astype(scores, y.dtype), -1)), -2)
      val shared = l.sharedDown(swiglu(l.sharedGate(x), l.sharedUp(x)))
      add(mixed, multiply(sigmoid(l.sharedRouter(x)), shared))

  /** each layer in a scope of its own, its output alone moved out (a handle held keeps its buffer) */
  def logits(ids: Arr, state: Model.State)(using s: Scope): Arr =
    val caches = state.asInstanceOf[State].caches
    val n = ids.dim(1)
    var h = Scope(s)(reshape(embed.rows(reshape(ids, n)), 1, n, hidden))
    var i = 0
    while i < layers do
      val l = blocks(i)
      val x = h
      h = Scope(s) {
        val normed = rmsNorm(x, l.inNorm, eps)
        val r = caches(i) match
          case c: KvCache => attention(l, normed, n, c)
          case st: LinearState => linearAttention(l, normed, n, st)
        val a = add(x, r)
        add(a, mlp(l, rmsNorm(a, l.postNorm, eps), n))
      }
      s.release(x)
      i += 1
    Scope(s) {
      val last = if n == 1 then h else slice(h, Seq(0, n - 1, 0), Seq(1, n, hidden))
      head(rmsNorm(last, norm, eps))
    }

  /** the hidden state's sum and sum of squares at the last position after each of the first `count` layers: a
   * comparison with another implementation */
  private[llm] def probe(ids: Arr, count: Int)(using s: Scope): Vector[(Float, Float)] =
    val caches = newState().caches
    val n = ids.dim(1)
    var h = reshape(embed.rows(reshape(ids, n)), 1, n, hidden)
    (0 until count).toVector.map { i =>
      val l = blocks(i)
      val normed = rmsNorm(h, l.inNorm, eps)
      val r = caches(i) match
        case c: KvCache => attention(l, normed, n, c)
        case st: LinearState => linearAttention(l, normed, n, st)
      val a = add(h, r)
      def stat(t: Arr) =
        val last = astype(slice(t, Seq(0, n - 1, 0), Seq(1, n, t.dim(-1))), Dtype.Float32)
        (sum(reshape(last, -1), 0).eval().itemFloat, sum(reshape(multiply(last, last), -1), 0).eval().itemFloat)
      h = add(a, mlp(l, rmsNorm(a, l.postNorm, eps), n))
      stat(h)
    }

  private[llm] def counts: (Long, Long) = (hotExperts.values.map(_.hits).sum, hotExperts.values.map(_.misses).sum)

  def close(): Unit =
    hotExperts.values.foreach(_.close())
    own.close()

object Qwen35:
  /** THE GATED DELTA RULE, a step a position: for each value head, state ← state·g; kv = state·k;
   * state += k ⊗ (v − kv)·β; y = state·q. MLX's mlx_lm kernel (gated_delta.py, MIT, Apple), unmasked, scalar gate */
  private[llm] val deltaRule = Kernel("okm_gated_delta_step", Seq("q", "k", "v", "g", "beta", "state_in", "T"),
    Seq("y", "state_out"), """
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        constexpr int n_per_t = Dk / 32;
        auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
        auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
        y += b_idx * T * Hv * Dv + hv_idx * Dv;
        auto dk_idx = thread_position_in_threadgroup.x;
        auto dv_idx = thread_position_in_grid.y;
        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;
        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;
        for (int t = 0; t < T; ++t) {
          float kv_mem = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] * g_[hv_idx];
            kv_mem += state[i] * k_[s_idx];
          }
          kv_mem = simd_sum(kv_mem);
          auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];
          float out = 0.0f;
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * dk_idx + i;
            state[i] = state[i] + k_[s_idx] * delta;
            out += state[i] * q_[s_idx];
          }
          out = simd_sum(out);
          if (thread_index_in_simdgroup == 0) {
            y[dv_idx] = static_cast<InT>(out);
          }
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * dk_idx + i;
          o_state[s_idx] = static_cast<StT>(state[i]);
        }
    """)

  /** MLX_SCALA_GDN=unpacked: the generic kernel everywhere (an A/B) */
  /** WHO SEES THE ROUTING: each MoE layer's chosen experts (layer, n·k ids), read as they are chosen — a probe's
   * (it waits for the GPU each layer); null: none */
  @volatile var routes: ((Int, Array[Int]) => Unit) | Null = null
  @volatile private[llm] var packedAllowed = !sys.env.get("MLX_SCALA_GDN").contains("unpacked")

  /** THE GATED DELTA RULE, Dk = 128: eight value rows packed in a SIMD-group, four lanes a row, each lane 32 state
   * elements in registers; the reductions in the generic kernel's order, so the numbers are its numbers. MLX's
   * mlx_lm (gated_delta.py, `_make_gated_delta_packed_kernel`, MIT, Apple) */
  private[llm] val deltaRulePacked = Kernel("okm_gated_delta_step_packed",
    Seq("q", "k", "v", "g", "beta", "state_in", "T"), Seq("y", "state_out"), """
        constexpr int lanes_per_row = 4;
        constexpr int rows_per_simdgroup = 32 / lanes_per_row;
        constexpr int values_per_lane = Dk / lanes_per_row;
        constexpr int partials_per_lane = values_per_lane / 4;
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto hk_idx = hv_idx / (Hv / Hk);
        auto lane = thread_index_in_simdgroup;
        auto row_in_simdgroup = lane / lanes_per_row;
        auto lane_in_row = lane & (lanes_per_row - 1);
        auto row_group = thread_position_in_grid.y;
        auto dv_idx = row_group * rows_per_simdgroup + row_in_simdgroup;
        auto q_ = q + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;
        auto k_ = k + (b_idx * T * Hk + hk_idx) * Dk + lane_in_row * values_per_lane;
        auto v_ = v + (b_idx * T * Hv + hv_idx) * Dv;
        y += (b_idx * T * Hv + hv_idx) * Dv;
        auto i_state = state_in + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk + lane_in_row * values_per_lane;
        float state[values_per_lane];
        for (int i = 0; i < values_per_lane; ++i) {
          state[i] = static_cast<float>(i_state[i]);
        }
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;
        for (int t = 0; t < T; ++t) {
          float gt = static_cast<float>(g_[hv_idx]);
          float part[partials_per_lane];
          for (int pb = 0; pb < partials_per_lane; ++pb) {
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
              int e = pb * 4 + i;
              state[e] = state[e] * gt;
              acc += state[e] * static_cast<float>(k_[e]);
            }
            part[pb] = acc;
          }
          float kv_mem =
              ((part[0] + part[1]) + (part[2] + part[3])) +
              ((part[4] + part[5]) + (part[6] + part[7]));
          kv_mem += simd_shuffle_xor(kv_mem, 1);
          kv_mem += simd_shuffle_xor(kv_mem, 2);
          auto delta =
              (static_cast<float>(v_[dv_idx]) - kv_mem) *
              static_cast<float>(beta_[hv_idx]);
          for (int pb = 0; pb < partials_per_lane; ++pb) {
            float acc = 0.0f;
            for (int i = 0; i < 4; ++i) {
              int e = pb * 4 + i;
              state[e] = state[e] + static_cast<float>(k_[e]) * delta;
              acc += state[e] * static_cast<float>(q_[e]);
            }
            part[pb] = acc;
          }
          float out =
              ((part[0] + part[1]) + (part[2] + part[3])) +
              ((part[4] + part[5]) + (part[6] + part[7]));
          out += simd_shuffle_xor(out, 1);
          out += simd_shuffle_xor(out, 2);
          if (lane_in_row == 0) {
            y[dv_idx] = static_cast<InT>(out);
          }
          q_ += Hk * Dk;
          k_ += Hk * Dk;
          v_ += Hv * Dv;
          y += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }
        for (int i = 0; i < values_per_lane; ++i) {
          o_state[i] = static_cast<StT>(state[i]);
        }
    """)

  /** the model in a directory: config.json and its shards — the language model's weights alone (not the vision
   * tower's, nor the multi-token predictor's) */
  /** `expertSlots` > 0, fewer than its experts: a mixture of experts' hot ones alone in memory, that many a layer
   * (`ExpertCache`), the rest read from the files as picked */
  def load(dir: Path, mapped: Boolean = SafeTensors.mappedByDefault,
      inPlace: Boolean = SafeTensors.alignInPlace, expertSlots: Int = 0): Qwen35 =
    Mlx.init()
    val config = Qwen35Config(Json.fromText(Files.readString(dir.resolve("config.json"))))
    val own = new Scope
    given Scope = own
    val hot = expertSlots > 0 && config.experts > expertSlots
    require(!hot || mapped, "hot experts need the weights mapped")
    // the experts mapped cold: their pages read only as a prompt's pass uses them
    val expertTensor = (k: String) => hot && k.contains(".mlp.switch_mlp.")
    val shards = SafeTensors.shards(dir)
    val all = shards.flatMap(SafeTensors.load(_, mapped, inPlace, expertTensor)).toMap
    val (keep, drop) = all.partition((k, _) => k.startsWith("language_model.") && !k.contains(".mtp."))
    drop.values.foreach(own.release)
    evalAll(keep.values.toSeq)
    val caches =
      if !hot then Map.empty[String, ExpertCache]
      else
        val where = shards.flatMap(SafeTensors.locate(_, inPlace)).toMap
        (0 until config.layers).map { i =>
          val prefix = s"language_model.model.layers.$i.mlp.switch_mlp"
          val parts = for m <- Vector("gate_proj", "up_proj", "down_proj"); t <- Vector("weight", "scales", "biases")
            yield where(s"$prefix.$m.$t")
          prefix -> ExpertCache(expertSlots, config.experts, parts, own)
        }.toMap
    new Qwen35(config, keep, own, caches)

  /** the hot experts' counts, all layers: (hits, misses) */
  def expertCounts(m: Model): (Long, Long) = m match
    case q: Qwen35 => q.counts
    case _ => (0L, 0L)
