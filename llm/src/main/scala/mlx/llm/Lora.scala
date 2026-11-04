package mlx.llm

import mlx.*
import mlx.Mlx.*

/**
 * LoRA TRAINING (okay-inlay specs/refine.md): adapters `(A [in, r], B [r, out])` on Qwen3's projections — A small
 * random, B zero, so training starts from the model as it is — trained by AdamW on the cross-entropy of an example's
 * answer tokens (the prompt's not counted). Gradients by `Mlx.valueAndGrad`, the adapters its inputs.
 */
/** a model whose projections take LoRA adapters (`"<layer>.<proj>"` → (A, B)): what `Lora` trains */
trait Adaptable extends Model:
  var lora: Map[String, (Arr, Arr)]
  var loraScale: Float
  def layerCount: Int
  /** a projection's input and output widths, if layer `i` has it */
  def widths(i: Int, proj: String): Option[(Int, Int)]
  /** gradients wanted through the forward pass (a model whose fast path has none takes a slower one) */
  def training(on: Boolean): Unit = ()

final class Lora(val model: Adaptable, rank: Int = 8, projs: Seq[String] = Seq("q", "v"), layers: Seq[Int] = Nil,
    alpha: Float = 16f, lr: Float = 1e-4f, weightDecay: Float = 0f, seed: Long = 1):
  private val keep = new Scope
  private val at = if layers.nonEmpty then layers else 0 until model.layerCount
  val names: Vector[String] = (for i <- at; p <- projs if model.widths(i, p).isDefined yield s"$i.$p").toVector
  private val r = scala.util.Random(seed)
  // the parameters, A then B for each adapter; AdamW's moments
  private var params: Vector[Arr] = names.flatMap { n =>
    val Array(i, p) = n.split('.')
    val (in, out) = model.widths(i.toInt, p).get
    val std = 1f / math.sqrt(in).toFloat
    Vector(floats(Array.fill(in * rank)(r.nextGaussian().toFloat * std), in, rank)(using keep),
      zeros(Seq(rank, out), Dtype.Float32)(using keep))
  }
  private var m: Vector[Arr] = params.map(p => zeros(p.shape, Dtype.Float32)(using keep))
  private var v: Vector[Arr] = params.map(p => zeros(p.shape, Dtype.Float32)(using keep))
  private var t = 0
  /** the adapters' parameters in all */
  def size: Long = params.map(_.shape.product.toLong).sum

  private def attach(ps: Seq[Arr]): Unit =
    model.training(true)
    model.lora = names.zipWithIndex.map((n, k) => n -> (ps(2 * k), ps(2 * k + 1))).toMap
    model.loraScale = alpha / rank

  /** the mean cross-entropy of `ids`' tokens from `from` on, each predicted from those before */
  private def loss(ids: Array[Int], from: Int)(ps: Seq[Arr])(using Scope): Arr =
    attach(ps)
    val st = model.newState()
    try
      val n = ids.length
      val logits = astype(model.logitsAll(ints(ids, 1, n), st), Dtype.Float32)
      val vocab = logits.dim(-1)
      val pred = reshape(slice(logits, Seq(0, from - 1, 0), Seq(1, n - 1, vocab)), n - from, vocab)
      val target = ints(ids.drop(from), n - from, 1)
      val ce = subtract(logsumexp(pred, 1), reshape(takeAlong(pred, target, 1), n - from))
      divide(sum(ce, 0), scalar((n - from).toFloat))
    finally st.close()

  /** the loss of an example as the adapters are now (no step) */
  def evaluate(ids: Array[Int], from: Int): Float =
    val s = new Scope
    try
      attach(params)
      loss(ids, from)(params)(using s).itemFloat
    finally
      s.close()
      model.lora = Map.empty
      model.training(false)

  /** one AdamW step on an example: its loss before the step */
  def step(ids: Array[Int], from: Int): Float =
    val s = new Scope
    given Scope = s
    try
      val (value, grads) = valueAndGrad(params, params.indices)(loss(ids, from))
      t += 1
      val (b1, b2, eps) = (0.9f, 0.999f, 1e-8f)
      val c1 = 1f - math.pow(b1, t).toFloat
      val c2 = 1f - math.pow(b2, t).toFloat
      val next = params.indices.map { k =>
        val g = astype(grads(k), Dtype.Float32)
        val mk = add(multiply(m(k), scalar(b1)), multiply(g, scalar(1 - b1)))
        val vk = add(multiply(v(k), scalar(b2)), multiply(multiply(g, g), scalar(1 - b2)))
        val upd = divide(divide(mk, scalar(c1)), add(sqrt(divide(vk, scalar(c2))), scalar(eps)))
        val pk = subtract(params(k), multiply(add(upd, multiply(params(k), scalar(weightDecay))), scalar(lr)))
        (pk, mk, vk)
      }
      val before = value.itemFloat
      evalAll(next.flatMap((p, a, b) => Seq(p, a, b)))
      val old = params ++ m ++ v
      params = next.map(_._1.moveTo(keep)).toVector
      m = next.map(_._2.moveTo(keep)).toVector
      v = next.map(_._3.moveTo(keep)).toVector
      old.foreach(keep.release)
      before
    finally
      s.close()
      model.lora = Map.empty
      model.training(false)

  /** the adapters on the model, for generation; `detach` takes them off */
  def attachTrained(): Unit =
    attach(params)
    model.training(false) // generation: the fast path
  def detach(): Unit =
    model.lora = Map.empty
    model.training(false)
  /** the adapters written as safetensors: `<layer>.<proj>.a` and `.b`, and `lora.scale` */
  def save(path: String): Unit =
    val s = new Scope
    try
      val tensors = names.zipWithIndex.flatMap((n, k) => Seq(s"$n.a" -> params(2 * k), s"$n.b" -> params(2 * k + 1))) :+
        ("lora.scale" -> scalar(alpha / rank)(using s))
      saveSafetensors(path, tensors, Map("kind" -> "lora", "rank" -> rank.toString, "alpha" -> alpha.toString))
    finally s.close()

  def close(): Unit =
    detach()
    keep.close()

object Lora:
  /** an adapter saved by `save`, put on the model for generation: its scope (close it, and take the adapter off
   * with `model.lora = Map.empty`, when done) */
  def attach(model: Adaptable, path: String): Scope =
    val s = new Scope
    val t = loadSafetensors(path)(using s)
    val names = t.keys.filter(_.endsWith(".a")).map(_.stripSuffix(".a"))
    model.lora = names.map(n => n -> (t(s"$n.a"), t(s"$n.b"))).toMap
    model.loraScale = t("lora.scale").itemFloat
    model.training(false)
    s
