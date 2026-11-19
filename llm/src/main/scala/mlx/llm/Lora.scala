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

/** `except`: adapters NOT made, by name ("1.o", "9.down") — a projection removed from a model (its output silenced)
 * must not grow a low-rank copy of itself under the adapter */
final class Lora(val model: Adaptable, rank: Int = 8, projs: Seq[String] = Seq("q", "v"), layers: Seq[Int] = Nil,
    alpha: Float = 16f, lr: Float = 1e-4f, weightDecay: Float = 0f, seed: Long = 1, except: Set[String] = Set.empty):
  private val keep = new Scope
  private val at = if layers.nonEmpty then layers else 0 until model.layerCount
  val names: Vector[String] = (for i <- at; p <- projs if model.widths(i, p).isDefined && !except(s"$i.$p") yield s"$i.$p").toVector
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

  /** THE MEAN OF B EXAMPLES' LOSSES IN ONE PASS (`Model.hiddenBatch`): each its own mean cross-entropy from its own
   * `from` on, as `loss` gives it alone; the examples padded on the right to the longest, the head run on the
   * answers' rows alone (B·n rows of a 151k vocabulary would not fit) */
  private def lossBatch(examples: Seq[(Array[Int], Int)])(ps: Seq[Arr])(using Scope): Arr =
    attach(ps)
    val b = examples.size
    val n = examples.map(_._1.length).max
    val padded = examples.flatMap((ids, _) => ids ++ Array.fill(n - ids.length)(0)).toArray
    val h = model.hiddenBatch(ints(padded, b, n))
    val hidden = h.dim(2)
    // the rows that predict an answer token (row j predicts token j + 1), the tokens, each example's weight
    val rows = examples.zipWithIndex.flatMap { case ((ids, from), e) => (from - 1 until ids.length - 1).map(e * n + _) }
    val targets = examples.flatMap((ids, from) => ids.drop(from))
    val weights = examples.flatMap((ids, from) => Seq.fill(ids.length - from)(1f / ((ids.length - from) * b)))
    val m = rows.size
    val at = take(reshape(h, b * n, hidden), ints(rows.toArray, m), 0)
    val logits = astype(model.head(reshape(at, 1, m, hidden)), Dtype.Float32)
    val pred = reshape(logits, m, logits.dim(-1))
    val ce = subtract(logsumexp(pred, 1), reshape(takeAlong(pred, ints(targets.toArray, m, 1), 1), m))
    sum(multiply(ce, floats(weights.toArray, m)), 0)

  /** THE MEAN OF B EXAMPLES' LOSSES OVER THEIR TRIE (`Model.hiddenTree`): a prefix they share — a rule, a table — is
   * one path of the trie, run once; each example's answer tokens predicted from its own path */
  private def lossTree(examples: Seq[(Array[Int], Int)])(ps: Seq[Arr])(using Scope): Arr =
    attach(ps)
    val b = examples.size
    val ids = scala.collection.mutable.ArrayBuffer.empty[Int]
    val parent = scala.collection.mutable.ArrayBuffer.empty[Int]
    val child = scala.collection.mutable.HashMap.empty[(Int, Int), Int]
    // each example's path: its tokens' nodes
    val paths = examples.map { (tokens, _) =>
      var at = -1
      tokens.map { t =>
        at = child.getOrElseUpdate((at, t), { ids += t; parent += at; ids.size - 1 })
        at
      }
    }
    val h = model.hiddenTree(ids.toArray, parent.toArray)
    val hidden = h.dim(2)
    val rows = examples.zip(paths).flatMap { case ((tokens, from), path) => (from - 1 until tokens.length - 1).map(path) }
    val targets = examples.flatMap((tokens, from) => tokens.drop(from))
    val weights = examples.flatMap((tokens, from) => Seq.fill(tokens.length - from)(1f / ((tokens.length - from) * b)))
    val m = rows.size
    val at = take(reshape(h, ids.size, hidden), ints(rows.toArray, m), 0)
    val logits = astype(model.head(reshape(at, 1, m, hidden)), Dtype.Float32)
    val pred = reshape(logits, m, logits.dim(-1))
    val ce = subtract(logsumexp(pred, 1), reshape(takeAlong(pred, ints(targets.toArray, m, 1), 1), m))
    sum(multiply(ce, floats(weights.toArray, m)), 0)

  import Lora.Choice

  /** THE LISTWISE LOSS OVER A TRIE (okay-inlay concept.md §5 role 2): every choice's context and options are paths
   * of one trie (`Model.hiddenTree`: a prompt and the lines decided run once); an option's score is the sum of its
   * tokens' log-probabilities; the loss the mean over choices of −log softmax(scores)[meant] */
  private def lossChoices(choices: Seq[Choice])(ps: Seq[Arr])(using Scope): Arr =
    attach(ps)
    val ids = scala.collection.mutable.ArrayBuffer.empty[Int]
    val parent = scala.collection.mutable.ArrayBuffer.empty[Int]
    val child = scala.collection.mutable.HashMap.empty[(Int, Int), Int]
    def walk(from: Int, tokens: Array[Int]): Array[Int] =
      var at = from
      tokens.map { t =>
        at = child.getOrElseUpdate((at, t), { ids += t; parent += at; ids.size - 1 })
        at
      }
    // each option's rows (the node before each of its tokens) and tokens
    val rows = scala.collection.mutable.ArrayBuffer.empty[Int]
    val targets = scala.collection.mutable.ArrayBuffer.empty[Int]
    val spans = choices.map { c =>
      val ctx = walk(-1, c.context)
      c.options.map { o =>
        val nodes = walk(ctx.last, o)
        val start = rows.size
        rows ++= (ctx.last +: nodes.init); targets ++= o
        (start, rows.size)
      }
    }
    val h = model.hiddenTree(ids.toArray, parent.toArray)
    val hidden = h.dim(2)
    val m = rows.size
    val at = take(reshape(h, ids.size, hidden), ints(rows.toArray, m), 0)
    val logits = astype(model.head(reshape(at, 1, m, hidden)), Dtype.Float32)
    val pred = reshape(logits, m, logits.dim(-1))
    // each token's log-probability, then each option's sum (a 0/1 matrix [options, m] times them)
    val lp = subtract(reshape(takeAlong(pred, ints(targets.toArray, m, 1), 1), m), logsumexp(pred, 1))
    val all = spans.flatten
    val sumOf = Array.fill(all.size * m)(0f)
    for ((s, e), k) <- all.zipWithIndex; r <- s until e do sumOf(k * m + r) = 1f
    val scores = reshape(matmul(floats(sumOf, all.size, m), reshape(lp, m, 1)), all.size)
    // per choice: −log softmax at the meant option; the mean over choices
    var at0 = 0
    val terms = choices.zip(spans).map { (c, sp) =>
      val s = slice(scores, Seq(at0), Seq(at0 + sp.size))
      at0 += sp.size
      subtract(logsumexp(s, 0), slice(s, Seq(c.meant), Seq(c.meant + 1)))
    }
    divide(sum(concatenate(terms, 0), 0), scalar(choices.size.toFloat))

  /** one AdamW step on choices among options, over their trie (`lossChoices`): the loss before the step */
  def stepChoices(choices: Seq[Choice]): Float = stepOn(lossChoices(choices))

  /** the choices' loss as the adapters are now (no step) */
  def evaluateChoices(choices: Seq[Choice]): Float =
    val s = new Scope
    try
      attach(params)
      lossChoices(choices)(params)(using s).itemFloat
    finally
      s.close()
      model.lora = Map.empty
      model.training(false)

  /** one AdamW step on B examples over their trie: the mean of their losses before the step (`lossTree`) */
  def stepTree(examples: Seq[(Array[Int], Int)]): Float = stepOn(lossTree(examples))

  /** AN EPOCH IN TRIES — the way to train a reader whose prompts share a beginning (a rule, a table, a schema):
   * the examples shuffled, within each window of 16 batches sorted by their tokens (examples over one table land
   * in one batch), each batch of `batch` one `stepTree`; the batches in random order. Its mean loss.
   * Measured (okay-desk specs/frames.md §5.12): batches of 8 at lr 2e-4 against one example a step — 1.8× faster an
   * epoch, and the reader wrong 51 against 93 of 1004 questions never trained on, over three seeds alike */
  def epochTree(examples: Seq[(Array[Int], Int)], rnd: scala.util.Random, batch: Int = 8): Float =
    given Ordering[Array[Int]] = (a, b) =>
      val i = a.indices.find(k => k >= b.length || a(k) != b(k))
      i.fold(a.length - b.length)(k => if k >= b.length then 1 else Integer.compare(a(k), b(k)))
    val batches = rnd.shuffle(examples.toVector).grouped(batch * 16).toVector
      .flatMap(w => w.sortBy(_._1).grouped(batch).toVector)
    var sum = 0.0; var n = 0
    for b <- rnd.shuffle(batches) do
      sum += (if b.size == 1 then step(b.head._1, b.head._2) else stepTree(b)) * b.size
      n += b.size
    (sum / math.max(1, n)).toFloat

  /** the trie's loss as the adapters are now (no step) — the law: the mean of `evaluate` over its examples */
  def evaluateTree(examples: Seq[(Array[Int], Int)]): Float =
    val s = new Scope
    try
      attach(params)
      lossTree(examples)(params)(using s).itemFloat
    finally
      s.close()
      model.lora = Map.empty
      model.training(false)

  /** one AdamW step on B examples at once: the mean of their losses before the step (`lossBatch`) */
  def stepBatch(examples: Seq[(Array[Int], Int)]): Float = stepOn(lossBatch(examples))

  /** the batch's loss as the adapters are now (no step) — the law: the mean of `evaluate` over its examples */
  def evaluateBatch(examples: Seq[(Array[Int], Int)]): Float =
    val s = new Scope
    try
      attach(params)
      lossBatch(examples)(params)(using s).itemFloat
    finally
      s.close()
      model.lora = Map.empty
      model.training(false)

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
  def step(ids: Array[Int], from: Int): Float = stepOn(loss(ids, from))

  /** one AdamW step on any loss of the model with the adapters on (`f`: the adapters attached, its value a scalar —
   * a draft layer's agreement with the model, say): the loss before the step */
  def stepWith(f: Scope ?=> Arr): Float = stepOn { ps =>
    attach(ps)
    f
  }

  /** a loss of the model with the adapters as they are now (no step) */
  def evaluateWith(f: Scope ?=> Arr): Float =
    val s = new Scope
    try
      attach(params)
      f(using s).itemFloat
    finally
      s.close()
      model.lora = Map.empty
      model.training(false)

  private def stepOn(loss: Scope ?=> Seq[Arr] => Arr): Float =
    val s = new Scope
    given Scope = s
    try
      val (value, grads) = valueAndGrad(params, params.indices)(loss)
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

  /** Warm-start from saved adapter parameters. AdamW moments and step count restart at zero.
    * Validate the complete checkpoint before replacing anything; a rejected load keeps the old parameters.
    */
  def restoreParameters(path: String): Unit =
    val s = new Scope
    try
      val tensors = loadSafetensors(path)(using s)
      val keys = names.flatMap(n => Vector(s"$n.a", s"$n.b"))
      require(tensors.keySet == (keys :+ "lora.scale").toSet, "adapter projections differ")
      require(tensors("lora.scale").itemFloat == alpha / rank, "adapter scale differs")
      keys.zip(params).foreach { (key, before) =>
        require(tensors(key).shape == before.shape, s"adapter shape differs: $key")
        require(tensors(key).toFloats.forall(_.isFinite), s"nonfinite adapter: $key")
      }
      val next = keys.map(k => tensors(k).moveTo(keep))
      detach()
      (params ++ m ++ v).foreach(keep.release)
      params = next
      m = params.map(p => zeros(p.shape, Dtype.Float32)(using keep))
      v = params.map(p => zeros(p.shape, Dtype.Float32)(using keep))
      t = 0
    finally s.close()

  def close(): Unit =
    detach()
    keep.close()

object Lora:
  /** A CHOICE AMONG OPTIONS (`Lora.stepChoices`): a context (its tokens) and its options' tokens (each with its own
   * end), the one meant — a reader trained as it reads: the options' log-probabilities after the context, the meant
   * one's share among them */
  final case class Choice(context: Array[Int], options: Vector[Array[Int]], meant: Int)

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
