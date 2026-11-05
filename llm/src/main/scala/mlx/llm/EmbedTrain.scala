package mlx.llm

import mlx.*
import mlx.Mlx.*

/** TRAINING AN EMBEDDING MODEL for a kind of search (contrastive, InfoNCE): LoRA adapters on its projections, a batch
 * of (query, document) pairs — each query's own document its positive, the batch's other documents its negatives —
 * the cross-entropy of the similarities (cosine / `temperature`) toward the diagonal. Batches from one project give
 * hard negatives (declarations alike). The embedder's vectors as `Embedder` makes them: the last token, L2-normed */
final class EmbedTrain(val embedder: Embedder, rank: Int = 8, lr: Float = 1e-4f,
    projs: Seq[String] = Seq("q", "v", "o"), temperature: Float = 0.05f):
  private val model = embedder.model match
    case a: Adaptable => a
    case m => sys.error(s"$m takes no adapters")
  val lora: Lora = Lora(model, rank, projs, Nil, alpha = 2f * rank, lr = lr)

  /** a text's vector [1, H], unit length, float32 — differentiable */
  private def vec(text: String)(using Scope): Arr =
    val ids = embedder.tokens(text)
    val st = model.newState()
    try
      val h = astype(reshape(model.hiddenStates(ints(ids, 1, ids.length), st, all = false), 1, -1), Dtype.Float32)
      divide(h, sqrt(add(sum(multiply(h, h), 1, keepdims = true), scalar(1e-12f))))
    finally st.close()

  /** the batch's mean cross-entropy: each query toward its own document among the batch's */
  def loss(pairs: Seq[(String, String)])(using Scope): Arr =
    val b = pairs.size
    val q = concatenate(pairs.map(p => vec(p._1)), 0)
    val d = concatenate(pairs.map(p => vec(p._2)), 0)
    val s = multiply(matmul(q, transpose(d, 1, 0)), scalar(1f / temperature))
    val diag = reshape(takeAlong(s, ints(Array.range(0, b), b, 1), 1), b)
    divide(sum(subtract(logsumexp(s, 1), diag), 0), scalar(b.toFloat))

  /** one step on a batch: its loss before the step */
  def step(pairs: Seq[(String, String)]): Float = lora.stepWith(loss(pairs))

  def close(): Unit = lora.close()
