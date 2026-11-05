package mlx.llm

import mlx.*
import mlx.Mlx.*

/** TRAINING THE DRAFT LAYER (multi-token prediction) for a kind of text: LoRA adapters on its projections (`fc`, `o`,
 * and a dense model's `gate`, `up`, `down`), trained to agree with the model. At each position of a text, the draft
 * from the hidden state at i and the token at i + 1 is trained toward the MODEL'S OWN pick at i + 1 (its argmax: what
 * a round checks a draft against), not toward the text. The model's answers cannot change (drafts are checked); only
 * how many drafts are taken — the agreement measured here is the first draft's chance to be taken, greedy. */
final class MtpTrain(val model: Adaptable, rank: Int = 16, lr: Float = 1e-4f, second: Boolean = true,
    projs: Seq[String] = Seq("fc", "o", "gate", "up", "down")):
  require(model.newDraft() != null, "the model has no draft layer (Mtp)")
  val lora: Lora = Lora(model, rank, projs, Seq(model.layerCount), alpha = 2f * rank, lr = lr)

  /** a text's pairs, computed once by the model (no adapters on it): the hidden states at 0 .. n − 2, the tokens at
   * 1 .. n − 1, the model's picks at 1 .. n − 1; the loss and the agreement counted from pair `from` on (a text the
   * model wrote after a prompt: its own part alone) */
  final class Pairs(val hidden: Arr, val next: Arr, val picks: Arr, val size: Int, val from: Int = 0)

  /** the pairs of a prompt and what the model wrote after it: counted from the prompt's last token on */
  def pairs(prompt: Array[Int], written: Array[Int])(using Scope): Pairs =
    val p = pairs(prompt ++ written)
    Pairs(p.hidden, p.next, p.picks, p.size, prompt.length - 1)

  def pairs(ids: Array[Int])(using s: Scope): Pairs =
    val n = ids.length
    val st = model.newState()
    try
      val h = model.hiddenStates(ints(ids, 1, n), st, all = true)
      val w = h.dim(2)
      val picks = Scope(s)(reshape(astype(slice(argmax(model.head(h), -1), Seq(0, 1), Seq(1, n)), Dtype.Int32),
        n - 1, 1))
      val hs = Scope(s)(contiguous(slice(h, Seq(0, 0, 0), Seq(1, n - 1, w))))
      val next = ints(ids.drop(1), 1, n - 1)
      evalAll(Seq(hs, next, picks))
      s.release(h)
      Pairs(hs, next, picks, n - 1)
    finally st.close()

  /** the draft's hidden states at each pair [1, m, H] */
  private def drafted(p: Pairs)(using Scope): Arr =
    val d = model.newDraft().nn
    try model.draft(p.hidden, p.next, d)
    finally d.close()

  /** logits [1, k, V] as [k, V], rows `from` .. on, float32 */
  private def rows(o: Arr, from: Int)(using Scope): Arr =
    val k = o.dim(1)
    val l = astype(model.head(slice(o, Seq(0, from, 0), Seq(1, k, o.dim(2)))), Dtype.Float32)
    reshape(l, k - from, l.dim(-1))

  private def draftLogits(p: Pairs)(using Scope): Arr = rows(drafted(p), p.from)

  /** the mean cross-entropy of the model's picks under logits [k, V] */
  private def ce(l: Arr, picks: Arr)(using Scope): Arr =
    val k = l.dim(0)
    divide(sum(subtract(logsumexp(l, 1), reshape(takeAlong(l, picks, 1), k)), 0), scalar(k.toFloat))

  /** the first draft's loss, and (`second`) the second's: the draft layer on its own hidden states and the token
   * after (in the model's own text, the first draft when it was taken), toward the model's pick one further */
  def loss(p: Pairs)(using Scope): Arr =
    val o = drafted(p)
    val first = ce(rows(o, p.from), slice(p.picks, Seq(p.from, 0), Seq(p.size, 1)))
    if !second || p.size - p.from < 2 then first
    else
      val m = p.size
      val d = model.newDraft().nn
      val o2 =
        try model.draft(slice(o, Seq(0, 0, 0), Seq(1, m - 1, o.dim(2))), slice(p.next, Seq(0, 1), Seq(1, m)), d)
        finally d.close()
      add(first, ce(rows(o2, p.from), slice(p.picks, Seq(p.from + 1, 0), Seq(m, 1))))

  /** one step on a text's pairs: the loss before it */
  def step(p: Pairs): Float = lora.stepWith(loss(p))

  /** how many of the pairs' drafts are the model's picks: with the adapters as trained (`trained`), or without */
  def agreement(p: Pairs, trained: Boolean): Int =
    if trained then lora.attachTrained() else lora.detach()
    val s = new Scope
    try
      given Scope = s
      val Seq(drafts, picks) = readInts(Seq(reshape(argmax(draftLogits(p), -1), p.size - p.from),
        slice(p.picks, Seq(p.from, 0), Seq(p.size, 1)))): @unchecked
      drafts.indices.count(i => drafts(i) == picks(i))
    finally
      s.close()
      lora.detach()

  def close(): Unit = lora.close()
