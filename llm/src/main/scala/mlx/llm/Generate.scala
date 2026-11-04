package mlx.llm

import mlx.*
import mlx.Mlx.*

/** what a generation measured: the prompt's tokens and the time to the first new one (the prefill), the tokens
 * made after it and their time (the decode) */
final case class Timing(promptTokens: Int, prefillNanos: Long, tokens: Int, decodeNanos: Long, cachedTokens: Int = 0):
  /** the prompt's tokens computed a second (not those a cache gave) */
  def prefillPerSecond: Double = (promptTokens - cachedTokens) / (prefillNanos / 1e9)
  def decodePerSecond: Double = if tokens <= 1 then 0 else (tokens - 1) / (decodeNanos / 1e9)

/** A GENERATION, a token at a time: the prompt in one pass, then a step a token — each next step built and started
 * on the GPU before the last one's token is read, so the CPU's work overlaps the GPU's. Each token chosen as
 * `sampling` says (greedy by default).
 *
 * GUESSES CHECKED (`guesses` > 0, no penalties): the next tokens guessed from the text so far — where its
 * last three (or two) tokens were before, what followed them (prompt lookup: an agent rewrites what it has read) —
 * and up to `guesses` of them checked in one pass: the model's own pick at each place, the guesses taken while they
 * agree, its pick after them too. A pass a token or several; the tokens greedy's (but for a near tie the pass of
 * several computes apart) — or, drawn, a draw at each place (`Sampling.picks`): the draws' distribution unchanged.
 * Wrong guesses undone: the state back to before the pass, the tokens taken fed again where it cannot go back to
 * between (a recurrent state).
 *
 * KEPT FOR THE NEXT TURN (a `cache`): besides the prompt's marks, the state after the answer (its stop token fed:
 * the next turn's prompt begins with the conversation so far and this answer) and the state just after an anchor
 * token was fed (`anchors`, `Chat.anchors`: a tool call's opener — a client writes the call back its own way, the
 * next prompt parts right after it). A linear layer's state cannot be cut back: those are the places it is kept */
final class Session(model: Model, prompt: Array[Int], maxTokens: Int, stops: Int => Boolean,
    cache: PrefixCache | Null = null, marks: Seq[Int] = Nil, sampling: Sampling = Sampling.Greedy,
    guesses: Int = 0, port: Port | Null = null, anchors: Set[Int] = Set.empty, drafts: Int = 0):
  private var state: Model.State | Null = null
  private var cached = 0
  // the tokens made that the state was fed, in order (the prompt's before them)
  private val fed = scala.collection.mutable.ArrayBuffer.empty[Int]
  private val carry = new Scope
  private val t0 = System.nanoTime()
  private var firstAt = 0L
  private var endAt = 0L
  private var y: Arr | Null = null
  private var made = 0
  private var finished = false
  private var stopped = false
  // with penalties: how often each token was made, and the token ids to count them by — both on the GPU
  private var counts: Arr | Null = null
  private var ids: Arr | Null = null
  private var started = false
  // DRAFTS of the model's own (`drafts` > 0 and a draft layer: multi-token prediction): its state, and the pairs it
  // is yet to be fed — the hidden states that chose tokens, and those tokens; the last of them the next token chosen
  // (`last`), not yet fed to the model
  private val draftState: Model.Draft | Null =
    if drafts > 0 && !sampling.penalizes && port == null then model.newDraft() else null
  private val drafting = draftState != null
  private var pendingH: Arr | Null = null
  private var pendingT: Arr | Null = null
  private var last = 0
  // guesses checked: the text so far, the tokens made and not yet given
  private val checking = guesses > 0 && !sampling.penalizes && port == null && !drafting
  private val text = scala.collection.mutable.ArrayBuffer.from(prompt)
  private val ready = scala.collection.mutable.Queue.empty[Int]
  private var guessing = guesses
  // a pass taking no guess: that many steps without guessing, doubling while they keep failing
  private var resting = 0
  private var misses = 0
  /** guesses checked, taken, and passes made */
  var guessed = 0
  var taken = 0
  var passes = 0

  private def step(input: Arr): Arr = Scope(carry) {
    val logits = masked(model.logits(input, state.nn))
    if !sampling.penalizes then sampling.pick(logits).evalAsync()
    else
      val v = logits.dim(-1)
      if counts == null then
        counts = zeros(Seq(1, v), Dtype.Float32).moveTo(carry)
        ids = ints(Array.range(0, v), 1, v).moveTo(carry)
      val token = sampling.pick(sampling.penalize(astype(reshape(logits, 1, v), Dtype.Float32), counts.nn))
      val now = Sampling.counted(counts.nn, ids.nn, token).moveTo(carry)
      carry.release(counts.nn)
      counts = now
      token.evalAsync()
      now.evalAsync()
      token
  }

  /** the logits with what the port does not allow at −∞ (as they are without a port, or when it allows any) */
  private def masked(logits: Arr)(using Scope): Arr =
    if port == null then logits
    else port.nn.allowed match
      case None => logits
      case Some(ok) =>
        val v = logits.dim(-1)
        val flags = new Array[Int](v)
        ok.foreach(t => if t >= 0 && t < v then flags(t) = 1)
        where(greater(ints(flags, 1, v), int(0)), logits, astype(scalar(Float.NegativeInfinity), logits.dtype))

  /** of whole continuations after `token` (read, not yet fed), the one the model scores best: each fed from a
   * checkpoint (`token` and it), its tokens' log-probabilities summed, the state put back */
  private def best(token: Int, alts: Seq[Array[Int]]): Array[Int] =
    val st = state.nn
    val scored = alts.map { a =>
      val at = st.checkpoint()
      val input = token +: a
      val lp = Scope(carry) {
        val logits = astype(model.logitsAll(ints(input, 1, input.length), st), Dtype.Float32)
        val v = logits.dim(-1)
        val probs = softmax(reshape(logits, input.length, v), 1)
        // the probability of each alternative's token where it comes: rows 0 .. a.length − 1
        takeAlong(probs, ints(a ++ Array(0), input.length, 1), 1).toFloats.take(a.length)
      }
      st.rewind(at, 0): Unit // back to the checkpoint itself (a recurrent state can too): nothing to feed again
      at.close()
      // the MEAN log-probability per token: a sum would favour the shortest alternative whatever the model means
      (a, lp.map(p => math.log(math.max(p, 1e-30f).toDouble)).sum / a.length)
    }
    scored.maxBy(_._2)._1

  /** the next token, or none: the end of the turn (a stop token, not given), or `maxTokens` made */
  def next(): Option[Int] =
    if finished then None
    else if ready.nonEmpty then give(ready.dequeue())
    else
      if !started then
        started = true
        sampling.seed.foreach(Mlx.randomSeed)
        // from the longest prefix a cache keeps, if any; then the prompt but its last token in chunks, the state
        // evaluated after each (memory bounded), a snapshot kept at each mark; the last token the first step — as
        // mlx_lm's generate_step, so without a cache the numbers are its numbers
        val reused = if cache == null then None else cache.nn.lookup(prompt)
        state = reused.fold(model.newState())((_, snap) => model.restore(snap))
        var at = reused.fold(0)(_._1)
        cached = at
        val points = marks.filter(m => m > at && m < prompt.length).sorted
        while prompt.length - at > 1 do
          val next = points.find(_ > at).getOrElse(Int.MaxValue)
          val m = math.min(math.min(Session.chunk, prompt.length - at - 1), next - at)
          if !drafting then Scope(carry)(model.logits(Mlx.ints(prompt.slice(at, at + m), 1, m), state.nn).eval())
          else Scope(carry) {
            // the draft layer fed too: each position's hidden state and the token after it
            val h = model.hiddenStates(Mlx.ints(prompt.slice(at, at + m), 1, m), state.nn, all = true)
            model.draft(h, Mlx.ints(prompt.slice(at + 1, at + m + 1), 1, m), draftState.nn).eval()
          }
          at += m
          if at == next && cache != null then
            cache.nn.add(prompt.take(at), state.nn.snapshot(), durable = marks.headOption.contains(at))
        val input = Scope(carry)(Mlx.ints(prompt.drop(at), 1, prompt.length - at))
        if !drafting then y = step(input)
        else
          pendingH = Scope(carry)(model.hiddenStates(input, state.nn, all = false))
          y = Scope(carry)(sampling.pick(model.head(pendingH.nn)))
        carry.release(input)
      if drafting then
        if made == 0 then
          // the first token: the prompt's pick, its hidden state pending
          last = y.nn.itemInt
          carry.release(y.nn)
          y = null
          pendingT = ints(Array(last), 1, 1)(using carry)
          ready += last
        else drafted()
        if made == 0 then firstAt = System.nanoTime()
        return give(ready.dequeue())
      val current = y.nn
      // a guess likely only if the last two tokens given were seen before: then the chosen one read now (no step
      // started ahead), the guesses looked up after it and checked; else a step ahead, as without guesses
      if checking && made + 1 < maxTokens && resting == 0 && seenBefore then
        val token = current.itemInt
        carry.release(current)
        ready += token
        text += token
        val guess = lookup(guessing)
        if guess.nonEmpty then checked(token, guess)
        else
          y = step(Scope(carry)(ints(Array(token), 1, 1)))
          fedMore(Array(token))
      else if port != null then
        // a port: the token read first, told to it, then the next step built under its mask (no overlap)
        val token = current.itemInt
        carry.release(current)
        port.nn.took(token)
        ready += token
        val alts = if made + 1 < maxTokens then port.nn.choices.map(_.filter(_.nonEmpty)).filter(_.nonEmpty) else None
        val chosen = alts.map(best(token, _))
        chosen.foreach { b =>
          b.foreach(port.nn.took)
          ready ++= b
        }
        val more = token +: chosen.getOrElse(Array.empty[Int])
        y = if made + more.length < maxTokens then step(Scope(carry)(ints(more, 1, more.length))) else null
        if y != null then fedMore(more)
      else
        // the next step built and started BEFORE this one is read — as mlx_lm: the graph's building and encoding
        // overlap the GPU's running this one (read first, they did not: 13% of the decode's speed)
        val next = if made + 1 < maxTokens then step(current) else null
        val token = current.itemInt
        y = next
        carry.release(current)
        ready += token
        if next != null then fedMore(Array(token))
        if checking then
          text += token
          if resting > 0 then resting -= 1
      if made == 0 then firstAt = System.nanoTime()
      give(ready.dequeue())

  /** tokens made fed to the state: kept (`cache`) when one of them is an anchor */
  private def fedMore(tokens: Array[Int]): Unit =
    fed ++= tokens
    if cache != null && tokens.exists(anchors) then keepNow()

  /** the state as it is now kept, for the prompt and the tokens fed */
  private def keepNow(): Unit =
    cache.nn.add(prompt ++ fed, state.nn.snapshot())

  /** a token given: the end at a stop token or `maxTokens` */
  private def give(token: Int): Option[Int] =
    made += 1
    if stops(token) || made >= maxTokens then
      endAt = System.nanoTime()
      finished = true
      stopped = stops(token)
      // the answer's state kept when the stop token is the last fed: its step run to the end first (its state then
      // computed, not a graph holding the one before)
      // (fed as far as given, the stop token too); else the stop token fed now when all before it were and no more
      if stopped && cache != null then
        if fed.length == made && fed.last == token then
          if y != null then y.nn.eval()
          keepNow()
        else if fed.length == made - 1 then
          Scope(carry)(model.logits(ints(Array(token), 1, 1), state.nn).eval())
          fed += token
          keepNow()
      close()
      if stopped then None else Some(token)
    else Some(token)

  /** A ROUND OF DRAFTS (multi-token prediction): the draft layer fed the pairs pending, its last guessing the token
   * after `last`, then `drafts` − 1 more from its own hidden states — all on the GPU, no wait; then one pass of the
   * model over `last` and the drafts, the model's pick at each place, the drafts taken while they agree and its pick
   * after them the next chosen (one wait). The draft layer's guesses past the pairs dropped (its offset set back),
   * the model's state back to after the last taken — and the pairs it now owes the draft layer: each hidden state of
   * the pass up to that one, with the token after it */
  private def drafted(): Unit =
    val st = state.nn
    val d = draftState.nn
    val m = pendingT.nn.dim(1)
    var kept = 0
    val input = Scope(carry) {
      var o = model.draft(pendingH.nn, pendingT.nn, d)
      if m > 1 then o = slice(o, Seq(0, m - 1, 0), Seq(1, m, o.dim(2)))
      kept = d.offset
      def guess(h: Arr) = astype(reshape(argmax(model.head(h), -1), 1, 1), Dtype.Int32)
      val ts = scala.collection.mutable.ArrayBuffer(ints(Array(last), 1, 1), guess(o))
      while ts.length <= drafts do
        o = model.draft(o, ts.last, d)
        ts += guess(o)
      concatenate(ts.toSeq, 1)
    }
    // the drafts started on the GPU while the pass checking them is built
    input.evalAsync()
    d.offset = kept
    carry.release(pendingH.nn)
    carry.release(pendingT.nn)
    val at = st.checkpoint()
    val h = Scope(carry)(model.hiddenStates(input, st, all = true))
    val p = Scope(carry)(sampling.picks(model.head(h)))
    val Seq(in, picks) = Scope(carry)(readInts(Seq(input, p))): @unchecked
    carry.release(input)
    carry.release(p)
    var a = 0
    while a < drafts && in(a + 1) == picks(a) do a += 1
    if a < drafts && !st.rewind(at, a + 1) then
      Scope(carry)(model.logits(ints(in.take(a + 1), 1, a + 1), st).eval())
    at.close()
    passes += 1
    guessed += drafts
    taken += a
    fedMore(in.take(a + 1))
    last = picks(a)
    val chosen = in.slice(1, a + 1) :+ last
    ready ++= chosen
    pendingH = if a == drafts then h else
      val r = Scope(carry)(slice(h, Seq(0, 0, 0), Seq(1, a + 1, h.dim(2))))
      carry.release(h)
      r
    pendingT = ints(chosen, 1, a + 1)(using carry)

  /** a pass checking guesses: `token` the last chosen (read, not fed), `guess` what the text says follows it. One
   * pass over `token` and the guesses; they are taken while the model's picks agree, its pick after them the next
   * chosen. Wrong ones undone: the state back, the tokens taken fed again if need be. A pass taking none: guesses
   * rest a while, longer each time again */
  private def checked(token: Int, guess: Array[Int]): Unit =
    val st = state.nn
    val at = st.checkpoint()
    val input = token +: guess
    // the model's pick at each place: greedy's, or a draw (a guess taken when the draw is it — speculative sampling)
    val picks = Scope(carry)(sampling.picks(model.logitsAll(ints(input, 1, input.length), st)).toInts)
    var a = 0
    while a < guess.length && guess(a) == picks(a) do a += 1
    passes += 1
    guessed += guess.length
    taken += a
    if a < guess.length && !st.rewind(at, a + 1) then
      Scope(carry)(model.logits(ints(input.take(a + 1), 1, a + 1), st).eval())
    at.close()
    guessing = if a == guess.length then math.min(guesses, guessing + 1) else math.max(1, a)
    if a == 0 then
      misses = math.min(misses + 1, 6)
      resting = 1 << misses
    else misses = 0
    y = ints(Array(picks(a)), 1, 1)(using carry)
    fedMore(token +: guess.take(a))
    val out = guess.take(a)
    ready ++= out
    text ++= out

  /** whether the text's last two tokens were in it before, together (a guess likely after the next: one token
   * alone — "the" — was before nearly always, its guesses mostly wrong) */
  private def seenBefore: Boolean =
    val n = text.length
    n >= 3 && {
      val (a, b) = (text(n - 2), text(n - 1))
      var j = n - 3
      while j >= 0 && !(text(j) == a && text(j + 1) == b) do j -= 1
      j >= 0
    }

  /** what followed the text's last three (else two) tokens where they were last before, `k` at most */
  private def lookup(k: Int): Array[Int] = scala.util.boundary {
    val n = text.length
    for g <- Seq(3, 2) if n > g do
      var j = n - g - 1
      while j >= 0 do
        var i = 0
        while i < g && text(j + i) == text(n - g + i) do i += 1
        if i == g then scala.util.boundary.break(text.slice(j + g, math.min(j + g + k, n)).toArray)
        j -= 1
    Array.empty[Int]
  }

  /** whether it ended at a stop token (else: at `maxTokens`) */
  def endedByStop: Boolean = stopped
  def timing: Timing =
    Timing(prompt.length, firstAt - t0, made, (if endAt == 0 then System.nanoTime() else endAt) - firstAt, cached)
  def close(): Unit =
    if endAt == 0 then endAt = System.nanoTime()
    finished = true
    carry.close()
    if state != null then state.nn.close()
    if draftState != null then draftState.nn.close()

object Session:
  /** a prompt's tokens a pass */
  val chunk = 2048

object Generate:
  /** every token to `emit`, the stop token too; what it measured */
  def greedy(model: Model, prompt: Array[Int], maxTokens: Int, stop: Int => Boolean)(emit: Int => Unit): Timing =
    val s = Session(model, prompt, maxTokens, _ => false)
    var done = false
    while !done do
      s.next() match
        case Some(t) => emit(t); done = stop(t)
        case None => done = true
    val t = s.timing
    s.close()
    t
