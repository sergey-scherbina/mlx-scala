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
 * between (a recurrent state) */
final class Session(model: Model, prompt: Array[Int], maxTokens: Int, stops: Int => Boolean,
    cache: PrefixCache | Null = null, marks: Seq[Int] = Nil, sampling: Sampling = Sampling.Greedy,
    guesses: Int = 0):
  private var state: Model.State | Null = null
  private var cached = 0
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
  // guesses checked: the text so far, the tokens made and not yet given
  private val checking = guesses > 0 && !sampling.penalizes
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
    val logits = model.logits(input, state.nn)
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

  /** the next token, or none: the end of the turn (a stop token, not given), or `maxTokens` made */
  def next(): Option[Int] =
    if finished then None
    else if ready.nonEmpty then give(ready.dequeue())
    else
      if y == null then
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
          Scope(carry)(model.logits(Mlx.ints(prompt.slice(at, at + m), 1, m), state.nn).eval())
          at += m
          if at == next && cache != null then
            cache.nn.add(prompt.take(at), state.nn.snapshot(), durable = marks.headOption.contains(at))
        val input = Scope(carry)(Mlx.ints(prompt.drop(at), 1, prompt.length - at))
        y = step(input)
        carry.release(input)
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
        else y = step(Scope(carry)(ints(Array(token), 1, 1)))
      else
        // the next step built and started BEFORE this one is read — as mlx_lm: the graph's building and encoding
        // overlap the GPU's running this one (read first, they did not: 13% of the decode's speed)
        val next = if made + 1 < maxTokens then step(current) else null
        val token = current.itemInt
        y = next
        carry.release(current)
        ready += token
        if checking then
          text += token
          if resting > 0 then resting -= 1
      if made == 0 then firstAt = System.nanoTime()
      give(ready.dequeue())

  /** a token given: the end at a stop token or `maxTokens` */
  private def give(token: Int): Option[Int] =
    made += 1
    if stops(token) || made >= maxTokens then
      endAt = System.nanoTime()
      finished = true
      stopped = stops(token)
      close()
      if stopped then None else Some(token)
    else Some(token)

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
