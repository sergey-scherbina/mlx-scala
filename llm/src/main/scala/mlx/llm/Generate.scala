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
 * `sampling` says (greedy by default) */
final class Session(model: Model, prompt: Array[Int], maxTokens: Int, stops: Int => Boolean,
    cache: PrefixCache | Null = null, marks: Seq[Int] = Nil, sampling: Sampling = Sampling.Greedy):
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

  private def step(input: Arr): Arr = Scope(carry) {
    sampling.pick(model.logits(input, state.nn)).evalAsync()
  }

  /** the next token, or none: the end of the turn (a stop token, not given), or `maxTokens` made */
  def next(): Option[Int] =
    if finished then None
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
      // the next step built and started BEFORE this one is read — as mlx_lm: the graph's building and encoding
      // overlap the GPU's running this one (read first, they did not: 13% of the decode's speed)
      val next = if made + 1 < maxTokens then step(current) else null
      val token = current.itemInt
      if made == 0 then firstAt = System.nanoTime()
      made += 1
      if stops(token) || made >= maxTokens then
        endAt = System.nanoTime()
        finished = true
        stopped = stops(token)
        close()
        if stopped then None else Some(token)
      else
        y = next
        carry.release(current)
        Some(token)

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
