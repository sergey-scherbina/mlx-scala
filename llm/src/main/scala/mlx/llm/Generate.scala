package mlx.llm

import mlx.*
import mlx.Mlx.*

/** what a generation measured: the prompt's tokens and the time to the first new one (the prefill), the tokens
 * made after it and their time (the decode) */
final case class Timing(promptTokens: Int, prefillNanos: Long, tokens: Int, decodeNanos: Long):
  def prefillPerSecond: Double = promptTokens / (prefillNanos / 1e9)
  def decodePerSecond: Double = if tokens <= 1 then 0 else (tokens - 1) / (decodeNanos / 1e9)

/** A GENERATION, a token at a time: the prompt in one pass, then a step a token — each next step built and started
 * on the GPU before the last one's token is read, so the CPU's work overlaps the GPU's. Greedy */
final class Session(model: Model, prompt: Array[Int], maxTokens: Int, stops: Int => Boolean):
  private val state = model.newState()
  private val carry = new Scope
  private val t0 = System.nanoTime()
  private var firstAt = 0L
  private var endAt = 0L
  private var y: Arr | Null = null
  private var made = 0
  private var finished = false
  private var stopped = false

  private def step(input: Arr): Arr = Scope(carry) {
    Mlx.reshape(Mlx.argmax(model.logits(input, state), -1), 1, 1).evalAsync()
  }

  /** the next token, or none: the end of the turn (a stop token, not given), or `maxTokens` made */
  def next(): Option[Int] =
    if finished then None
    else
      if y == null then
        val input = Scope(carry)(Mlx.ints(prompt, 1, prompt.length))
        y = step(input)
        carry.release(input)
      val current = y.nn
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
        y = step(current)                 // the next started before this one is given out
        carry.release(current)
        Some(token)

  /** whether it ended at a stop token (else: at `maxTokens`) */
  def endedByStop: Boolean = stopped
  def timing: Timing =
    Timing(prompt.length, firstAt - t0, made, (if endAt == 0 then System.nanoTime() else endAt) - firstAt)
  def close(): Unit =
    if endAt == 0 then endAt = System.nanoTime()
    finished = true
    carry.close()
    state.close()

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
