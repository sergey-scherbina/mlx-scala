package mlx.llm

import mlx.*
import mlx.Mlx.*

/** what a generation measured: the prompt's tokens and the time to the first new one (the prefill), the tokens
 * made after it and their time (the decode) */
final case class Timing(promptTokens: Int, prefillNanos: Long, tokens: Int, decodeNanos: Long):
  def prefillPerSecond: Double = promptTokens / (prefillNanos / 1e9)
  def decodePerSecond: Double = if tokens <= 1 then 0 else (tokens - 1) / (decodeNanos / 1e9)

/** GREEDY GENERATION: the prompt in one pass, then a token at a time — each next step built and started on the GPU
 * before the last one's token is read, so the CPU's work overlaps the GPU's */
object Generate:
  def greedy(model: Qwen3, prompt: Array[Int], maxTokens: Int, stop: Int => Boolean)(emit: Int => Unit): Timing =
    val cache = model.newCache()
    val carry = new Scope
    def step(input: Arr): Arr = Scope(carry) {
      reshape(argmax(model.logits(input, cache), -1), 1, 1).evalAsync()
    }
    val t0 = System.nanoTime()
    val input = Scope(carry)(ints(prompt, 1, prompt.length))
    var y = step(input)
    carry.release(input)
    var first = y.itemInt
    val t1 = System.nanoTime()
    var made = 1
    emit(first)
    var done = stop(first) || made >= maxTokens
    while !done do
      val next = step(y)
      carry.release(y)
      y = next
      first = y.itemInt
      made += 1
      emit(first)
      done = stop(first) || made >= maxTokens
    val t2 = System.nanoTime()
    carry.close()
    cache.foreach(_.close())
    Timing(prompt.length, t1 - t0, made, t2 - t1)
