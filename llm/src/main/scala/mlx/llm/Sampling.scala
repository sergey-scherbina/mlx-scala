package mlx.llm

import java.nio.file.{Files, Path}
import okay.effects.json.Json
import mlx.*
import mlx.Mlx.*

/** HOW A TOKEN IS CHOSEN: greedily (temperature 0), or drawn — from the `topK` likeliest (0: all), of them the
 * smallest set whose probability reaches `topP` (1: all), softmax(logits / temperature). On the GPU: the k by
 * argpartition, sorted, their cumulative probability; a draw among those left. `seed`: the same draws each run */
final case class Sampling(temperature: Double = 0, topP: Double = 1, topK: Int = 0, seed: Option[Long] = None):
  def greedy: Boolean = temperature <= 0

  /** the next token of logits [1, 1, V] (or [1, V]): [1, 1] */
  def pick(logits: Arr)(using Scope): Arr =
    val v = logits.dim(-1)
    if greedy then reshape(argmax(logits, -1), 1, 1)
    else
      val l = astype(reshape(logits, 1, v), Dtype.Float32)
      val inv = scalar((1.0 / temperature).toFloat)
      if topK <= 0 && topP >= 1 then reshape(categorical(multiply(l, inv), -1), 1, 1)
      else
        // the candidates, likeliest first: the k (argpartition: the k largest, unordered), or all
        val (cand0, ids0) =
          if topK > 0 && topK < v then
            val ids = slice(argpartition(negative(l), topK - 1, -1), Seq(0, 0), Seq(1, topK))
            (takeAlong(l, ids, -1), ids)
          else
            val ids = argsort(negative(l), -1)
            (takeAlong(l, ids, -1), ids)
        val order = argsort(negative(cand0), -1)
        val cand = takeAlong(cand0, order, -1)
        val ids = takeAlong(ids0, order, -1)
        // top-p: a candidate out when the ones before it already reach p
        val kept =
          if topP >= 1 then cand
          else
            val probs = softmax(multiply(cand, inv), -1)
            val before = subtract(cumsum(probs, -1), probs)
            where(greater(before, scalar(topP.toFloat)), scalar(Float.NegativeInfinity), cand)
        val choice = reshape(categorical(multiply(kept, inv), -1), 1, 1)
        takeAlong(ids, choice, -1)

object Sampling:
  val Greedy: Sampling = Sampling()

  /** the model's own (generation_config.json): its temperature, top_p, top_k when it samples; else greedy */
  def of(dir: Path): Sampling =
    val f = dir.resolve("generation_config.json")
    if !Files.exists(f) then Greedy
    else
      val c = Json.fromText(Files.readString(f))
      if c.field("do_sample").exists(_ == Json.Bool(false)) then Greedy
      else Sampling(c.field("temperature").map(_.double).getOrElse(0.0), c.field("top_p").map(_.double).getOrElse(1.0),
        c.field("top_k").map(_.int).getOrElse(0))
