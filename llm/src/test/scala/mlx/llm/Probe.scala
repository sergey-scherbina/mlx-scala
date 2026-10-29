package mlx.llm

import mlx.*
import mlx.Mlx.*

/** where the time of a long prompt goes: prefill alone at sizes, peak memory; then decode steps at that context */
@main def probe(sizes: Int*): Unit =
  val dir = Models.snapshot("mlx-community/Qwen3-4B-4bit").get
  val model = Qwen3.load(dir)
  for n <- if sizes.isEmpty then Seq(512, 1024, 2048, 4096) else sizes do
    val cache = model.newState()
    val s = new Scope
    given Scope = s
    val ids = ints(Array.fill(n)(1000 + util.Random.nextInt(5000)), 1, n)
    val t0 = System.nanoTime()
    val y = Scope(s)(reshape(argmax(model.logits(ids, cache), -1), 1, 1).eval())
    val t1 = System.nanoTime()
    var tok = y
    val d0 = System.nanoTime()
    for _ <- 1 to 32 do
      val next = Scope(s)(reshape(argmax(model.logits(tok, cache), -1), 1, 1).eval())
      s.release(tok)
      tok = next
    val d1 = System.nanoTime()
    println(f"n=$n%5d prefill ${(t1 - t0) / 1e6}%8.1f ms (${n / ((t1 - t0) / 1e9)}%6.0f tok/s)  decode " +
      f"${32 / ((d1 - d0) / 1e9)}%6.1f tok/s  peak ${Mlx.peakMemory / 1e9}%.1f GB  live arrays ${s.size}")
    s.close()
    cache.close()
