package mlx.llm

import mlx.*
import scala.collection.mutable

/** HOT EXPERTS: a mixture-of-experts layer's experts kept `slots` at a time — of Qwen3.6-35B's 256 a layer, 64 hold
 * 82% of a decode's picks — the rest left in the file. Each of the layer's expert tensors (weight, scales, biases of
 * gate, up, down: [E, …]) has a buffer of `slots` experts ([slots, …]); an expert picked and not there is read from
 * the file into the slot least recently used (`pread` straight into the buffer's memory), and the layer computes on
 * the slots, its ids mapped to theirs. The model's memory: its other weights and these buffers (Qwen3.6-35B, 64 a
 * layer: ~6 GB instead of 19.5).
 *
 * The caller knows the picks on the CPU (the router read each layer: what it costs) and keeps a slot's old expert
 * until the GPU is done with it — a fill happens before the layer's kernels, after the last ones that read it. A
 * decode's picks only: a prompt uses nearly every expert, and runs on the file's pages, mapped */
final class ExpertCache(val slots: Int, experts: Int, parts: Vector[SafeTensors.Located], own: Scope):
  require(parts.forall(_.shape.head == experts), "each part's first axis its experts")
  private val fds = mutable.Map.empty[java.nio.file.Path, Int]
  private def fd(p: java.nio.file.Path) = fds.getOrElseUpdate(p, Mlx.openFd(p.toString))
  /** each part's buffer, [slots, …] */
  val buffers: Vector[Arr] = parts.map(p => Mlx.buffer(slots +: p.shape.tail, p.dtype)(using own))
  private val perExpert = parts.map(_.nbytes / experts)
  private val slotOf = Array.fill(experts)(-1)
  private val expertIn = Array.fill(slots)(-1)
  // the slots, least recently used first
  private val order = mutable.LinkedHashSet.from(0 until slots)
  var hits = 0L
  var misses = 0L
  /** experts read ahead (`prefetch`), and of those, picked */
  var prefetched = 0L
  var prefetchedUsed = 0L
  private var guess: Set[Int] = Set.empty
  /** where a decode's time goes: reading misses (on the caller), waiting for a read ahead */
  var missNanos = 0L
  var awaitNanos = 0L
  @volatile private var pending: java.util.concurrent.Future[?] | Null = null

  /** READ AHEAD: `ids` (a guess at the next picks — the next layer's router on this layer's input) read in on a
   * thread of its own, while the GPU computes; `slotsOf` waits for it first. The GPU must be done with this layer's
   * slots (it is: the last that read them, a step before) */
  def prefetch(ids: Array[Int]): Unit =
    await()
    val wanted = ids.distinct.take(slots)
    guess = wanted.toSet
    pending = ExpertCache.io.submit((() => bring(wanted, prefetching = true)): Runnable)

  private def await(): Unit =
    val p = pending
    if p != null then
      val t = System.nanoTime()
      p.get()
      awaitNanos += System.nanoTime() - t
      pending = null

  /** the slots of `ids`, each there now (read in if not), none of `ids` put out for another of them */
  def slotsOf(ids: Array[Int]): Array[Int] =
    await()
    val wanted = ids.distinct
    require(wanted.length <= slots, s"${wanted.length} experts at once, $slots slots")
    if guess.nonEmpty then
      prefetchedUsed += wanted.count(guess)
      guess = Set.empty
    val t = System.nanoTime()
    bring(wanted, prefetching = false)
    missNanos += System.nanoTime() - t
    ids.map(slotOf)

  private def bring(wanted: Array[Int], prefetching: Boolean): Unit = synchronized {
    val keep = mutable.Set.from(wanted.flatMap(e => Option(slotOf(e)).filter(_ >= 0)))
    for e <- wanted do
      if slotOf(e) >= 0 then { if !prefetching then hits += 1 }
      else
        if prefetching then prefetched += 1 else misses += 1
        val s = order.find(s => !keep(s)).get
        if expertIn(s) >= 0 then slotOf(expertIn(s)) = -1
        var i = 0
        while i < parts.length do
          val p = parts(i)
          Mlx.readInto(fd(p.path), buffers(i), s * perExpert(i), p.offset + e * perExpert(i), perExpert(i))
          i += 1
        expertIn(s) = e
        slotOf(e) = s
        keep += s
      val s = slotOf(e)
      order -= s
      order += s
  }

  def close(): Unit =
    await()
    fds.values.foreach(Mlx.closeFd)
    fds.clear()

object ExpertCache:
  /** the reads ahead: one thread, a daemon */
  private val io = java.util.concurrent.Executors.newSingleThreadExecutor { r =>
    val t = Thread(r, "expert-prefetch")
    t.setDaemon(true)
    t
  }
