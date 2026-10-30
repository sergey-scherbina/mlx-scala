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

  /** the slots of `ids`, each there now (read in if not), none of `ids` put out for another of them */
  def slotsOf(ids: Array[Int]): Array[Int] =
    val wanted = ids.distinct
    require(wanted.length <= slots, s"${wanted.length} experts at once, $slots slots")
    val keep = mutable.Set.from(wanted.flatMap(e => Option(slotOf(e)).filter(_ >= 0)))
    for e <- wanted do
      if slotOf(e) >= 0 then hits += 1
      else
        misses += 1
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
    ids.map(slotOf)

  def close(): Unit =
    fds.values.foreach(Mlx.closeFd)
    fds.clear()
