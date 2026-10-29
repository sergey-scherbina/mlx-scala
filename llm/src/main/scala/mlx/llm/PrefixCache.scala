package mlx.llm

import scala.collection.mutable

/** PROMPTS' PREFIXES KEPT: a model's state snapshot after a prompt's first n tokens, so a prompt beginning with them is
 * computed from there — an agent sends the same system prompt and tools, and the conversation so far, every turn.
 * Snapshots are taken at marks (`Chat.marks`: where messages begin), not anywhere: a linear layer's recurrent state
 * (Qwen3.5, Qwen3.6) cannot be cut back to a shorter prefix, so a prefix is reused whole or not at all. The least
 * recently used go first when the snapshots hold more than `budget` bytes. With a `store`, kept on disk too: a prefix
 * longer than any in memory is loaded from there */
final class PrefixCache(val budget: Long = 4L << 30, val store: PrefixStore | Null = null):
  private final class Entry(val tokens: Array[Int], val snapshot: Model.Snapshot)
  private val entries = mutable.ArrayBuffer.empty[Entry]     // the most recently used last

  def size: Int = entries.size
  def bytes: Long = entries.iterator.map(_.snapshot.bytes).sum

  /** the longest prefix kept that `prompt` begins with, shorter than it: its length and snapshot */
  def lookup(prompt: Array[Int]): Option[(Int, Model.Snapshot)] =
    val best = entries.filter(e => e.tokens.length < prompt.length && PrefixCache.startsWith(prompt, e.tokens))
      .maxByOption(_.tokens.length)
    val inMemory = best.fold(0)(_.tokens.length)
    val fromDisk =
      if store == null then None
      else store.nn.best(prompt, inMemory).flatMap(n => store.nn.load(prompt, n)).map { (tokens, snap) =>
        keep(tokens, snap)
        (tokens.length, snap)
      }
    fromDisk.orElse(best.map { e =>
      entries -= e
      entries += e
      (e.tokens.length, e.snapshot)
    })

  /** a prefix's snapshot kept (the cache owns it now); written to the store too when `durable` — a prompt's first
   * mark (the system prompt and the tools, the same across conversations and restarts), not each turn's (a long
   * conversation would write hundreds of megabytes a turn) */
  def add(tokens: Array[Int], snapshot: Model.Snapshot, durable: Boolean = false): Unit =
    keep(tokens, snapshot)
    if durable && store != null then store.nn.save(tokens, snapshot)

  private def keep(tokens: Array[Int], snapshot: Model.Snapshot): Unit =
    entries.indexWhere(e => e.tokens.sameElements(tokens)) match
      case -1 =>
      case i => entries.remove(i).snapshot.close()
    entries += Entry(tokens, snapshot)
    while bytes > budget && entries.size > 1 do entries.remove(0).snapshot.close()

  def clear(): Unit =
    entries.foreach(_.snapshot.close())
    entries.clear()

object PrefixCache:
  def startsWith(a: Array[Int], prefix: Array[Int]): Boolean =
    var i = 0
    while i < prefix.length && a(i) == prefix(i) do i += 1
    i == prefix.length
