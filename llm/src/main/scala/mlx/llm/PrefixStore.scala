package mlx.llm

import java.nio.file.{Files, Path}
import scala.collection.mutable
import okay.effects.json.Json
import mlx.*

/** PREFIXES ON DISK: snapshots of a model's state written as .safetensors — the layers' tensors, the prefix's tokens
 * in the metadata — in a directory of the model's own, so a prefix computed before a restart is not computed again.
 * The index is the files' headers, read at start (no tensor loaded); a snapshot is loaded when a prompt wants it.
 * Prefixes shorter than `minTokens` are not written (computing them is cheaper than reading); the oldest files go
 * when they hold more than `budget` bytes */
final class PrefixStore(val dir: Path, model: Model, val budget: Long = 4L << 30, val minTokens: Int = 1024):
  Files.createDirectories(dir)
  private final class Entry(val tokens: Array[Int], val path: Path)
  private val index = mutable.ArrayBuffer.from(
    Files.list(dir).toArray.map(_.asInstanceOf[Path]).filter(_.toString.endsWith(".safetensors"))
      .flatMap(p => PrefixStore.tokensOf(p).map(Entry(_, p))))

  def size: Int = index.size

  /** the longest prefix on disk that `prompt` begins with, shorter than it */
  def best(prompt: Array[Int], longerThan: Int): Option[Int] =
    index.iterator.filter(e => e.tokens.length > longerThan && e.tokens.length < prompt.length &&
      PrefixCache.startsWith(prompt, e.tokens)).map(_.tokens.length).maxOption

  /** the snapshot of that prefix, loaded */
  def load(prompt: Array[Int], length: Int): Option[(Array[Int], Model.Snapshot)] =
    index.find(e => e.tokens.length == length && PrefixCache.startsWith(prompt, e.tokens)).flatMap { e =>
      try
        val owner = new Scope
        given Scope = owner
        val tensors = Mlx.loadSafetensors(e.path.toString)
        Files.setLastModifiedTime(e.path, java.nio.file.attribute.FileTime.fromMillis(System.currentTimeMillis()))
        Some((e.tokens, model.snapshotOf(tensors, PrefixStore.metaOf(e.path), owner)))
      catch case err: RuntimeException =>
        System.err.println(s"prefix store: ${e.path} unread ($err), dropped")
        index -= e
        Files.deleteIfExists(e.path)
        None
    }

  /** a prefix written, unless short or there already */
  def save(tokens: Array[Int], snapshot: Model.Snapshot): Unit =
    if tokens.length >= minTokens && !index.exists(_.tokens.sameElements(tokens)) then
      val path = dir.resolve(f"${java.util.Arrays.hashCode(tokens)}%08x-${tokens.length}.safetensors")
      val s = new Scope
      given Scope = s
      try
        Mlx.saveSafetensors(path.toString, snapshot.tensors,
          snapshot.meta + ("tokens" -> tokens.mkString(",")) + ("mlx-scala" -> "prefix 1"))
        index += Entry(tokens, path)
      finally s.close()
      // the oldest first, until the files fit the budget
      val files = index.sortBy(e => Files.getLastModifiedTime(e.path).toMillis)
      var total = index.iterator.map(e => Files.size(e.path)).sum
      for e <- files if total > budget do
        total -= Files.size(e.path)
        Files.deleteIfExists(e.path)
        index -= e

object PrefixStore:
  /** a directory for a model's prefixes: under `root`, by the model's directory's name */
  def forModel(root: Path, modelDir: Path, model: Model): PrefixStore =
    val snap = modelDir.getFileName.toString
    val repo = Option(modelDir.getParent).flatMap(p => Option(p.getParent)).map(_.getFileName.toString).getOrElse("")
    PrefixStore(root.resolve(s"$repo-$snap".replaceAll("[^A-Za-z0-9._-]", "_")), model)

  /** a .safetensors file's metadata: its header's `__metadata__`, read without the tensors */
  private def header(p: Path): Option[Map[String, String]] =
    try
      val in = Files.newInputStream(p)
      try
        val len = java.nio.ByteBuffer.wrap(in.readNBytes(8)).order(java.nio.ByteOrder.LITTLE_ENDIAN).getLong
        if len <= 0 || len > (64L << 20) then None
        else
          val h = Json.fromText(String(in.readNBytes(len.toInt), "UTF-8"))
          Some(h("__metadata__") match
            case Json.Obj(f) => f.collect { case (k, Json.Str(v)) => k -> v }.toMap
            case _ => Map.empty)
      finally in.close()
    catch case _: Exception => None
  private def metaOf(p: Path): Map[String, String] = header(p).getOrElse(Map.empty)
  private def tokensOf(p: Path): Option[Array[Int]] =
    header(p).filter(_.get("mlx-scala").contains("prefix 1")).flatMap(_.get("tokens"))
      .map(t => if t.isEmpty then Array.empty[Int] else t.split(',').map(_.toInt))
