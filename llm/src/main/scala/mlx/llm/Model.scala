package mlx.llm

import okay.effects.json.Json

import java.nio.file.{Files, Path}
import mlx.*

/** A LANGUAGE MODEL on MLX: the next token's logits after some, its state (caches) carried from call to call */
trait Model:
  def newState(): Model.State
  /** a state as a snapshot of one was taken */
  def restore(snapshot: Model.Snapshot): Model.State
  /** a snapshot from its tensors and metadata (as `Snapshot.tensors`, `.meta` gave them: a file's), owned by `owner` */
  def snapshotOf(tensors: Map[String, Arr], meta: Map[String, String], owner: Scope): Model.Snapshot
  /** the next token's logits after `ids` [1, n], the state moved on by n */
  def logits(ids: Arr, state: Model.State)(using Scope): Arr
  def close(): Unit

object Model:
  /** what a model keeps between calls: caches */
  trait State:
    def close(): Unit
    /** the state as it is now, kept: a prompt's prefix not computed again (`PrefixCache`) */
    def snapshot(): Snapshot
  /** a state kept: what it holds on the GPU, about */
  trait Snapshot:
    def bytes: Long
    def close(): Unit
    /** what it holds, by name — the buffers cut to what they hold — and its numbers: for a file */
    def tensors(using Scope): Vector[(String, Arr)]
    def meta: Map[String, String]

  private def modelType(dir: Path): Option[String] =
    val f = dir.resolve("config.json")
    if !Files.exists(f) then None else Json.fromText(Files.readString(f)).field("model_type").map(_.string)

  /** whether `load` loads the model of a directory: its model_type one of those below */
  def loads(dir: Path): Boolean = modelType(dir).exists(Set("qwen3", "qwen3_5", "qwen3_5_moe"))

  /** the model of a directory, by its config.json's model_type; its weights read, or mapped (`SafeTensors`) */
  def load(dir: Path, mapped: Boolean = SafeTensors.mappedByDefault): Model =
    modelType(dir).getOrElse(sys.error(s"no config.json in $dir")) match
      case "qwen3" => Qwen3.load(dir, mapped)
      case "qwen3_5" | "qwen3_5_moe" => Qwen35.load(dir, mapped)
      case other => sys.error(s"a model type not supported: $other")
