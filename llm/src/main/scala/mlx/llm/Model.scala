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
  /** the logits after each of `ids` [1, n]: [1, n, V], the state moved on by n — n guesses checked in one pass */
  def logitsAll(ids: Arr, state: Model.State)(using Scope): Arr
  /** the hidden states after the last norm, of the last of `ids` [1, n] or of each: [1, 1 or n, H], the state moved
   * on by n — the logits by `head`, a draft layer's input (`draft`) */
  def hiddenStates(ids: Arr, state: Model.State, all: Boolean)(using Scope): Arr = sys.error("no hidden states")
  /** the logits of hidden states [1, n, H]: [1, n, V] */
  def head(h: Arr)(using Scope): Arr = sys.error("no hidden states")
  /** a state for the model's draft layer (multi-token prediction: the next token's next guessed), or null: it has
   * none */
  def newDraft(): Model.Draft | Null = null
  /** the draft layer on hidden states `h` [1, n, H] (`hiddenStates`', or its own) and the token after each [1, n]: its
   * hidden states [1, n, H] — the logits of the token after those by `head` — the draft's state moved on by n */
  def draft(h: Arr, ids: Arr, d: Model.Draft)(using Scope): Arr = sys.error("no draft layer")
  def close(): Unit

object Model:
  /** what a model keeps between calls: caches */
  trait State:
    def close(): Unit
    /** the state as it is now, kept: a prompt's prefix not computed again (`PrefixCache`) */
    def snapshot(): Snapshot
    /** where it is now, to come back to (`rewind`): guesses checked, some of them wrong */
    def checkpoint(): Checkpoint
    /** back to `c` with `keep` of the tokens fed since kept: true — or false when only `c` itself can be come back
     * to (a recurrent state keeps no position but its last), the caller to feed those `keep` again */
    def rewind(c: Checkpoint, keep: Int): Boolean
  /** a draft layer's state: its positions so far, set back to drop the guesses past them */
  trait Draft:
    def offset: Int
    def offset_=(at: Int): Unit
    def close(): Unit
  /** a state's place, kept for a while */
  trait Checkpoint:
    def close(): Unit
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
  def load(dir: Path, mapped: Boolean = SafeTensors.mappedByDefault,
      inPlace: Boolean = SafeTensors.alignInPlace, expertSlots: Int = 0, mtp: Boolean = true): Model =
    modelType(dir).getOrElse(sys.error(s"no config.json in $dir")) match
      case "qwen3" => Qwen3.load(dir, mapped, inPlace)
      case "qwen3_5" | "qwen3_5_moe" => Qwen35.load(dir, mapped, inPlace, expertSlots, mtp)
      case other => sys.error(s"a model type not supported: $other")
