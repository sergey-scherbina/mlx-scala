package mlx.llm

import java.nio.file.{Files, Path}
import mlx.*

/** A LANGUAGE MODEL on MLX: the next token's logits after some, its state (caches) carried from call to call */
trait Model:
  def newState(): Model.State
  /** the next token's logits after `ids` [1, n], the state moved on by n */
  def logits(ids: Arr, state: Model.State)(using Scope): Arr
  def close(): Unit

object Model:
  /** what a model keeps between calls: caches */
  trait State:
    def close(): Unit

  /** the model of a directory, by its config.json's model_type */
  def load(dir: Path): Model =
    Json.parse(Files.readString(dir.resolve("config.json")))("model_type").str match
      case "qwen3" => Qwen3.load(dir)
      case "qwen3_5" | "qwen3_5_moe" => Qwen35.load(dir)
      case other => sys.error(s"a model type not supported: $other")
