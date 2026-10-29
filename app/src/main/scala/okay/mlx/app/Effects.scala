package okay.mlx.app

import java.nio.file.{Files as NioFiles, Path}
import okay.scala3.*
import okay.mlx.llm.*

/** A GENERATION: the text, and what it measured */
final case class Generation(text: String, timing: Timing)

/** THE LANGUAGE MODEL, as the app sees it: text in, text out — which model and on what is the handler's */
enum Llm[X]:
  case Complete(prompt: String, maxTokens: Int) extends Llm[Generation]

object Llm:
  def complete[R[_]](prompt: String, maxTokens: Int)(using In[Llm, R]): Generation ! R =
    perform[Llm](Complete(prompt, maxTokens))

  /** Qwen3 on MLX: the prompt as one chat turn, greedy, to the end of the turn */
  def mlx[R[_], Rest[_], A](model: Qwen3, tokenizer: Tokenizer)(p: A ! R)(using Split[Llm, R, Rest]): A ! Rest =
    val end = Set(model.config.eos, tokenizer.id("<|im_end|>"))
    handle(p, Handler.tail[Llm, Rest, A, A](pure)([X] => (op: Llm[X]) => op match
      case Complete(prompt, maxTokens) =>
        val ids = Vector.newBuilder[Int]
        val timing = Generate.greedy(model, tokenizer.encode(Generate.chat(prompt)), maxTokens, end)(ids += _)
        Generation(tokenizer.decode(ids.result().filterNot(end)), timing)))

  /** an answer for every prompt, and timings made up: the app's logic without a GPU */
  def fake[R[_], Rest[_], A](answer: String => String)(p: A ! R)(using Split[Llm, R, Rest]): A ! Rest =
    handle(p, Handler.tail[Llm, Rest, A, A](pure)([X] => (op: Llm[X]) => op match
      case Complete(prompt, maxTokens) =>
        Generation(answer(prompt), Timing(prompt.length / 4, 1_000_000L, maxTokens, maxTokens * 10_000_000L))))

/** FILES: a model's snapshot in the HuggingFace cache, a text read */
enum Files[X]:
  case Snapshot(repo: String) extends Files[Option[Path]]
  case Read(path: Path) extends Files[String]

object Files:
  def snapshot[R[_]](repo: String)(using In[Files, R]): Option[Path] ! R = perform[Files](Snapshot(repo))
  def read[R[_]](path: Path)(using In[Files, R]): String ! R = perform[Files](Read(path))

  /** the disk: the HuggingFace cache where HF_HOME / HF_HUB_CACHE say, else ~/.cache/huggingface/hub */
  def disk[R[_], Rest[_], A](p: A ! R)(using Split[Files, R, Rest]): A ! Rest =
    handle(p, Handler.tail[Files, Rest, A, A](pure)([X] => (op: Files[X]) => op match
      case Snapshot(repo) => Models.snapshot(repo)
      case Read(path) => NioFiles.readString(path)))

  /** files given: a map of paths to texts */
  def of[R[_], Rest[_], A](files: Map[Path, String], snapshots: Map[String, Path] = Map.empty)(p: A ! R)(
    using Split[Files, R, Rest]): A ! Rest =
    handle(p, Handler.tail[Files, Rest, A, A](pure)([X] => (op: Files[X]) => op match
      case Snapshot(repo) => snapshots.get(repo)
      case Read(path) => files.getOrElse(path, throw java.io.FileNotFoundException(path.toString))))

/** THE CONSOLE: a line out */
enum Console[X]:
  case Line(text: String) extends Console[Unit]

object Console:
  def line[R[_]](text: String)(using In[Console, R]): Unit ! R = perform[Console](Line(text))

  def stdout[R[_], Rest[_], A](p: A ! R)(using Split[Console, R, Rest]): A ! Rest =
    handle(p, Handler.tail[Console, Rest, A, A](pure)([X] => (op: Console[X]) => op match
      case Line(text) => println(text)))

  /** the lines, in order, and the value */
  def collect[R[_], Rest[_], A](p: A ! R)(using Split[Console, R, Rest]): (Vector[String], A) ! Rest =
    handle(p, Handler.stateful[Console, Rest, A, (Vector[String], A), Vector[String]](Vector.empty)(
      (lines, a) => pure((lines, a)))(
      [X] => (op: Console[X], cell: Cell[Vector[String]]) => op match
        case Line(text) => cell.set(cell.get :+ text)))
