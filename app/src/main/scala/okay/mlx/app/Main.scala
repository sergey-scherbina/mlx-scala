package okay.mlx.app

import java.nio.file.Path
import okay.scala3.*
import okay.scala3.std.*
import okay.mlx.llm.*

/** `ask <question>` — one answer; `bench [runs] [source file]` — prefill and decode speed at three prompt sizes.
 * The model: mlx-community/Qwen3-4B-4bit, or MLX_MODEL */
@main def mlxApp(args: String*): Unit =
  val repo = sys.env.getOrElse("MLX_MODEL", "mlx-community/Qwen3-4B-4bit")
  val platform = if System.getProperty("java.vm.name", "").contains("Scala Native") then "native" else "jvm"

  // the model's files, found and read through Files
  type Find[X] = (Files + Log)[X]
  def find: (Path, String) ! Find =
    for
      dir <- Files.snapshot[Find](repo).map(_.getOrElse(sys.error(s"$repo is not in the HuggingFace cache")))
      _ <- Log.info[Find](s"$repo at $dir")
      tok <- Files.read[Find](dir.resolve("tokenizer.json"))
    yield (dir, tok)
  val stderr = (l: Level, m: String) => System.err.println(s"[$l] $m")
  val (dir, tokJson) = run(Log.to(stderr, Level.Info)(Files.disk(find)))
  val tokenizer = Tokenizer.fromJson(Json.parse(tokJson))
  val model = Qwen3.load(dir)

  type App[X] = (Llm + Console + Clock + Log)[X]
  def go[A](p: A ! App): A = run(Log.to(stderr, Level.Info)(Clock.system(Console.stdout(Llm.mlx(model, tokenizer)(p)))))
  args.toList match
    case "ask" :: words => go(Programs.ask[App](words.mkString(" "), 256))
    case "bench" :: rest =>
      val runs = rest.headOption.map(_.toInt).getOrElse(3)
      val source = rest.drop(1).headOption.map(Path.of(_))
        .getOrElse(Path.of(sys.props("user.home"), "work/my/okay-rust/src/typed.rs"))
      val text = run(Files.disk(Files.read[Files](source)))
      go(Programs.ask[App]("Say hello.", 8))     // warm-up: the kernels compiled
      go(Programs.bench[App](s"mlx-scala-$platform", text, Seq(400, 16000, 32000), runs, 256))
    case _ => System.err.println("usage: ask <question> | bench [runs] [source file]")
  model.close()
