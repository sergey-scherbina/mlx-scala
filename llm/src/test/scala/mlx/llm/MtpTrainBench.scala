package mlx.llm

import java.nio.file.{Files, Path, Paths}
import scala.jdk.CollectionConverters.*
import mlx.*

/** THE DRAFT LAYER TRAINED ON SCALA (`MtpTrain`): texts from the training projects' .scala files, measured on OTHER
 * projects — their .scala (the kind trained on) and their .md (prose: nothing lost) — by the first draft's agreement
 * with the model; then a real decode on Scala prompts, alternated, with and without the adapters. The adapters are
 * saved beside the pack (mtp-lora-scala.safetensors).
 *
 * `sbt "llmJVM/Test/runMain mlx.llm.mtpTrainBench mlx-community/Qwen3.5-4B-MLX-4bit 400 256"`
 * MTP_TRAIN (comma-separated directories), MTP_HELD (the same, held out) */
@main def mtpTrainBench(repo: String, steps: Int, chunk: Int): Unit =
  val dir = Models.snapshot(repo).get
  val model = Model.load(dir).asInstanceOf[Adaptable]
  val chat = Chat.load(dir)
  val home = Paths.get(sys.props("user.home"), "work", "my")
  def dirs(v: String, d: String) = sys.env.getOrElse(v, d).split(',').map(n => home.resolve(n.trim)).filter(
    Files.isDirectory(_)).toVector
  val train = dirs("MTP_TRAIN", "okay-scala,okay-simple,okay-store,okay-serverless,okay-faces,okay-ui,okay-guard," +
    "okay-agent,okay-llm,okay-mlx,okay-cardano,okay-uplc,okay-facts,okay-reliability")
  val held = dirs("MTP_HELD", "symdex,okay-datalog")
  def files(roots: Vector[Path], ext: String): Vector[Path] = roots.flatMap { r =>
    Files.walk(r).iterator.asScala.filter(p => p.toString.endsWith(ext) && Files.isRegularFile(p) &&
      !p.toString.contains("/target/") && !p.toString.contains("/.")).toVector
  }
  val r = scala.util.Random(7)
  def chunks(fs: Vector[Path]): Vector[Array[Int]] = r.shuffle(fs.flatMap { f =>
    val ids = chat.tokenizer.encode(Files.readString(f))
    ids.grouped(chunk).filter(_.length == chunk).toVector
  })
  val trainChunks = chunks(files(train, ".scala"))
  val heldCode = chunks(files(held, ".scala")).take(40)
  val heldProse = chunks(files(held, ".md")).take(20)
  println(s"$repo: ${trainChunks.length} training texts of $chunk tokens; held out ${heldCode.length} code, " +
    s"${heldProse.length} prose")
  val t = MtpTrain(model)
  println(s"adapters: ${t.lora.size} parameters on ${t.lora.names.mkString(", ")}")
  val keep = new Scope
  def pairsOf(cs: Vector[Array[Int]]) = cs.map(c => t.pairs(c)(using keep))
  val code = pairsOf(heldCode)
  val prose = pairsOf(heldProse)
  def agree(ps: Vector[t.Pairs], trained: Boolean) =
    ps.map(t.agreement(_, trained)).sum * 100.0 / ps.map(p => p.size - p.from).sum
  def report(at: Int) =
    println(f"step $at: held-out code ${agree(code, true)}%.1f%%, prose ${agree(prose, true)}%.1f%%")
  println(f"base: held-out code ${agree(code, false)}%.1f%%, prose ${agree(prose, false)}%.1f%%")
  val t0 = System.nanoTime()
  var losses = Vector.empty[Float]
  for k <- 0 until steps do
    val s = new Scope
    val p = t.pairs(trainChunks(k % trainChunks.length))(using s)
    losses :+= t.step(p)
    s.close()
    if (k + 1) % 100 == 0 || k + 1 == steps then
      println(f"  loss ${losses.sum / losses.length}%.3f over the last ${losses.length} steps, " +
        f"${(System.nanoTime() - t0) / 1e9 / (k + 1)}%.2f s a step")
      losses = Vector.empty
      report(k + 1)
  val out = Mtp.path(dir).resolveSibling("mtp-lora-scala.safetensors")
  t.lora.save(out.toString)
  println(s"saved $out")
  // a real decode: Scala asked for, drafts 2, with and without the adapters, alternated
  val prompts = Seq("Write a Scala 3 function that parses a CSV line with quoted fields into a List[String].",
    "Write a Scala 3 LRU cache class with get and put, and a short test in munit.",
    "Write a Scala 3 enum for a JSON value and a pretty printer for it.")
    .map(q => chat.ask(q, Map("enable_thinking" -> false)))
  val speeds = Map(false -> scala.collection.mutable.ArrayBuffer.empty[Double],
    true -> scala.collection.mutable.ArrayBuffer.empty[Double])
  val per = Map(false -> scala.collection.mutable.ArrayBuffer.empty[Double],
    true -> scala.collection.mutable.ArrayBuffer.empty[Double])
  for round <- 0 to 3; p <- prompts; on <- Seq(false, true) do
    if on then t.lora.attachTrained() else t.lora.detach()
    val s = Session(model, p, 300, chat.stops, drafts = 2)
    val n = Iterator.continually(s.next()).takeWhile(_.isDefined).size
    s.close()
    if round > 0 then
      speeds(on) += s.timing.decodePerSecond
      per(on) += n.toDouble / s.passes
  t.lora.detach()
  def med(xs: Iterable[Double]) = xs.toVector.sorted.apply(xs.size / 2)
  for on <- Seq(false, true) do
    println(f"decode, Scala, drafts 2, ${if on then "trained" else "base"}: ${med(speeds(on))}%.1f tok/s, " +
      f"${med(per(on))}%.2f tokens a pass")
  keep.close()
  t.close()
  model.close()
