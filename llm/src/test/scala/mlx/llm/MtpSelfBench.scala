package mlx.llm

import java.nio.file.{Files, Path, Paths}
import scala.jdk.CollectionConverters.*
import mlx.*

/** THE DRAFT LAYER TRAINED ON THE MODEL'S OWN SCALA (`MtpTrain`, its own part of each text): the model asked to go on
 * with the beginning of a .scala file of a training project, its answer (greedy) kept — cached in
 * ~/.cache/mlx-scala/mtp/<model>/self-<set>.tsv — and trained on; measured on answers to OTHER projects' files: the
 * first draft's agreement, and a real decode (drafts 2) with and without the adapters, alternated.
 *
 * `sbt "llmJVM/Test/runMain mlx.llm.mtpSelfBench mlx-community/Qwen3.5-4B-MLX-4bit 500 1500"` (texts, steps) */
@main def mtpSelfBench(repo: String, texts: Int, steps: Int): Unit =
  val dir = Models.snapshot(repo).get
  val model = Model.load(dir).asInstanceOf[Adaptable]
  val chat = Chat.load(dir)
  val home = Paths.get(sys.props("user.home"), "work", "my")
  def dirs(v: String, d: String) = sys.env.getOrElse(v, d).split(',').map(n => home.resolve(n.trim)).filter(
    Files.isDirectory(_)).toVector
  val train = dirs("MTP_TRAIN", "okay-scala,okay-simple,okay-store,okay-serverless,okay-faces,okay-ui,okay-guard," +
    "okay-agent,okay-llm,okay-mlx,okay-cardano,okay-uplc,okay-facts,okay-reliability")
  val held = dirs("MTP_HELD", "symdex,okay-datalog")
  def files(roots: Vector[Path]): Vector[Path] = roots.flatMap { r =>
    Files.walk(r).iterator.asScala.filter(p => p.toString.endsWith(".scala") && Files.isRegularFile(p) &&
      !p.toString.contains("/target/") && !p.toString.contains("/.")).toVector
  }
  val r = scala.util.Random(11)
  /** prompts: a file's beginning (150 tokens), the model asked to go on; answers greedy, 256 tokens at most */
  def prompts(fs: Vector[Path], n: Int): Vector[Array[Int]] =
    r.shuffle(fs).iterator.map(f => chat.tokenizer.encode(Files.readString(f))).filter(_.length > 400).take(n)
      .map { ids =>
        val head = chat.tokenizer.decode(ids.take(150).toIndexedSeq)
        chat.ask(s"Go on with this Scala file from my project. Code only.\n\n```scala\n$head", Map(
          "enable_thinking" -> false))
      }.toVector
  def answers(set: String, ps: Vector[Array[Int]]): Vector[(Array[Int], Array[Int])] =
    val f = Mtp.path(dir).resolveSibling(s"self-$set.tsv")
    val cached =
      if Files.exists(f) then Files.readAllLines(f).asScala.map { l =>
        val Array(a, b) = l.split('\t')
        (a.split(' ').map(_.toInt), b.split(' ').map(_.toInt))
      }.toVector
      else Vector.empty
    if cached.length >= ps.length then cached.take(ps.length)
    else
      val t0 = System.nanoTime()
      val made = ps.drop(cached.length).zipWithIndex.map { (p, k) =>
        val s = Session(model, p, 256, chat.stops, drafts = 2)
        val out = Iterator.continually(s.next()).takeWhile(_.isDefined).flatten.toArray
        s.close()
        if (k + 1) % 50 == 0 then println(f"  $set: ${cached.length + k + 1} of ${ps.length}, " +
          f"${(System.nanoTime() - t0) / 1e9}%.0f s")
        (p, out)
      }
      val all = cached ++ made
      Files.writeString(f, all.map((a, b) => a.mkString(" ") + "\t" + b.mkString(" ")).mkString("\n"))
      all
  val heldSet = answers("held", prompts(files(held), 40)).filter(_._2.length > 16)
  val trainSet = answers("train", prompts(files(train), texts)).filter(_._2.length > 16)
  println(s"$repo: ${trainSet.length} answers to train on, ${heldSet.length} held out")
  val t = MtpTrain(model)
  val keep = new Scope
  val heldPairs = heldSet.map((p, a) => t.pairs(p, a)(using keep))
  def agree(trained: Boolean) =
    heldPairs.map(t.agreement(_, trained)).sum * 100.0 / heldPairs.map(p => p.size - p.from).sum
  println(f"base: held-out agreement ${agree(false)}%.1f%%")
  val t0 = System.nanoTime()
  var losses = Vector.empty[Float]
  for k <- 0 until steps do
    val s = new Scope
    val (p, a) = trainSet(k % trainSet.length)
    losses :+= t.step(t.pairs(p, a)(using s))
    s.close()
    if (k + 1) % 250 == 0 || k + 1 == steps then
      println(f"step ${k + 1}: loss ${losses.sum / losses.length}%.3f, held-out agreement ${agree(true)}%.1f%%, " +
        f"${(System.nanoTime() - t0) / 1e9 / (k + 1)}%.2f s a step")
      losses = Vector.empty
  val out = Mtp.path(dir).resolveSibling("mtp-lora-self.safetensors")
  t.lora.save(out.toString)
  println(s"saved $out")
  // a real decode on held-out prompts, drafts 2, alternated
  val bench = heldSet.take(8).map(_._1)
  val speed = Map(false -> scala.collection.mutable.ArrayBuffer.empty[Double],
    true -> scala.collection.mutable.ArrayBuffer.empty[Double])
  val per = Map(false -> scala.collection.mutable.ArrayBuffer.empty[Double],
    true -> scala.collection.mutable.ArrayBuffer.empty[Double])
  for round <- 0 to 2; p <- bench; on <- Seq(false, true) do
    if on then t.lora.attachTrained() else t.lora.detach()
    val s = Session(model, p, 256, chat.stops, drafts = 2)
    val n = Iterator.continually(s.next()).takeWhile(_.isDefined).size
    s.close()
    if round > 0 then
      speed(on) += s.timing.decodePerSecond
      per(on) += n.toDouble / s.passes
  t.lora.detach()
  def med(xs: Iterable[Double]) = xs.toVector.sorted.apply(xs.size / 2)
  for on <- Seq(false, true) do
    println(f"decode, held-out, drafts 2, ${if on then "trained" else "base"}: ${med(speed(on))}%.1f tok/s, " +
      f"${med(per(on))}%.2f tokens a pass")
  keep.close()
  t.close()
  model.close()
