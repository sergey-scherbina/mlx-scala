package mlx.llm

import mlx.*

/** how long a model takes to load, what MLX then holds, and its decode after: `llmJVM/Test/runMain
 * mlx.llm.loadProbe <repo>` (MLX_SCALA_MMAP=1: the weights mapped, not read) */
@main def loadProbe(repo: String): Unit =
  val dir = if repo.startsWith("/") then java.nio.file.Path.of(repo) else Models.snapshot(repo).get
  Mlx.init()
  val t0 = System.nanoTime()
  val slots = sys.env.get("EXPERT_SLOTS").map(_.toInt).getOrElse(0)
  val model = Model.load(dir, mapped = slots > 0 || SafeTensors.mappedByDefault, expertSlots = slots)
  val t1 = System.nanoTime()
  def footprint(): String = scala.sys.process.Process(Seq("footprint", ProcessHandle.current().pid.toString)).!!
    .linesIterator.filter(l => l.contains("Footprint:") || l.contains("IOAccelerator (") || l.contains("mapped file"))
    .map(_.trim).mkString(" | ")
  println(s"AFTER LOAD ${footprint()}")
  val chat = Chat.load(dir)
  val text = sys.env.get("PROBE_PROMPT").map(f => java.nio.file.Files.readString(java.nio.file.Path.of(f)))
    .getOrElse("Write a long story about a lighthouse keeper.")
  val prompt = chat.ask(text, Map("enable_thinking" -> false))
  println(s"PROMPT ${prompt.length} tokens")
  def decode(): (Double, Double) =
    val s = Session(model, prompt, 128, _ => false)
    Iterator.continually(s.next()).takeWhile(_.isDefined).foreach(_ => ())
    s.close()
    (s.timing.prefillNanos / 1e6, s.timing.decodePerSecond)
  val (first, _) = decode()
  Qwen35.waitNanos = 0L
  val (miss0, await0) = Qwen35.expertTimes(model)
  val t0d = System.nanoTime()
  val runs = (1 to 3).map(_ => decode())
  val total = (System.nanoTime() - t0d) / 1e6
  val (miss1, await1) = Qwen35.expertTimes(model)
  if slots > 0 then println(f"TIMES 3 runs $total%.0f ms: waiting for picks ${Qwen35.waitNanos / 1e6}%.0f ms, " +
    f"reading misses ${(miss1 - miss0) / 1e6}%.0f ms, waiting for reads ahead ${(await1 - await0) / 1e6}%.0f ms")
  val pid = ProcessHandle.current().pid
  val fp = scala.sys.process.Process(Seq("footprint", pid.toString)).!!.linesIterator
    .find(_.contains("phys_footprint:")).map(_.trim).getOrElse("?")
  val (hits, misses) = Qwen35.expertCounts(model)
  val (ahead, used) = Qwen35.expertsAhead(model)
  if slots > 0 then println(f"EXPERTS $slots a layer: hits ${100.0 * hits / math.max(1, hits + misses)}%.1f%%, " +
    s"misses $misses, read ahead $ahead, guesses picked $used")
  println(f"LOAD $repo slots=$slots mmap=${sys.env.getOrElse("MLX_SCALA_MMAP", "0")} load ${(t1 - t0) / 1e9}%.2f s, " +
    f"MLX holds ${Mlx.activeMemory / 1e9}%.1f GB, first prefill $first%.0f ms, decode " +
    runs.map(r => f"${r._2}%.1f").mkString(" / ") + s" tok/s, $fp")
  if sys.env.contains("PROBE_FOOTPRINT") then
    println(scala.sys.process.Process(Seq("footprint", pid.toString)).!!.linesIterator.take(25).mkString("\n"))
  model.close()
