package mlx.llm

import mlx.*
import mlx.Mlx.*

/** where a decode step's time goes: the graph built on the CPU, the whole step (built, run, read) */
@main def decodeProbe(repo: String, steps: Int): Unit =
  val dir = Models.snapshot(repo).get
  val model = Model.load(dir)
  val chat = Chat.load(dir)
  val prompt = chat.ask("Write a long story about a lighthouse keeper.", Map("enable_thinking" -> false))
  locally {
    val state = model.newState()
    val s0 = new Scope
    given Scope = s0
    val y = reshape(argmax(model.logits(ints(prompt, 1, prompt.length), state), -1), 1, 1).eval()
    exportDot(model.logits(y, state), "/tmp/step-scala.dot")
    s0.close()
    state.close()
  }
  for round <- 1 to 1 do
    val state = model.newState()
    val s0 = new Scope
    given Scope = s0
    var y = reshape(argmax(model.logits(ints(prompt, 1, prompt.length), state), -1), 1, 1).eval()
    var build = 0L
    val t0 = System.nanoTime()
    for _ <- 1 to steps do
      val b0 = System.nanoTime()
      val next = Scope(s0)(reshape(argmax(model.logits(y, state), -1), 1, 1))
      build += System.nanoTime() - b0
      next.eval()
      s0.release(y)
      y = next
    val total = System.nanoTime() - t0
    println(f"round $round: ${steps / (total / 1e9)}%.1f tok/s, graph built ${build / 1e6 / steps}%.2f ms of " +
      f"${total / 1e6 / steps}%.2f ms a step (no overlap)")
    s0.close()
    state.close()
  for round <- 1 to 6 do
    val session = Session(model, prompt, steps + 1, _ => false)
    Iterator.continually(session.next()).takeWhile(_.isDefined).foreach(_ => ())
    println(f"session: ${session.timing.decodePerSecond}%.1f tok/s, active ${Mlx.activeMemory / 1e9}%.2f GB, " +
      f"peak ${Mlx.peakMemory / 1e9}%.2f GB")
    session.close()
