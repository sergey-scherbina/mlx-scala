package mlx.llm

import mlx.*
import mlx.Mlx.*

/** the prefill curve: n tokens of one text, in steps of `step`, the state evaluated after each — tokens a second
 * (the best of 2 after a warm-up) and MLX's peak memory */
@main def prefillCurve(repo: String): Unit =
  val dir = Models.snapshot(repo).get
  val model = Model.load(dir)
  val chat = Chat.load(dir)
  val src = java.nio.file.Files.readString(
    java.nio.file.Path.of(sys.props("user.home"), "work/my/okay-rust/src/typed.rs"))
  val all = chat.tokenizer.encode(src * 40)
  def once(n: Int, step: Int): Double =
    val state = model.newState()
    val t = System.nanoTime()
    var i = 0
    while i < n do
      val m = math.min(step, n - i)
      val s = new Scope
      given Scope = s
      model.logits(ints(all.slice(i, i + m), 1, m), state).eval()
      s.close()
      i += m
    state.close()
    n / ((System.nanoTime() - t) / 1e9)
  once(2048, 2048)
  for step <- Seq(512, 1024, 2048, 4096) do
    val row = for n <- Seq(2048, 5120, 10240, 20480) yield
      val best = (1 to 2).map(_ => once(n, step)).max
      f"$n%6d: $best%5.0f"
    println(f"step $step%5d  " + row.mkString("  ") + f"  peak ${Mlx.peakMemory / 1e9}%.1f GB")
