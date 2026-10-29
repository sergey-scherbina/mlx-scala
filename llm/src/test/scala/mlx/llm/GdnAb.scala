package mlx.llm

import mlx.*
import mlx.Mlx.*

/** the generic and the packed delta-rule kernels alternated in one process: prefill of n tokens, `rounds` pairs */
@main def gdnAb(repo: String, n: Int, rounds: Int): Unit =
  val dir = Models.snapshot(repo).get
  val model = Model.load(dir)
  val chat = Chat.load(dir)
  val src = java.nio.file.Files.readString(
    java.nio.file.Path.of(sys.props("user.home"), "work/my/okay-rust/src/typed.rs"))
  val ids = chat.tokenizer.encode((src * 10).take(n * 4)).take(n)
  def once(): Double =
    val state = model.newState()
    val t = System.nanoTime()
    var i = 0
    while i < n do
      val m = math.min(2048, n - i)
      val s = new Scope
      given Scope = s
      model.logits(ints(ids.slice(i, i + m), 1, m), state).eval()
      s.close()
      i += m
    state.close()
    n / ((System.nanoTime() - t) / 1e9)
  Qwen35.packedAllowed = true; once(); Qwen35.packedAllowed = false; once()      // warm both
  val results = (1 to rounds).map { _ =>
    Qwen35.packedAllowed = false; val u = once()
    Qwen35.packedAllowed = true; val p = once()
    (u, p)
  }
  def median(xs: Seq[Double]) = xs.sorted.apply(xs.size / 2)
  println(results.map((u, p) => f"$u%.0f/$p%.0f").mkString("unpacked/packed: ", " ", ""))
  println(f"median unpacked ${median(results.map(_._1))}%.0f, packed ${median(results.map(_._2))}%.0f tok/s")
