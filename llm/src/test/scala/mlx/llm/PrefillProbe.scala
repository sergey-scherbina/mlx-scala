package mlx.llm

import mlx.*
import mlx.Mlx.*

/** prefill speed as mlx_lm's prefill.py measures it: n tokens in chunks of 2048, the state evaluated after each */
@main def prefillProbe(repo: String, sizes: Int*): Unit =
  val dir = Models.snapshot(repo).get
  val model = Model.load(dir)
  val chat = Chat.load(dir)
  val src = java.nio.file.Files.readString(
    java.nio.file.Path.of(sys.props("user.home"), "work/my/okay-rust/src/typed.rs"))
  for n <- sizes do
    val ids = chat.tokenizer.encode((src * 10).take(n * 4)).take(n)
    for r <- 0 until 3 do
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
      val dt = (System.nanoTime() - t) / 1e9
      if r > 0 then println(f"mlx-scala prefill $n: ${n / dt}%.0f tok/s")
      state.close()
    for r <- 0 until 3 do
      val t = System.nanoTime()
      val prompt = chat.ask(s"Here is code:\n${(src * 10).take(n * 4)}\nExplain.", Map("enable_thinking" -> false))
      val t1 = System.nanoTime()
      val session = Session(model, prompt, 8, chat.stops)
      session.next()
      val t2 = System.nanoTime()
      session.close()
      if r > 0 then println(f"session ${prompt.length}: tokenize ${(t1 - t) / 1e6}%.0f ms, " +
        f"first token ${(t2 - t1) / 1e6}%.0f ms (${prompt.length / ((t2 - t1) / 1e9)}%.0f tok/s)")
