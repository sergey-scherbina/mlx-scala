package mlx.llm

import mlx.*
import mlx.Mlx.*

/** the draft layer teacher-forced on a text the model wrote: its pick after the hidden state at i + s and the token
 * at i + 1, against the tokens at i + 1, i + 2, i + 3 */
@main def mtpProbe(repo: String): Unit =
  val dir = Models.snapshot(repo).get
  val model = Model.load(dir).asInstanceOf[Qwen35]
  val chat = Chat.load(dir)
  val prompt = chat.ask("Explain in five sentences how a lighthouse works.", Map("enable_thinking" -> false))
  val s = Session(model, prompt, 100, chat.stops)
  val out = Iterator.continually(s.next()).takeWhile(_.isDefined).flatten.toArray
  s.close()
  val seq = prompt ++ out
  val n = seq.length
  for shift <- Seq(-1, 0, 1) do
    val sc = new Scope
    given Scope = sc
    val st = model.newState()
    val h = model.hiddenStates(ints(seq, 1, n), st, all = true)
    val d = model.newDraft().nn
    // pairs i = 1 .. n − 3: the hidden state at i + shift, the token at i + 1
    val m = n - 3
    val hs = slice(h, Seq(0, 1 + shift, 0), Seq(1, 1 + shift + m, h.dim(2)))
    val o = model.draft(hs, ints(seq.slice(2, 2 + m), 1, m), d)
    val picks = argmax(model.head(o), -1).toInts // picks(j) for i = j + 1
    val from = prompt.length
    def at(t: Int) = (from until n - 3).count(i => picks(i - 1) == seq(i + t)) * 100 / (n - 3 - from)
    println(s"shift=$shift: →i+1 ${at(1)}%, i+2 ${at(2)}%, i+3 ${at(3)}%")
    if shift == 0 then
      println("text:  " + chat.tokenizer.decode(seq.slice(from + 1, from + 41).toIndexedSeq).replace("\n", " "))
      println("draft: " + (from until from + 40).map(i => chat.tokenizer.decode(Seq(picks(i - 1)))).mkString("|"))
      def rms(a: Arr) =
        val f = astype(a, Dtype.Float32)
        math.sqrt(sum(reshape(multiply(f, f), -1), 0).eval().itemFloat / a.shape.product)
      println(f"rms h ${rms(h)}%.3f o ${rms(o)}%.3f")
    d.close()
    st.close()
    sc.close()
  model.close()
