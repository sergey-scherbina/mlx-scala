package mlx.llm

/** a session's decode with drafts (k) and without, alternated, medians: tok/s, tokens a pass */
@main def draftBench(repo: String, reps: Int, tokens: Int): Unit =
  val dir = Models.snapshot(repo).get
  val model = Model.load(dir)
  val chat = Chat.load(dir)
  val prompt = chat.ask("Explain in detail how a lighthouse works, its history and its parts.",
    Map("enable_thinking" -> false))
  val ks = Seq(0, 1, 2, 3)
  val speed = ks.map(_ -> scala.collection.mutable.ArrayBuffer.empty[Double]).toMap
  val perPass = ks.map(_ -> scala.collection.mutable.ArrayBuffer.empty[Double]).toMap
  for r <- 0 to reps; k <- ks do
    val s = Session(model, prompt, tokens, _ => false, drafts = k)
    val n = Iterator.continually(s.next()).takeWhile(_.isDefined).size
    s.close()
    if r > 0 then
      speed(k) += s.timing.decodePerSecond
      perPass(k) += (if s.passes == 0 then 1.0 else n.toDouble / s.passes)
  def med(xs: Seq[Double]) = xs.sorted.apply(xs.length / 2)
  for k <- ks do println(f"k=$k: ${med(speed(k).toSeq)}%.1f tok/s, ${med(perPass(k).toSeq)}%.2f tokens a pass")
  model.close()
