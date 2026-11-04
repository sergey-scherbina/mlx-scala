package mlx.llm

import mlx.*
import mlx.Mlx.*

/** where a round of drafts spends its time: a step of one token, a pass of n checked (the state kept at each
 * position, then rewound), a draft step (its layer and the head) — medians of alternated runs */
@main def draftProbe(repo: String, reps: Int): Unit =
  val dir = Models.snapshot(repo).get
  val model = Model.load(dir)
  val chat = Chat.load(dir)
  val prompt = chat.ask("Write a long story about a lighthouse keeper.", Map("enable_thinking" -> false))
  val st = model.newState()
  val keep = new Scope
  locally {
    given Scope = keep
    model.logits(ints(prompt, 1, prompt.length), st).eval()
  }
  val d = model.newDraft()
  val h0 = Scope(keep)(model.hiddenStates(ints(Array(13), 1, 1), st, all = false).eval())
  def time(f: Scope ?=> Unit): Double =
    val t = System.nanoTime()
    val sc = new Scope
    f(using sc)
    sc.close()
    (System.nanoTime() - t) / 1e6
  val kinds = Seq("step 1") ++ (2 to 5).map(n => s"pass $n") ++ (if d == null then Nil else Seq("draft"))
  val times = kinds.map(_ -> scala.collection.mutable.ArrayBuffer.empty[Double]).toMap
  for _ <- 1 to reps; k <- kinds do
    val ms = k match
      case "step 1" =>
        time {
          val at = st.checkpoint()
          argmax(model.logits(ints(Array(13), 1, 1), st), -1).eval()
          st.rewind(at, 0)
          at.close()
        }
      case "draft" =>
        time {
          val o = d.nn.offset
          argmax(model.head(model.draft(h0, ints(Array(13), 1, 1), d.nn)), -1).eval()
          d.nn.offset = o
        }
      case pass =>
        val n = pass.stripPrefix("pass ").toInt
        time {
          val at = st.checkpoint()
          readInts(Seq(argmax(model.head(model.hiddenStates(ints(Array.fill(n)(13), 1, n), st, all = true)), -1)))
          st.rewind(at, 1)
          at.close()
        }
    times(k) += ms
  for k <- kinds do
    val s = times(k).sorted
    println(f"$k: median ${s(s.length / 2)}%.2f ms, min ${s.head}%.2f")
  keep.close()
  st.close()
  model.close()
