package mlx.llm

import java.nio.file.Files
import mlx.*
import mlx.Mlx.*

/** the model's own drafts (multi-token prediction, `Mtp`): greedy's tokens — but for near ties, which a pass of
 * several tokens computes a bf16 step apart from a step of one — and most drafts taken; on models whose pack was
 * made (`mtpFetch`) */
class MtpSuite extends munit.FunSuite:
  override val munitTimeout = scala.concurrent.duration.Duration(20, "min")

  for repo <- Seq("mlx-community/Qwen3.5-4B-MLX-4bit", "mlx-community/Qwen3.6-35B-A3B-4bit") do
    test(s"$repo: drafted, greedy's tokens"):
      val dir = Models.snapshot(repo)
      assume(dir.isDefined, s"$repo is not in the HuggingFace cache")
      assume(Files.exists(Mtp.path(dir.get)), s"no draft pack for $repo (mtpFetch)")
      val chat = Chat.load(dir.get)
      val model = Model.load(dir.get)
      assert(model.newDraft() != null)
      val prompt = chat.ask("Explain in five sentences how a lighthouse works.", Map("enable_thinking" -> false))
      def run(drafts: Int, sampling: Sampling = Sampling.Greedy, n: Int = 120) =
        val s = Session(model, prompt, n, chat.stops, sampling = sampling, drafts = drafts)
        val out = Iterator.continually(s.next()).takeWhile(_.isDefined).flatten.toVector
        s.close()
        (out, s)
      /** each token at most four bf16 steps (at the best's magnitude) below the best, the text computed in one pass —
       * rows of a pass of 2 against a pass of the whole text: a mixture of experts' near-tie picks part them by more
       * than a step (35B-A3B: 3 steps once); a wrong state parts them by whole units, and the text after */
      def greedy(out: Vector[Int], what: String): Unit =
        val sc = new Scope
        given Scope = sc
        val st = model.newState()
        val all = prompt ++ out
        val l = astype(model.logitsAll(ints(all, 1, all.length), st), Dtype.Float32)
        val v = l.dim(-1)
        val rows = slice(reshape(l, all.length, v), Seq(prompt.length - 1, 0), Seq(all.length - 1, v))
        val best = max(rows, 1).toFloats
        val took = takeAlong(rows, ints(out.toArray, out.length, 1), 1).toFloats
        def tie(b: Float) = 4 * math.pow(2, math.floor(math.log(math.abs(b).max(1e-3f)) / math.log(2)) - 7)
        val worst = out.indices.maxBy(i => (best(i) - took(i)) / tie(best(i)))
        val below = best(worst) - took(worst)
        println(f"$repo $what: at most ${below / tie(best(worst)) * 4}%.1f bf16 steps below the best")
        assert(below <= tie(best(worst)), s"$what: token $worst $below below the best ${best(worst)}")
        st.close()
        sc.close()
      run(0, n = 8)
      run(3, n = 8) // warmed up
      val (plain, p) = run(0)
      greedy(plain, "plain")
      for k <- Seq(1, 2, 3, 4) do
        val (drafted, d) = run(k)
        println(f"$repo k=$k: ${d.taken} of ${d.guessed} drafts taken (${d.taken.toDouble / d.guessed * 100}%.0f%%), " +
          f"${drafted.length.toDouble / d.passes}%.2f tokens a pass; decode ${p.timing.decodePerSecond}%.1f → " +
          f"${d.timing.decodePerSecond}%.1f tok/s")
        greedy(drafted, s"k=$k")
        assert(drafted.length > 1.5 * d.passes, s"few drafts taken: ${d.taken} of ${d.guessed}")
      // drawn: top_k 1 draws greedy's; the model's own sampling, drafts still taken
      greedy(run(3, Sampling(temperature = 1.0, topK = 1, seed = Some(1)))._1, "top_k 1")
      val (_, s) = run(3, Sampling.of(dir.get).copy(seed = Some(1)))
      println(f"$repo drawn as its generation_config says: ${s.taken} of ${s.guessed} taken, " +
        f"${s.timing.decodePerSecond}%.1f tok/s")
      assert(s.taken > 0)
      model.close()
