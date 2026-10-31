package mlx.llm

/** guesses checked: the tokens greedy's, and some of the guesses taken — on a text that repeats what it read */
class GuessSuite extends munit.FunSuite:
  override val munitTimeout = scala.concurrent.duration.Duration(10, "min")
  private val code =
    """def fib(n: Int): Int =
      |  if n < 2 then n
      |  else fib(n - 1) + fib(n - 2)
      |
      |def sum(xs: List[Int]): Int = xs.foldLeft(0)(_ + _)
      |""".stripMargin

  for repo <- Seq("mlx-community/Qwen3-4B-4bit", "mlx-community/Qwen3.5-9B-MLX-4bit",
      "mlx-community/Qwen3.6-35B-A3B-4bit") do
    test(s"$repo: guessed, the same tokens"):
      val dir = Models.snapshot(repo)
      assume(dir.isDefined, s"$repo is not in the HuggingFace cache")
      val chat = Chat.load(dir.get)
      val model = Model.load(dir.get, mapped = true)
      val prompt = chat.ask(s"Rewrite this Scala code with `fib` renamed to `fibonacci`. Code only.\n\n$code",
        Map("enable_thinking" -> false))
      def run(guesses: Int, sampling: Sampling = Sampling.Greedy) =
        val s = Session(model, prompt, 80, chat.stops, sampling = sampling, guesses = guesses)
        val out = Iterator.continually(s.next()).takeWhile(_.isDefined).flatten.toVector
        s.close()
        (out, s)
      val (plain, p) = run(0)
      val (checked, c) = run(6)
      println(s"$repo: ${plain.length} tokens; ${c.passes} passes, ${c.taken} of ${c.guessed} guesses taken; " +
        f"decode ${p.timing.decodePerSecond}%.1f → ${c.timing.decodePerSecond}%.1f tok/s")
      assertEquals(checked, plain)
      assert(c.taken > 0 && c.passes < plain.length)
      // drawn, guesses checked by the draws: top_k 1 draws greedy's tokens
      assertEquals(run(6, Sampling(temperature = 1.0, topK = 1, seed = Some(1)))._1, plain)
      // the model's own sampling: guesses still taken
      val (_, d) = run(6, Sampling.of(dir.get).copy(seed = Some(1)))
      println(s"$repo drawn as its generation_config says: ${d.taken} of ${d.guessed} guesses taken, " +
        f"${d.timing.decodePerSecond}%.1f tok/s")
      model.close()
