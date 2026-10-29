package mlx.llm

import mlx.*
import mlx.Mlx.*

/** sampling on logits made up: the candidates it may draw, how often, the same draws for the same seed; and on a
 * model, top_k 1 is greedy */
class SamplingSuite extends munit.FunSuite:
  Mlx.init()
  // logits of 6 tokens: softmax ≈ 0.64, 0.24, 0.09, 0.03, …
  private val logits = Array(3f, 2f, 1f, 0f, -1f, -2f)
  private def draws(sampling: Sampling, n: Int): Vector[Int] =
    val s = new Scope
    given Scope = s
    val l = astype(reshape(ints(Array(30, 20, 10, 0, -10, -20), 6), 1, 6), Dtype.Float32)
    val scaled = multiply(l, scalar(0.1f))
    val out = Vector.fill(n)(sampling.pick(scaled).eval().itemInt)
    s.close()
    out

  test("top_k: the k likeliest only"):
    assertEquals(draws(Sampling(1.0, topK = 2, seed = Some(1)), 300).toSet, Set(0, 1))

  test("top_p: the smallest set reaching p — a small p, the likeliest alone"):
    assertEquals(draws(Sampling(1.0, topP = 0.5), 200).toSet, Set(0))
    assert(draws(Sampling(1.0, topP = 0.9), 400).toSet.subsetOf(Set(0, 1, 2)))

  test("drawn as softmax says; the same seed, the same draws"):
    Mlx.randomSeed(7)
    val d = draws(Sampling(1.0), 2000)
    val p0 = d.count(_ == 0) / 2000.0
    assert(math.abs(p0 - 0.636) < 0.05, s"token 0 drawn ${p0 * 100}% of the time, softmax says 63.6%")
    Mlx.randomSeed(42)
    val a = draws(Sampling(1.0, topK = 4), 50)
    Mlx.randomSeed(42)
    val b = draws(Sampling(1.0, topK = 4), 50)
    assertEquals(a, b)

  test("penalties: a token made is less likely, the more the more often"):
    given s: Scope = new Scope
    val l = astype(reshape(ints(Array(30, 20, 10), 3), 1, 3), Dtype.Float32)
    val ids = ints(Array(0, 1, 2), 1, 3)
    val once = Sampling.counted(zeros(Seq(1, 3), Dtype.Float32), ids, ints(Array(0), 1, 1))
    val twice = Sampling.counted(once, ids, ints(Array(0), 1, 1))
    val p = Sampling(presencePenalty = 15).penalize(l, twice)
    assertEquals(Sampling.Greedy.pick(p).eval().itemInt, 1)
    val f = Sampling(frequencyPenalty = 6).penalize(l, twice)
    assertEquals(Sampling.Greedy.pick(f).eval().itemInt, 1)
    val f1 = Sampling(frequencyPenalty = 6).penalize(l, once)
    assertEquals(Sampling.Greedy.pick(f1).eval().itemInt, 0)
    s.close()

  test("a penalty on a model: no token over and over"):
    val dir = Models.snapshot("mlx-community/Qwen3-4B-4bit")
    assume(dir.isDefined, "mlx-community/Qwen3-4B-4bit is not in the HuggingFace cache")
    val chat = Chat.load(dir.get)
    val model = Model.load(dir.get)
    val prompt = chat.ask("Write the word okay forty times, separated by spaces.", Map("enable_thinking" -> false))
    def run(s: Sampling) =
      val session = Session(model, prompt, 48, chat.stops, sampling = s)
      val out = Iterator.continually(session.next()).takeWhile(_.isDefined).flatten.toVector
      session.close()
      out
    val plain = run(Sampling.Greedy)
    // asked to repeat, its margin is large: a penalty that large to show it counts
    val penalized = run(Sampling(presencePenalty = 100))
    def top(v: Vector[Int]) = v.groupBy(identity).values.map(_.size).max
    assert(top(penalized) < top(plain), s"most repeated: ${top(penalized)} with the penalty, ${top(plain)} without")
    model.close()

  test("temperature 0 and top_k 1 are greedy, on a model"):
    val dir = Models.snapshot("mlx-community/Qwen3-4B-4bit")
    assume(dir.isDefined, "mlx-community/Qwen3-4B-4bit is not in the HuggingFace cache")
    val chat = Chat.load(dir.get)
    val model = Model.load(dir.get)
    val prompt = chat.ask("Name three colors.", Map("enable_thinking" -> false))
    def run(s: Sampling) =
      val session = Session(model, prompt, 16, chat.stops, sampling = s)
      val out = Iterator.continually(session.next()).takeWhile(_.isDefined).flatten.toVector
      session.close()
      out
    assertEquals(run(Sampling(1.0, topK = 1, seed = Some(3))), run(Sampling.Greedy))
    model.close()
