package mlx.llm

/** Qwen3-Embedding-0.6B: the model card's example (two queries, two documents: [[0.7646, 0.1414], [0.1355, 0.6000]]
 * at full precision) — this 4-bit DWQ model as mlx_lm computes it (its `model.model`, the last token, L2-normed):
 * [[0.7990, 0.1725], [0.2004, 0.6187]]; each query nearer its own document. The same vectors on the CPU as on
 * the GPU */
class EmbedderSuite extends munit.FunSuite:
  override val munitTimeout = scala.concurrent.duration.Duration(10, "min")
  private val repo = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"

  test(s"$repo: the model card's example; CPU as GPU"):
    val dir = Models.snapshot(repo)
    assume(dir.isDefined, s"$repo is not in the HuggingFace cache")
    val e = Embedder.load(dir.get)
    val task = "Given a web search query, retrieve relevant passages that answer the query"
    val docs = Seq("The capital of China is Beijing.",
      "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is " +
        "responsible for the movement of planets around the sun.")
    def scores(x: Embedder) =
      val qs = Seq("What is the capital of China?", "Explain gravity").map(x.query(task, _))
      val ds = x.vectors(docs)
      qs.map(q => ds.map(Embedder.cosine(q, _)))
    val gpu = scores(e)
    println(f"$repo: ${gpu.map(_.map(s => f"$s%.4f").mkString("[", ", ", "]")).mkString("[", ", ", "]")}")
    val mlxLm = Seq(Seq(0.7990, 0.1725), Seq(0.2004, 0.6187))
    for i <- 0 to 1; j <- 0 to 1 do
      assert(math.abs(gpu(i)(j) - mlxLm(i)(j)) < 0.01, s"[$i][$j] ${gpu(i)(j)} vs mlx_lm's ${mlxLm(i)(j)}")
    assert(gpu(0)(0) > gpu(0)(1) && gpu(1)(1) > gpu(1)(0))
    val onCpu = Embedder.load(dir.get, cpu = true)
    val cpu = scores(onCpu)
    for i <- 0 to 1; j <- 0 to 1 do
      assert(math.abs(gpu(i)(j) - cpu(i)(j)) < 0.01, s"[$i][$j] GPU ${gpu(i)(j)}, CPU ${cpu(i)(j)}")
    // timing: a text of ~200 tokens, each way, after a warm-up
    val text = "def parse(s: String): Option[Json] = " * 25
    for (name, x) <- Seq("GPU" -> e, "CPU" -> onCpu) do
      x.vectors(Seq(text))
      val t0 = System.nanoTime()
      for _ <- 1 to 10 do x.vectors(Seq(text))
      println(f"$name: ${(System.nanoTime() - t0) / 1e6 / 10}%.1f ms a text of ~200 tokens")
    e.model.close()
    onCpu.model.close()

/** the code-search adapter (`embedTrainBench`) put on: the same on the CPU as on the GPU, and not the base's */
class CodeSearchSuite extends munit.FunSuite:
  override val munitTimeout = scala.concurrent.duration.Duration(10, "min")
  test("the code-search adapter, CPU as GPU"):
    val dir = Models.snapshot("mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ")
    assume(dir.isDefined, "Qwen3-Embedding-0.6B is not in the HuggingFace cache")
    val adapter = Embedder.codeSearch(dir.get)
    assume(adapter.isDefined, "no code-search adapter (embedTrainBench)")
    val q = "the declarations of a project whose names match a pattern"
    val d = Seq("Index.named(pattern: String): Vector[Symbol]", "Store.save(path: Path, data: Bytes): Unit")
    def run(e: Embedder) =
      val qv = e.query(Search.task, q)
      val r = e.vectors(d).map(Embedder.cosine(qv, _))
      e.model.close()
      r
    val base = run(Embedder.load(dir.get))
    val gpu = run(Embedder.load(dir.get, adapter = adapter.get))
    val cpu = run(Embedder.load(dir.get, cpu = true, adapter = adapter.get))
    println(f"base ${base.mkString(" ")}, adapter GPU ${gpu.mkString(" ")}, CPU ${cpu.mkString(" ")}")
    assert(gpu(0) > gpu(1))
    assert(gpu.zip(cpu).forall((a, b) => math.abs(a - b) < 0.01))
    assert(gpu.zip(base).exists((a, b) => math.abs(a - b) > 0.005), "the adapter changed nothing")
