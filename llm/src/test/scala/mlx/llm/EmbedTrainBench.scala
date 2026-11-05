package mlx.llm

import java.nio.file.{Files, Paths}

/** CODE SEARCH, TRAINED (`EmbedTrain`): okay-forge's pairs split by PROJECT — trained on some, measured on the others
 * (never seen) — recall and MRR before and after each epoch, beside TF-IDF. The adapter saved to
 * ~/.cache/mlx-scala/embed/<model>/code-search.safetensors.
 *
 * `sbt "llmJVM/Test/runMain mlx.llm.embedTrainBench mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ <pairs.tsv> 3"`
 * EMBED_HELD: the held-out projects (comma-separated) */
@main def embedTrainBench(repo: String, pairs: String, epochs: Int): Unit =
  val dir = Models.snapshot(repo).get
  val e = Embedder.load(dir)
  val held = sys.env.getOrElse("EMBED_HELD", "symdex,okay-scala,okay-store,okay-datalog").split(',').toSet
  val projects = Search.byProject(Search.rows(pairs))
  val (test, train) = projects.partition((p, _) => held(p))
  val batches0 = train.flatMap { (_, rs) =>
    Search.queries(rs).map((q, i) => (e.queryText(Search.task, q), rs(i)(3)))
  }
  println(s"train: ${train.map((p, rs) => s"$p ${Search.queries(rs).size}").mkString(", ")}; held out: " +
    test.map((p, rs) => s"$p ${Search.queries(rs).size}").mkString(", "))
  def measure(label: String): Search.Score =
    val s = test.map((_, rs) => Search.embedded(e, rs)).reduce(_ + _)
    println(f"$label%-10s held-out ${s.n}%d queries: @1 @5 @10 MRR  $s")
    s
  println(f"${"TF-IDF"}%-10s held-out: ${test.map((_, rs) => Search.lexical(rs)).reduce(_ + _)}")
  measure("base")
  val t = EmbedTrain(e)
  val r = scala.util.Random(5)
  val t0 = System.nanoTime()
  for epoch <- 1 to epochs do
    // batches of 8 within a project: hard negatives, declarations alike
    val batches = r.shuffle(train.flatMap { (_, rs) =>
      r.shuffle(Search.queries(rs).map((q, i) => (e.queryText(Search.task, q), rs(i)(3)))).grouped(8)
        .filter(_.size >= 4).toVector
    })
    val losses = batches.map(t.step)
    t.lora.attachTrained()
    measure(f"epoch $epoch")
    t.lora.detach()
    println(f"  loss ${losses.sum / losses.size}%.3f, ${batches.size} batches, " +
      f"${(System.nanoTime() - t0) / 1e9}%.0f s")
  val out = Paths.get(sys.props("user.home"), ".cache", "mlx-scala", "embed",
    dir.getParent.getParent.getFileName.toString,
    "code-search.safetensors")
  Files.createDirectories(out.getParent)
  t.lora.save(out.toString)
  println(s"saved $out (${batches0.size} pairs trained on)")
  t.close()
  e.model.close()
