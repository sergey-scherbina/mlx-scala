package mlx.llm

import java.nio.file.Paths

/** the code-search adapter merged into the weights (`LoraMerge`), at 4 and 8 bits, against it on top: the held-out
 * projects' recall and MRR — whether the merge keeps what the adapter learned.
 *
 * `sbt "llmJVM/Test/runMain mlx.llm.embedMergeBench mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ <pairs.tsv>"` */
@main def embedMergeBench(repo: String, pairs: String): Unit =
  val dir = Models.snapshot(repo).get
  val adapter = Embedder.codeSearch(dir).get
  val held = sys.env.getOrElse("EMBED_HELD", "symdex,okay-scala,okay-store,okay-datalog").split(',').toSet
  val test = Search.byProject(Search.rows(pairs)).filter((p, _) => held(p))
  def measure(label: String, e: Embedder) =
    val s = test.map((_, rs) => Search.embedded(e, rs)).reduce(_ + _)
    println(f"$label%-24s @1 @5 @10 MRR  $s")
    e.model.close()
  measure("base", Embedder.load(dir))
  measure("adapter on top", Embedder.load(dir, adapter = adapter))
  for bits <- Seq(4, 8) do
    val out = Paths.get(sys.props("user.home"), ".cache", "mlx-scala", "embed", s"code-search-merged-$bits")
    LoraMerge.qwen3(dir, adapter, out, bits)
    measure(s"merged, $bits bits", Embedder.load(out))
