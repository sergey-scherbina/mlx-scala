package mlx.llm

import java.nio.file.{Files, Path, Paths}
import scala.jdk.CollectionConverters.*
import okay.effects.json.Json

/** AN INDEPENDENT CHECK of code search: rozum's own question set on okay's code (rozum scripts/rag-eval/okay-code.json —
 * 20 questions written without the definitions' names, a project the adapter never saw), the corpus every declaration
 * line of the questions' modules' main sources (`File.scala: def …`); the right one's rank by base, the code-search
 * adapter on top, a merged model (`LoraMerge`, a directory given), and TF-IDF.
 *
 * `sbt "llmJVM/Test/runMain mlx.llm.okayEval mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ <questions.json> <okay> [merged]"` */
@main def okayEval(repo: String, questions: String, okay: String, merged: String*): Unit =
  val root = Paths.get(okay)
  val qs = Json.fromText(Files.readString(Paths.get(questions)))("questions").list.map(q =>
    (q("q").string, q("path").string, q("def").string.r))
  val modules = qs.map(_._2.split('/').head).distinct
  val Decl = """^\s*(?:(?:private\[\w+\]|protected|final|override|implicit|inline|transparent|lazy|sealed|abstract|""" +
    """case|opaque)\s+)*(?:def|val|class|trait|object|enum|given|type)\s.*"""
  val decls = modules.flatMap { m =>
    val d = root.resolve(m)
    if !Files.isDirectory(d) then Vector.empty
    else Files.walk(d).iterator.asScala.filter(p => p.toString.endsWith(".scala") && p.toString.contains("/main/") ||
        (m == "src" && p.toString.endsWith(".scala") && !p.toString.contains("/test/")))
      .filterNot(p => p.toString.contains("/target/") || p.toString.contains("/."))
      .flatMap(p => Files.readAllLines(p).asScala.filter(_.matches(Decl)).map(l =>
        (root.relativize(p).toString, s"${p.getFileName}: ${l.trim.take(300)}"))).toVector
  }.distinct
  val targets = qs.map((_, path, re) => decls.indexWhere((p, t) => p.endsWith(path) && re.findFirstIn(t).isDefined))
  println(s"${qs.size} questions, ${decls.size} declarations in ${modules.mkString(", ")}; " +
    s"${targets.count(_ < 0)} questions' definitions not found")
  val ok = qs.indices.filter(targets(_) >= 0)
  val rows = decls.map((p, t) => Array("okay", p, "", t))
  val withQ = ok.foldLeft(rows)((rs, i) => rs.updated(targets(i), Array("okay", rs(targets(i))(1), qs(i)._1,
    rs(targets(i))(3))))
  // several questions on one declaration would collapse into one row: count them by their own targets instead
  def ranks(e: Embedder): Vector[Int] =
    val dv = e.vectors(decls.map(_._2))
    ok.map { i =>
      val qv = e.query(Search.task, qs(i)._1)
      val s = dv.map(Embedder.cosine(qv, _))
      s.count(_ > s(targets(i)))
    }.toVector
  def report(label: String, rs: Vector[Int]) = println(f"$label%-22s @1 @5 @10 MRR  ${Search.score(rs)}")
  val dir = Models.snapshot(repo).get
  val base = Embedder.load(dir)
  report("base", ranks(base))
  base.model.close()
  for a <- Embedder.codeSearch(dir) do
    val e = Embedder.load(dir, adapter = a)
    report("adapter on top", ranks(e))
    e.model.close()
  for m <- merged do
    val e = Embedder.load(Paths.get(m))
    report(s"merged ${Paths.get(m).getFileName}", ranks(e))
    e.model.close()
  println(f"${"TF-IDF"}%-22s @1 @5 @10 MRR  ${Search.lexical(withQ)} (one question a declaration)")
