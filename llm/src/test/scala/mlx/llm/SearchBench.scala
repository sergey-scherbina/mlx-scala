package mlx.llm

import java.nio.file.{Files, Paths}
import scala.jdk.CollectionConverters.*

/** CODE SEARCH, measured: okay-forge's pairs (okay-inlay bench/forge/search-pairs.tsv: a member's doc comment's first
 * line as the query, its declaration the one to find among all of its project's declarations) — the rank of the right
 * one by an embedding model (`Embedder`), and by TF-IDF over words (identifiers split at their case: the lexical
 * baseline). Recall at 1, 5, 10 and the mean reciprocal rank, per project and in all.
 *
 * `sbt "llmJVM/Test/runMain mlx.llm.searchBench mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ <pairs.tsv>"` */
@main def searchBench(repo: String, pairs: String): Unit =
  val e = Embedder.load(Models.snapshot(repo).get)
  val t0 = System.nanoTime()
  val rows = Search.rows(pairs)
  println(f"${"project"}%-14s ${"n"}%4s   embedding: @1 @5 @10 MRR      TF-IDF: @1 @5 @10 MRR")
  val all = Search.byProject(rows).map { (project, rs) =>
    val (se, sl) = (Search.embedded(e, rs), Search.lexical(rs))
    println(f"$project%-14s ${se.n}%4d   $se     $sl")
    (se, sl)
  }
  println(f"${"all"}%-14s ${all.map(_._1).reduce(_ + _).n}%4d   ${all.map(_._1).reduce(_ + _)}     " +
    s"${all.map(_._2).reduce(_ + _)}")
  println(f"(${(System.nanoTime() - t0) / 1e9}%.0f s; ${rows.size} declarations)")
  e.model.close()

object Search:
  val task = "Given a description of what a piece of Scala code does, retrieve the declaration it describes"

  /** a pairs file's rows: project, member, doc (empty: a distractor alone), declaration */
  def rows(pairs: String): Vector[Array[String]] =
    Files.readAllLines(Paths.get(pairs)).asScala.filter(_.nonEmpty).map(_.split("\t", -1)).toVector
  def byProject(rows: Vector[Array[String]]): Vector[(String, Vector[Array[String]])] =
    rows.groupBy(_(0)).toVector.sortBy(_._1)
  /** a project's queries: (doc, the index of its declaration) */
  def queries(rs: Vector[Array[String]]): Vector[(String, Int)] =
    rs.zipWithIndex.filter(_._1(2).nonEmpty).map((r, i) => (r(2), i))

  final case class Score(n: Int, at1: Int, at5: Int, at10: Int, rr: Double):
    def +(o: Score) = Score(n + o.n, at1 + o.at1, at5 + o.at5, at10 + o.at10, rr + o.rr)
    def mrr: Double = rr / n
    override def toString =
      f"${at1 * 100.0 / n}%5.1f ${at5 * 100.0 / n}%5.1f ${at10 * 100.0 / n}%5.1f  ${rr / n}%.3f"
  def score(ranks: Seq[Int]) = Score(ranks.size, ranks.count(_ < 1), ranks.count(_ < 5), ranks.count(_ < 10),
    ranks.map(r => 1.0 / (r + 1)).sum)

  /** the right declaration's rank by the embeddings' cosine */
  def embedded(e: Embedder, rs: Vector[Array[String]]): Score =
    val dv = e.vectors(rs.map(_(3)))
    score(queries(rs).map { (q, target) =>
      val qv = e.query(task, q)
      val s = dv.map(Embedder.cosine(qv, _))
      s.count(_ > s(target))
    })

  def words(s: String): Vector[String] =
    s.replaceAll("([a-z0-9])([A-Z])", "$1 $2").toLowerCase.split("[^a-z0-9]+").filter(_.length > 1).toVector

  /** the right declaration's rank by TF-IDF, cosine */
  def lexical(rs: Vector[Array[String]]): Score =
    val docs = rs.map(r => words(r(3)))
    val df = docs.flatMap(_.distinct).groupBy(identity).view.mapValues(_.size).toMap
    def vec(ws: Vector[String]) =
      val tf = ws.groupBy(identity).view.mapValues(_.size).toMap
      val v = tf.map((w, c) => w -> c * math.log((docs.size + 1.0) / (df.getOrElse(w, 0) + 1.0)))
      val norm = math.sqrt(v.values.map(x => x * x).sum).max(1e-9)
      v.map((w, x) => w -> x / norm)
    val dvs = docs.map(vec)
    score(queries(rs).map { (q, target) =>
      val qv = vec(words(q))
      val s = dvs.map(d => qv.map((w, x) => x * d.getOrElse(w, 0.0)).sum)
      s.count(_ > s(target))
    })
