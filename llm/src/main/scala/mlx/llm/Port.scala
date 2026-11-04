package mlx.llm

/** THE DECODE PORT: a per-step hook in a generation — before each token is picked, the tokens allowed (none: any;
 * one: forced), the rest's logits −∞; then told the token taken. While it constrains, a step is built after the last
 * token is read (the mask depends on it). okay-inlay specs/quotes.md */
trait Port:
  /** the tokens the next pick may take, or none: any */
  def allowed: Option[Array[Int]]
  /** the token taken */
  def took(token: Int): Unit

/**
 * EXACT QUOTES: between the markers `open` and `close` (token sequences), every token continues a span of `source`
 * matching what was copied so far; the close marker allowed at any point (its first token; the rest then forced).
 * The state: off, or the source positions where the copy could continue — open: all of them; a token taken: those
 * followed by it, each one on. Several spans matching narrow as the copy goes on; the close marker always allowed,
  * so the model is never left with nothing. A quote may end inside a source token (`)` where the source has `)\n`):
 * a token that is a strict prefix of one the copy may take (`prefixes`) allowed too — after it, only the close.
 */
final class Quote(source: Array[Int], open: Array[Int], close: Array[Int],
    prefixes: Int => Array[Int] = _ => Array.empty) extends Port:
  require(open.nonEmpty && close.nonEmpty, "markers of at least one token")
  private val recent = scala.collection.mutable.ArrayBuffer.empty[Int]
  // the positions of `source` whose token the copy may take next (inside a quote), else null
  private var matches: Array[Int] | Null = null
  // the close marker's tokens forced so far (0: not closing)
  private var closing = 0
  /** the quotes made: each the tokens copied */
  val quotes = scala.collection.mutable.ArrayBuffer.empty[Vector[Int]]
  private var copied = Vector.empty[Int]

  def inside: Boolean = matches != null || closing > 0

  def allowed: Option[Array[Int]] =
    if closing > 0 then Some(Array(close(closing)))
    else if matches == null then None
    else
      val m = matches.nn
      val next = scala.collection.mutable.LinkedHashSet.empty[Int]
      var i = 0
      while i < m.length do
        if m(i) < source.length then
          next += source(m(i))
          next ++= prefixes(source(m(i)))
        i += 1
      next += close(0)
      Some(next.toArray)

  def took(token: Int): Unit =
    if closing > 0 then
      closing += 1
      if closing == close.length then closing = 0
    else if matches != null then
      if token == close(0) then
        matches = null
        quotes += copied
        copied = Vector.empty
        if close.length > 1 then closing = 1
      else
        matches = matches.nn.filter(p => p < source.length && source(p) == token).map(_ + 1)
        copied :+= token
    else
      recent += token
      if recent.length > open.length then recent.remove(0)
      if recent.length == open.length && recent.sameElements(open) then
        matches = Array.range(0, source.length)
        recent.clear()

object Quote:
  /** for each token, the tokens whose bytes are a strict, non-empty prefix of its bytes — so a quote can end where a
   * source token goes on past it */
  def prefixes(tok: Tokenizer): Int => Array[Int] =
    val byText = java.util.HashMap[String, Integer]()
    for id <- 0 until tok.size do byText.putIfAbsent(String(tok.bytes(Seq(id)), "ISO-8859-1"), id)
    val memo = scala.collection.mutable.HashMap.empty[Int, Array[Int]]
    id => memo.getOrElseUpdate(id, {
      val s = String(tok.bytes(Seq(id)), "ISO-8859-1")
      (1 until s.length).flatMap(n => Option(byText.get(s.take(n))).map(_.toInt)).toArray
    })

  /** the quote port for a tokenizer's model: markers by their text, prefixes from its vocabulary */
  def apply(tok: Tokenizer, source: Array[Int], open: String = "⟦", close: String = "⟧"): Quote =
    new Quote(source, tok.encode(open), tok.encode(close), prefixes(tok))
