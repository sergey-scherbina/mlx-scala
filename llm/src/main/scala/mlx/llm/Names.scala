package mlx.llm

/**
 * CODE WITHOUT INVENTED APIs (okay-inlay specs/symbols.md): after `Owner.` — an object, class or package of a
 * project, `members` from its index — the identifier written is one of Owner's members. From the text so far:
 * it ends with an owner's name → a token beginning with `.` must begin a member; it ends with `Owner.partial` → a
 * token must keep `partial` a member's prefix, or complete a member and end the identifier; otherwise free. A dot not
 * followed by an identifier (a sentence's end, `Tools. `) is not a selection: free. `texts`: each token's text.
 */
final class Names(members: Map[String, Set[String]], texts: Int => String, vocabSize: Int,
    encode: String => Array[Int] = _ => Array.empty) extends Port:
  private val sorted: Map[String, Array[String]] = members.view.mapValues(_.toArray.sorted).toMap
  private val text = new StringBuilder
  /** how many times a constrained step was asked, and the tokens it allowed on average */
  var constrained = 0

  private def isId(c: Char) = c.isLetterOrDigit || c == '_' || c == '$'

  /** some member of `owner` starts with `p` (binary search in the sorted names) */
  private def prefixOf(owner: String, p: String): Boolean =
    val ms = sorted(owner)
    var lo = 0
    var hi = ms.length
    while lo < hi do
      val mid = (lo + hi) >>> 1
      if ms(mid) < p then lo = mid + 1 else hi = mid
    lo < ms.length && ms(lo).startsWith(p)

  /** whether `owner.` followed by `cand` (the identifier so far and what comes after) is allowed */
  private def fits(owner: String, cand: String): Boolean =
    val id = cand.takeWhile(isId)
    if id.length == cand.length then prefixOf(owner, id) // the identifier goes on
    else id.isEmpty || members(owner).contains(id) // ended: nothing selected, or a whole member

  private val After = """(?:^|[^\w$.])([A-Za-z_$][\w$]*)$""".r.unanchored
  private val Inside = """(?:^|[^\w$.])([A-Za-z_$][\w$]*)\.([\w$]*)$""".r.unanchored

  /** the state: Some(owner, Some(partial)) after `Owner.partial`, Some(owner, None) after `Owner`, None: free */
  def state: Option[(String, Option[String])] =
    val tail = text.takeRight(200).toString
    tail match
      case Inside(o, p) if members.contains(o) => Some((o, Some(p)))
      case After(o) if members.contains(o) => Some((o, None))
      case _ => None

  def allowed: Option[Array[Int]] = state.map { (owner, partial) =>
    constrained += 1
    val ok = Array.newBuilder[Int]
    var t = 0
    while t < vocabSize do
      val s = texts(t)
      val fine = partial match
        case None => if s.startsWith(".") then fits(owner, s.drop(1)) else !dropsTheDot(s)
        case Some(p) => fits(owner, p + s)
      if fine then ok += t
      t += 1
    ok.result()
  }

  // after an owner, a word with a space and no dot (`Equivalence generators`): a selection with its dot dropped —
  // refused, but for the keywords that may follow a name
  private val keywords = Set("then", "else", "do", "match", "extends", "with", "derives", "yield", "if", "case",
    "while", "for", "new", "end")
  private def dropsTheDot(s: String): Boolean =
    val t = s.dropWhile(c => c == ' ' || c == '\t')
    t.length < s.length && t.nonEmpty && t.head.isLower && !keywords(t.takeWhile(isId))

  /** the members whole, once a selection has begun and is not yet one (`Owner.` or `Owner.imp`): the model
   * chooses among them by its own scores, not prefix by prefix */
  override def choices: Option[Seq[Array[Int]]] = state match
    case Some((owner, Some(p))) if !members(owner).contains(p) =>
      val ms = members(owner).filter(m => m.startsWith(p) && m.length > p.length).toSeq.sorted
      if ms.isEmpty then None else Some(ms.map(m => encode(m.drop(p.length))))
    case _ => None

  def took(token: Int): Unit = text ++= texts(token)

object Names:
  /** for a tokenizer's model: each token's text once (bytes as Latin-1: identifiers are ASCII) */
  def apply(tok: Tokenizer, members: Map[String, Set[String]]): Names =
    val texts = Array.tabulate(tok.size)(id => String(tok.bytes(Seq(id)), "ISO-8859-1"))
    new Names(members, texts, texts.length, tok.encode)

  /**
   * owners and their members from SemanticDB symbols and names (symdex's `named(S, N)`): `pkg/Obj.m().` → Obj: m;
   * `pkg/Cls#m().` → Cls: m; `pkg/Obj.` → pkg's last segment: Obj
   */
  def fromSymbols(named: Seq[(String, String)]): Map[String, Set[String]] =
    // parameters (`m().(p)`), type parameters (`[T]`) and locals are not members
    named.filterNot((s, _) => s.endsWith(")") || s.contains("[") || s.startsWith("local")).flatMap { (sym, name) =>
      val body = sym.stripSuffix(".").stripSuffix("#").replaceAll("""\(\+?\d*\)$""", "")
      val cut = math.max(body.lastIndexOf('.'), math.max(body.lastIndexOf('#'), body.lastIndexOf('/')))
      if cut <= 0 || name.isEmpty || !name.head.isLetter && name.head != '_' then None
      else
        val owner = body.take(cut).split("[./#]").lastOption.getOrElse("")
        if owner.isEmpty || owner == "package" then None else Some(owner -> name)
    }.groupMap(_._1)(_._2).view.mapValues(_.toSet).toMap
