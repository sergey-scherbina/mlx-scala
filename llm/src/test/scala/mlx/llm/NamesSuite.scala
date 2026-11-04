package mlx.llm

/** the Names port's laws on a vocabulary of texts */
class NamesSuite extends munit.FunSuite:
  private val vocab = Vector("Tools", ".", ".equ", "equ", "ivalent", "ivalence", "(", " ", ".\n", "Tool", "s", ".x",
    ".specs", "specs", "Other", ".query(", "query")
  private val members = Map("Tools" -> Set("equivalent", "specs", "query"))
  private def port() = new Names(members, vocab, vocab.size)
  private def take(p: Names, ts: String*) = ts.foreach(t => p.took(vocab.indexOf(t)))
  private def allowed(p: Names) = p.allowed.map(_.map(vocab).toSet)

  test("free until an owner; after it, a dot must begin a member, other tokens free"):
    val p = port()
    assertEquals(allowed(p), None)
    take(p, "Tools")
    val a = allowed(p).get
    assert(a(".equ") && a(".specs") && a(".") && a(".query(") && a("(") && a(" ") && a(".\n"), a)
    assert(!a(".x"), a)

  test("inside a member: prefixes go on, a whole member may end, an invented one may not"):
    val p = port()
    take(p, "Tools", ".equ")
    val a = allowed(p).get
    assert(a("ivalent"), a)
    assert(!a("ivalence") && !a("(") && !a(" "), a)
    take(p, "ivalent")
    // a whole member: anything that ends the identifier (a call, a chained selection, free in v0), nothing that
    // would make it another name
    val c = allowed(p).get
    assert(c("(") && c(" ") && c(".x") && c(".query("), c)
    assert(!c("s") && !c("ivalent"), c)
    val q = port()
    take(q, "Tools", ".")
    val b = allowed(q).get
    assert(b("equ") && b("specs") && b("query") && b(" ") && b(".\n"), b)
    assert(b("s"), b) // the start of "specs"
    assert(!b("ivalent") && !b("Tools") && !b("Other"), b)

  test("an owner's name inside a longer identifier, or after a dot, is not an owner"):
    val p = port()
    take(p, "Other", "Tools")
    assertEquals(allowed(p), None)
    val q = port()
    take(q, "Tool", "s")
    assert(allowed(q).isDefined) // "Tools" written in two tokens is the owner
    val r = port()
    take(r, "query", ".", "Tools")
    assertEquals(allowed(r), None)

  test("owners and members from SemanticDB symbols"):
    val m = Names.fromSymbols(Seq("symdex/Equivalence." -> "Equivalence", "symdex/Equivalence.target()." -> "target",
      "symdex/Equivalence.Target#" -> "Target", "symdex/Equivalence.Target#arity." -> "arity",
      "symdex/Tools.equivalent()." -> "equivalent", "symdex/Exec.Run#" -> "Run"))
    assertEquals(m("Equivalence"), Set("target", "Target"))
    assertEquals(m("Target"), Set("arity"))
    assertEquals(m("symdex"), Set("Equivalence"))
    assertEquals(m("Tools"), Set("equivalent"))
