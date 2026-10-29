package mlx.llm

import okay.effects.json.Json

import scala.collection.immutable.VectorMap
import scala.collection.mutable

/** A CHAT TEMPLATE: the Jinja a model's tokenizer_config.json (or chat_template.jinja) carries, rendered as
 * HuggingFace transformers renders it — jinja2 with `trim_blocks` and `lstrip_blocks`, `tojson` as `json.dumps`
 * (", " and ": "), `raise_exception`. The subset chat templates use: output, `if`/`elif`/`else`, `for` (with
 * `loop`, unpacking, `else`), `set` (names, `namespace` attributes), `macro`; expressions with Python's values —
 * truth, `==`, `in`, `str()` — slices, filters, tests, string and dict methods.
 *
 * Values: String, Long, Double, Boolean, null (None), `Undefined`, Vector (lists), VectorMap (dicts) */
final class ChatTemplate private (body: Vector[ChatTemplate.Node]):
  import ChatTemplate.*

  def render(context: Map[String, Any]): String =
    val out = new java.lang.StringBuilder
    val root = Frame(null)
    for (k, v) <- context do root.vars(k) = v
    Interp(root).exec(body, root, out)
    out.toString

  /** the conversation as a prompt: messages (role, content, …) from JSON, tools, the answer begun when asked, and
   * the template's own variables (`enable_thinking`, …) */
  def apply(messages: Seq[Json], tools: Seq[Json] = Nil, addGenerationPrompt: Boolean = true,
      extra: Map[String, Any] = Map.empty): String =
    render(Map("messages" -> messages.map(fromJson).toVector, "add_generation_prompt" -> addGenerationPrompt,
      "bos_token" -> "", "eos_token" -> "") ++ (if tools.isEmpty then Map.empty else
      Map("tools" -> tools.map(fromJson).toVector)) ++ extra)

object ChatTemplate:
  def parse(source: String): ChatTemplate = new ChatTemplate(Parser(Lexer(source).tokens()).template())

  /** a model directory's template: chat_template.jinja, else tokenizer_config.json's `chat_template` */
  def load(dir: java.nio.file.Path): ChatTemplate =
    val jinja = dir.resolve("chat_template.jinja")
    if java.nio.file.Files.exists(jinja) then parse(java.nio.file.Files.readString(jinja))
    else
      val config = Json.fromText(java.nio.file.Files.readString(dir.resolve("tokenizer_config.json")))
      config("chat_template") match
        case Json.Str(t) => parse(t)
        case Json.Arr(ts) => parse(ts.find(_("name").string == "default").getOrElse(ts.head)("template").string)
        case other => sys.error(s"no chat template: $other")

  /** a template raised `raise_exception`, or went wrong */
  final class TemplateError(message: String) extends RuntimeException(message)

  /** Jinja's undefined: false, empty, "" */
  case object Undefined

  def fromJson(j: Json): Any = j match
    case Json.Obj(f) => VectorMap.from(f.map((k, v) => k -> fromJson(v)))
    case Json.Arr(xs) => xs.map(fromJson)
    case Json.Str(s) => s
    case Json.Num(n) => if n.isWhole && n.abs < BigDecimal(9e15) then n.toLong else n.toDouble
    case Json.Bool(b) => b
    case Json.Null => null

  // ---- the template's text into tokens: text, {{ output }}, {% statement %}, with Jinja's whitespace rules

  private enum Tok:
    case Text(s: String)
    case Out(src: String)
    case Stmt(src: String)

  private final class Lexer(src: String):
    def tokens(): Vector[Tok] =
      val out = Vector.newBuilder[Tok]
      var i = 0
      var stripNext = false       // a `-` closed the last tag: the text after it loses its leading whitespace
      var trimNewline = false     // a block closed: trim_blocks drops one newline after it
      while i < src.length do
        val open = nextOpen(i)
        var text = src.substring(i, if open < 0 then src.length else open)
        if stripNext then text = text.dropWhile(_.isWhitespace)
        else if trimNewline then
          if text.startsWith("\r\n") then text = text.drop(2) else if text.startsWith("\n") then text = text.drop(1)
        if open < 0 then
          out += Tok.Text(text)
          i = src.length
        else
          val kind = src.charAt(open + 1)
          val minus = open + 2 < src.length && src.charAt(open + 2) == '-'
          val plus = open + 2 < src.length && src.charAt(open + 2) == '+'
          if minus then text = text.reverse.dropWhile(_.isWhitespace).reverse
          else if kind != '{' && !plus then
            // lstrip_blocks: spaces and tabs from the line's start to a block tag
            val nl = text.lastIndexOf('\n')
            val tail = text.substring(nl + 1)
            if tail.forall(c => c == ' ' || c == '\t') && (nl >= 0 || atLineStart(i)) then
              text = text.substring(0, nl + 1)
          out += Tok.Text(text)
          val close = kind match { case '{' => "}}"; case '%' => "%}"; case _ => "#}" }
          val end = findClose(open + 2, close)
          if end < 0 then throw TemplateError(s"unclosed tag at $open")
          var inner = src.substring(open + 2, end)
          if inner.startsWith("-") || inner.startsWith("+") then inner = inner.drop(1)
          stripNext = inner.endsWith("-")
          if stripNext then inner = inner.dropRight(1)
          trimNewline = kind != '{'
          kind match
            case '{' => out += Tok.Out(inner.trim)
            case '%' => out += Tok.Stmt(inner.trim)
            case _ =>
          i = end + 2
      out.result().filter { case Tok.Text(t) => t.nonEmpty; case _ => true }

    /** whether position `i` (where a text begins) starts a line: the start, or after a newline */
    private def atLineStart(i: Int): Boolean = i == 0 || src.charAt(i - 1) == '\n'

    private def nextOpen(from: Int): Int =
      var i = src.indexOf('{', from)
      while i >= 0 && i + 1 < src.length && "{%#".indexOf(src.charAt(i + 1).toInt) < 0 do i = src.indexOf('{', i + 1)
      if i >= 0 && i + 1 < src.length then i else -1

    /** the closing delimiter, past string literals */
    private def findClose(from: Int, close: String): Int =
      var i = from
      var quote = 0.toChar
      while i < src.length do
        val c = src.charAt(i)
        if quote != 0 then
          if c == '\\' then i += 1 else if c == quote then quote = 0.toChar
        else if c == '\'' || c == '"' then quote = c
        else if src.startsWith(close, i) then return i
        i += 1
      -1

  // ---- statements and expressions

  enum Node:
    case Text(s: String)
    case Output(e: Expr)
    case If(branches: Vector[(Expr, Vector[Node])], otherwise: Vector[Node])
    case For(targets: Vector[String], iter: Expr, cond: Option[Expr], body: Vector[Node], otherwise: Vector[Node])
    case SetVar(name: String, attr: Option[String], value: Expr)
    case Macro(name: String, params: Vector[(String, Option[Expr])], body: Vector[Node])

  enum Expr:
    case Lit(v: Any)
    case Name(n: String)
    case Attr(e: Expr, name: String)
    case Item(e: Expr, key: Expr)
    case Slice(e: Expr, start: Option[Expr], stop: Option[Expr], step: Option[Expr])
    case Call(f: Expr, args: Vector[Expr], kwargs: Vector[(String, Expr)])
    case Filter(e: Expr, name: String, args: Vector[Expr], kwargs: Vector[(String, Expr)])
    case Test(e: Expr, name: String, args: Vector[Expr], negated: Boolean)
    case Bin(op: String, a: Expr, b: Expr)
    case Not(e: Expr)
    case Neg(e: Expr)
    case Cond(cond: Expr, yes: Expr, no: Option[Expr])
    case ListE(items: Vector[Expr])
    case DictE(items: Vector[(Expr, Expr)])

  private final class Parser(toks: Vector[Tok]):
    private var i = 0

    def template(): Vector[Node] =
      val (body, end) = block(Set.empty)
      if end.nonEmpty then throw TemplateError(s"unexpected {% ${end.get} %}")
      body

    /** nodes up to a statement whose first word is in `ends`: the nodes, and that statement */
    private def block(ends: Set[String]): (Vector[Node], Option[String]) =
      val b = Vector.newBuilder[Node]
      while i < toks.length do
        toks(i) match
          case Tok.Text(s) => b += Node.Text(s); i += 1
          case Tok.Out(src) => b += Node.Output(Expr0(src).whole()); i += 1
          case Tok.Stmt(src) =>
            val word = src.takeWhile(c => c.isLetterOrDigit || c == '_')
            if ends(word) then { i += 1; return (b.result(), Some(src)) }
            i += 1
            b += statement(word, src.drop(word.length).trim)
      if ends.nonEmpty then throw TemplateError(s"missing one of ${ends.mkString(", ")}")
      (b.result(), None)

    private def statement(word: String, rest: String): Node = word match
      case "if" =>
        val branches = Vector.newBuilder[(Expr, Vector[Node])]
        var cond = Expr0(rest).whole()
        var otherwise = Vector.empty[Node]
        var more = true
        while more do
          val (body, end) = block(Set("elif", "else", "endif"))
          branches += cond -> body
          val e = end.get
          if e.startsWith("elif") then cond = Expr0(e.drop(4).trim).whole()
          else if e.startsWith("else") then { otherwise = block(Set("endif"))._1; more = false }
          else more = false
        Node.If(branches.result(), otherwise)
      case "for" =>
        val p = Expr0(rest)
        val targets = p.targets()
        p.keyword("in")
        val iter = p.ternaryNoCond()
        val cond = if p.atKeyword("if") then { p.keyword("if"); Some(p.ternary()) } else None
        p.end()
        val (body, end) = block(Set("else", "endfor"))
        val otherwise = if end.get.startsWith("else") then block(Set("endfor"))._1 else Vector.empty
        Node.For(targets, iter, cond, body, otherwise)
      case "set" =>
        val eq = topLevelEquals(rest)
        val target = rest.substring(0, eq).trim
        val value = Expr0(rest.substring(eq + 1)).whole()
        target.split('.') match
          case Array(n) => Node.SetVar(n, None, value)
          case Array(n, a) => Node.SetVar(n.trim, Some(a.trim), value)
          case _ => throw TemplateError(s"set: $target")
      case "macro" =>
        val p = Expr0(rest)
        val name = p.name()
        val params = p.params()
        p.end()
        Node.Macro(name, params, block(Set("endmacro"))._1)
      case other => throw TemplateError(s"a statement not supported: {% $other %}")

    private def topLevelEquals(s: String): Int =
      var depth = 0
      var quote = 0.toChar
      var k = 0
      while k < s.length do
        val c = s.charAt(k)
        if quote != 0 then { if c == '\\' then k += 1 else if c == quote then quote = 0.toChar }
        else if c == '\'' || c == '"' then quote = c
        else if "([{".indexOf(c.toInt) >= 0 then depth += 1
        else if ")]}".indexOf(c.toInt) >= 0 then depth -= 1
        else if c == '=' && depth == 0 && (k + 1 >= s.length || s.charAt(k + 1) != '=') &&
          (k == 0 || "!<>=".indexOf(s.charAt(k - 1).toInt) < 0) then return k
        k += 1
      throw TemplateError(s"set without '=': $s")

  /** an expression's tokens: names, strings, numbers, operators */
  private final class Expr0(src: String):
    private val toks: Vector[String] =
      val b = Vector.newBuilder[String]
      var k = 0
      while k < src.length do
        val c = src.charAt(k)
        if c.isWhitespace then k += 1
        else if c == '\'' || c == '"' then
          val sb = new java.lang.StringBuilder
          var j = k + 1
          while src.charAt(j) != c do
            if src.charAt(j) == '\\' then
              j += 1
              src.charAt(j) match
                case 'n' => sb.append('\n')
                case 't' => sb.append('\t')
                case 'r' => sb.append('\r')
                case other => sb.append(other)
            else sb.append(src.charAt(j))
            j += 1
          b += "\u0000" + sb.toString       // a string literal, marked
          k = j + 1
        else if c.isDigit then
          var j = k
          while j < src.length && (src.charAt(j).isDigit || src.charAt(j) == '.') do j += 1
          b += src.substring(k, j)
          k = j
        else if c.isLetter || c == '_' then
          var j = k
          while j < src.length && (src.charAt(j).isLetterOrDigit || src.charAt(j) == '_') do j += 1
          b += src.substring(k, j)
          k = j
        else
          val two = if k + 1 < src.length then src.substring(k, k + 2) else ""
          if Set("==", "!=", "<=", ">=", "//", "**").contains(two) then { b += two; k += 2 }
          else { b += c.toString; k += 1 }
      b.result()
    private var p = 0
    private def peek: String = if p < toks.length then toks(p) else ""
    private def peekAt(n: Int): String = if p + n < toks.length then toks(p + n) else ""
    private def next(): String = { val t = peek; p += 1; t }
    private def expect(t: String): Unit = if next() != t then throw TemplateError(s"'$t' expected in: $src")
    def atKeyword(k: String): Boolean = peek == k
    def keyword(k: String): Unit = expect(k)
    def end(): Unit = if p != toks.length then throw TemplateError(s"unexpected '${peek}' in: $src")
    def name(): String =
      val t = next()
      if t.isEmpty || !(t.head.isLetter || t.head == '_') then throw TemplateError(s"a name expected in: $src")
      t
    def whole(): Expr = { val e = ternary(); end(); e }
    def targets(): Vector[String] =
      val b = Vector.newBuilder[String]
      b += name()
      while peek == "," do { next(); b += name() }
      b.result()
    def params(): Vector[(String, Option[Expr])] =
      expect("(")
      val b = Vector.newBuilder[(String, Option[Expr])]
      while peek != ")" do
        val n = name()
        val d = if peek == "=" then { next(); Some(ternary()) } else None
        b += n -> d
        if peek == "," then next()
      expect(")")
      b.result()

    def ternary(): Expr =
      val a = or()
      if peek == "if" then
        next()
        val c = or()
        val no = if peek == "else" then { next(); Some(ternary()) } else None
        Expr.Cond(c, a, no)
      else a
    /** a for loop's iterable: its `if` is the loop's filter, not a conditional */
    def ternaryNoCond(): Expr = or()

    private def or(): Expr =
      var a = and()
      while peek == "or" do { next(); a = Expr.Bin("or", a, and()) }
      a
    private def and(): Expr =
      var a = not()
      while peek == "and" do { next(); a = Expr.Bin("and", a, not()) }
      a
    private def not(): Expr =
      if peek == "not" then { next(); Expr.Not(not()) } else compare()
    private def compare(): Expr =
      var a = math1()
      var more = true
      while more do
        peek match
          case op @ ("==" | "!=" | "<" | ">" | "<=" | ">=") => next(); a = Expr.Bin(op, a, math1())
          case "in" => next(); a = Expr.Bin("in", a, math1())
          case "not" if peekAt(1) == "in" => next(); next(); a = Expr.Not(Expr.Bin("in", a, math1()))
          case _ => more = false
      a
    private def math1(): Expr =
      var a = concat()
      while peek == "+" || peek == "-" do { val op = next(); a = Expr.Bin(op, a, concat()) }
      a
    private def concat(): Expr =
      var a = math2()
      while peek == "~" do { next(); a = Expr.Bin("~", a, math2()) }
      a
    private def math2(): Expr =
      var a = pow()
      while Set("*", "/", "//", "%").contains(peek) do { val op = next(); a = Expr.Bin(op, a, pow()) }
      a
    private def pow(): Expr =
      var a = unary()
      while peek == "**" do { next(); a = Expr.Bin("**", a, unary()) }
      a
    private def unary(): Expr =
      if peek == "-" then { next(); Expr.Neg(unary()) }
      else if peek == "+" then { next(); unary() }
      else filters(postfix(primary()))

    private def filters(e0: Expr): Expr =
      var e = e0
      var more = true
      while more do
        if peek == "|" then
          next()
          val n = name()
          val (args, kwargs) = if peek == "(" then callArgs() else (Vector.empty, Vector.empty)
          e = Expr.Filter(e, n, args, kwargs)
        else if peek == "is" then
          next()
          val negated = if peek == "not" then { next(); true } else false
          val n = name()
          val args =
            if peek == "(" then callArgs()._1
            else if peek.nonEmpty && (peek.head == '\u0000' || peek.head.isDigit || peek == "none" || peek == "None")
              && !Set("and", "or", "else", "if").contains(peek) then Vector(primary())
            else Vector.empty
          e = Expr.Test(e, n, args, negated)
        else more = false
      e

    private def postfix(e0: Expr): Expr =
      var e = e0
      var more = true
      while more do
        peek match
          case "." => next(); e = Expr.Attr(e, name())
          case "[" =>
            next()
            def part(): Option[Expr] = if peek == ":" || peek == "]" then None else Some(ternary())
            val start = part()
            if peek == ":" then
              next()
              val stop = part()
              val step = if peek == ":" then { next(); part() } else None
              expect("]")
              e = Expr.Slice(e, start, stop, step)
            else
              expect("]")
              e = Expr.Item(e, start.get)
          case "(" =>
            val (args, kwargs) = callArgs()
            e = Expr.Call(e, args, kwargs)
          case _ => more = false
      e

    private def callArgs(): (Vector[Expr], Vector[(String, Expr)]) =
      expect("(")
      val args = Vector.newBuilder[Expr]
      val kwargs = Vector.newBuilder[(String, Expr)]
      while peek != ")" do
        if peekAt(1) == "=" && peek.nonEmpty && peek.head.isLetter then
          val k = next(); next()
          kwargs += k -> ternary()
        else args += ternary()
        if peek == "," then next()
      expect(")")
      (args.result(), kwargs.result())

    private def primary(): Expr =
      val t = next()
      if t.isEmpty then throw TemplateError(s"an expression expected in: $src")
      if t.head == '\u0000' then
        var s = t.drop(1)
        while peek.nonEmpty && peek.head == '\u0000' do s += next().drop(1)    // "a" "b": one string
        Expr.Lit(s)
      else if t.head.isDigit then Expr.Lit(if t.contains('.') then t.toDouble else t.toLong)
      else t match
        case "true" | "True" => Expr.Lit(true)
        case "false" | "False" => Expr.Lit(false)
        case "none" | "None" => Expr.Lit(null)
        case "(" =>
          val e = ternary()
          if peek == "," then
            val b = Vector.newBuilder[Expr]
            b += e
            while peek == "," do { next(); if peek != ")" then b += ternary() }
            expect(")")
            Expr.ListE(b.result())
          else { expect(")"); e }
        case "[" =>
          val b = Vector.newBuilder[Expr]
          while peek != "]" do { b += ternary(); if peek == "," then next() }
          expect("]")
          Expr.ListE(b.result())
        case "{" =>
          val b = Vector.newBuilder[(Expr, Expr)]
          while peek != "}" do
            val k = ternary(); expect(":"); b += k -> ternary()
            if peek == "," then next()
          expect("}")
          Expr.DictE(b.result())
        case n if n.head.isLetter || n.head == '_' => Expr.Name(n)
        case other => throw TemplateError(s"unexpected '$other' in: $src")

  // ---- rendering

  private final class Frame(val parent: Frame | Null):
    val vars = mutable.HashMap.empty[String, Any]
    def lookup(n: String): Any =
      var f: Frame | Null = this
      while f != null do
        val v = f.nn.vars.getOrElse(n, Missing)
        if v != Missing then return v
        f = f.nn.parent
      Undefined
  private case object Missing

  /** a namespace(): attributes set from anywhere */
  private final class Namespace(val attrs: mutable.LinkedHashMap[String, Any])
  private final class MacroV(val params: Vector[(String, Option[Expr])], val body: Vector[Node], val home: Frame)
  private final class Loop(items: Vector[Any], val index0: Int):
    def attr(n: String): Any = n match
      case "index0" => index0.toLong
      case "index" => index0 + 1L
      case "first" => index0 == 0
      case "last" => index0 == items.length - 1
      case "length" => items.length.toLong
      case "revindex" => (items.length - index0).toLong
      case "revindex0" => (items.length - index0 - 1).toLong
      case "previtem" => if index0 > 0 then items(index0 - 1) else Undefined
      case "nextitem" => if index0 + 1 < items.length then items(index0 + 1) else Undefined
      case _ => Undefined

  private final class Interp(root: Frame):
    def exec(nodes: Vector[Node], f: Frame, out: java.lang.StringBuilder): Unit =
      for n <- nodes do
        n match
          case Node.Text(s) => out.append(s)
          case Node.Output(e) => out.append(str(eval(e, f)))
          case Node.If(branches, otherwise) =>
            branches.find((c, _) => truth(eval(c, f))) match
              case Some((_, body)) => exec(body, f, out)
              case None => exec(otherwise, f, out)
          case Node.For(targets, iter, cond, body, otherwise) =>
            val all = iterate(eval(iter, f))
            val items = cond match
              case None => all
              case Some(c) => all.filter { x => val g = Frame(f); bind(g, targets, x); truth(eval(c, g)) }
            if items.isEmpty then exec(otherwise, f, out)
            else
              val g = Frame(f)
              var k = 0
              while k < items.length do
                bind(g, targets, items(k))
                g.vars("loop") = Loop(items, k)
                exec(body, g, out)
                k += 1
          case Node.SetVar(name, None, value) => f.vars(name) = eval(value, f)
          case Node.SetVar(name, Some(attr), value) =>
            f.lookup(name) match
              case ns: Namespace => ns.attrs(attr) = eval(value, f)
              case other => throw TemplateError(s"cannot set $attr of ${str(other)}")
          case Node.Macro(name, params, body) => f.vars(name) = MacroV(params, body, root)

    private def bind(g: Frame, targets: Vector[String], x: Any): Unit =
      if targets.length == 1 then g.vars(targets.head) = x
      else
        val xs = iterate(x)
        if xs.length != targets.length then throw TemplateError(s"cannot unpack ${str(x)} into ${targets.length}")
        targets.zip(xs).foreach((t, v) => g.vars(t) = v)

    def eval(e: Expr, f: Frame): Any = e match
      case Expr.Lit(v) => v
      case Expr.Name(n) => f.lookup(n)
      case Expr.Attr(o, n) => attr(eval(o, f), n)
      case Expr.Item(o, k) => item(eval(o, f), eval(k, f))
      case Expr.Slice(o, a, b, c) =>
        def opt(x: Option[Expr]) = x.map(eval(_, f)).filter(v => v != null && v != Undefined).map(int)
        slice(eval(o, f), opt(a), opt(b), opt(c))
      case Expr.ListE(xs) => xs.map(eval(_, f))
      case Expr.DictE(kvs) => VectorMap.from(kvs.map((k, v) => str(eval(k, f)) -> eval(v, f)))
      case Expr.Not(x) => !truth(eval(x, f))
      case Expr.Neg(x) => eval(x, f) match
        case n: Long => -n
        case d: Double => -d
        case other => throw TemplateError(s"-${str(other)}")
      case Expr.Cond(c, yes, no) => if truth(eval(c, f)) then eval(yes, f) else no.map(eval(_, f)).getOrElse(Undefined)
      case Expr.Bin("and", a, b) => val x = eval(a, f); if truth(x) then eval(b, f) else x
      case Expr.Bin("or", a, b) => val x = eval(a, f); if truth(x) then x else eval(b, f)
      case Expr.Bin(op, a, b) => binary(op, eval(a, f), eval(b, f))
      case Expr.Test(x, name, args, negated) => test(name, eval(x, f), args.map(eval(_, f))) != negated
      case Expr.Filter(x, name, args, kwargs) =>
        filter(name, eval(x, f), args.map(eval(_, f)), kwargs.map((k, v) => k -> eval(v, f)).toMap)
      case Expr.Call(fn, args, kwargs) =>
        val as = args.map(eval(_, f))
        val kws = kwargs.map((k, v) => k -> eval(v, f))
        fn match
          case Expr.Attr(o, m) => method(eval(o, f), m, as, kws.toMap)
          case Expr.Name("raise_exception") => throw TemplateError(str(as.headOption.getOrElse("")))
          case Expr.Name("namespace") => Namespace(mutable.LinkedHashMap.from(kws))
          case Expr.Name("range") => as.map(int) match
            case Seq(n) => (0L until n).toVector
            case Seq(a, b) => (a until b).toVector
            case Seq(a, b, s) => (a until b by s).toVector
            case _ => throw TemplateError("range()")
          case Expr.Name("strftime_now") => strftime(str(as.head), System.currentTimeMillis())
          case _ => eval(fn, f) match
            case m: MacroV => call(m, as, kws.toMap)
            case other => throw TemplateError(s"not callable: ${str(other)}")

    private def call(m: MacroV, args: Vector[Any], kwargs: Map[String, Any]): String =
      val g = Frame(m.home)
      for ((p, default), k) <- m.params.zipWithIndex do
        g.vars(p) =
          if k < args.length then args(k)
          else kwargs.getOrElse(p, default.map(eval(_, g)).getOrElse(Undefined))
      val out = new java.lang.StringBuilder
      exec(m.body, g, out)
      out.toString

    private def attr(o: Any, n: String): Any = o match
      case m: VectorMap[?, ?] => m.asInstanceOf[VectorMap[String, Any]].getOrElse(n, Undefined)
      case ns: Namespace => ns.attrs.getOrElse(n, Undefined)
      case l: Loop => l.attr(n)
      case _ => Undefined

    private def item(o: Any, k: Any): Any = (o, k) match
      case (m: VectorMap[?, ?], key) => m.asInstanceOf[VectorMap[String, Any]].getOrElse(str(key), Undefined)
      case (xs: Vector[?], i: Long) =>
        val j = if i < 0 then xs.length + i else i
        if j >= 0 && j < xs.length then xs(j.toInt) else Undefined
      case (s: String, i: Long) =>
        val j = if i < 0 then s.length + i else i
        if j >= 0 && j < s.length then s.charAt(j.toInt).toString else Undefined
      case (ns: Namespace, key) => ns.attrs.getOrElse(str(key), Undefined)
      case (l: Loop, key) => l.attr(str(key))
      case _ => Undefined

    private def slice(o: Any, a: Option[Long], b: Option[Long], c: Option[Long]): Any =
      val step = c.getOrElse(1L)
      def idx(n: Int): Vector[Int] =
        def clamp(x: Long, lo: Long, hi: Long) = math.max(lo, math.min(hi, x))
        def norm(x: Long, lo: Long, hi: Long) = clamp(if x < 0 then x + n else x, lo, hi)
        if step > 0 then
          val s = a.map(norm(_, 0, n)).getOrElse(0L); val e = b.map(norm(_, 0, n)).getOrElse(n.toLong)
          (s until e by step).map(_.toInt).toVector
        else
          val s = a.map(norm(_, -1, n - 1)).getOrElse(n - 1L); val e = b.map(norm(_, -1, n - 1)).getOrElse(-1L)
          (s until e by step).map(_.toInt).toVector
      o match
        case xs: Vector[?] => idx(xs.length).map(xs(_))
        case s: String => idx(s.length).map(s.charAt).mkString
        case other => throw TemplateError(s"cannot slice ${str(other)}")

    private def binary(op: String, a: Any, b: Any): Any = op match
      case "==" => equal(a, b)
      case "!=" => !equal(a, b)
      case "<" | ">" | "<=" | ">=" =>
        val c = compareValues(a, b)
        op match { case "<" => c < 0; case ">" => c > 0; case "<=" => c <= 0; case _ => c >= 0 }
      case "in" => b match
        case s: String => s.contains(str(a))
        case xs: Vector[?] => xs.exists(equal(_, a))
        case m: VectorMap[?, ?] => m.asInstanceOf[VectorMap[String, Any]].contains(str(a))
        case ns: Namespace => ns.attrs.contains(str(a))
        case Undefined | null => throw TemplateError(s"'in' on ${str(b)}")
        case other => throw TemplateError(s"'in' on ${str(other)}")
      case "~" => str(a) + str(b)
      case "+" => (a, b) match
        case (x: String, y: String) => x + y
        case (x: Vector[?], y: Vector[?]) => x ++ y
        case _ => arith(op, a, b)
      case _ => arith(op, a, b)

    private def arith(op: String, a: Any, b: Any): Any = (a, b) match
      case (x: Long, y: Long) => op match
        case "+" => x + y
        case "-" => x - y
        case "*" => x * y
        case "/" => x.toDouble / y
        case "//" => Math.floorDiv(x, y)
        case "%" => Math.floorMod(x, y)
        case _ => math.pow(x.toDouble, y.toDouble).toLong
      case (x: (Long | Double), y: (Long | Double)) =>
        val (p, q) = (num(x), num(y))
        op match
          case "+" => p + q
          case "-" => p - q
          case "*" => p * q
          case "/" => p / q
          case "//" => math.floor(p / q)
          case "%" => p - q * math.floor(p / q)
          case _ => math.pow(p, q)
      case (s: String, n: Long) if op == "*" => s * n.toInt
      case _ => throw TemplateError(s"${str(a)} $op ${str(b)}")

    private def test(name: String, v: Any, args: Vector[Any]): Boolean = name match
      case "defined" => v != Undefined
      case "undefined" => v == Undefined
      case "none" => v == null
      case "string" => v.isInstanceOf[String]
      case "number" => v.isInstanceOf[Long] || v.isInstanceOf[Double]
      case "integer" => v.isInstanceOf[Long]
      case "float" => v.isInstanceOf[Double]
      case "boolean" => v.isInstanceOf[Boolean]
      case "true" => v == true
      case "false" => v == false
      case "mapping" => v.isInstanceOf[VectorMap[?, ?]] || v.isInstanceOf[Namespace]
      case "iterable" => v.isInstanceOf[Vector[?]] || v.isInstanceOf[String] || v.isInstanceOf[VectorMap[?, ?]]
      case "sequence" => v.isInstanceOf[Vector[?]] || v.isInstanceOf[String] || v.isInstanceOf[VectorMap[?, ?]]
      case "callable" => v.isInstanceOf[MacroV]
      case "eq" | "equalto" | "sameas" => equal(v, args.head)
      case "in" => binary("in", v, args.head) == true
      case other => throw TemplateError(s"a test not supported: $other")

    private def filter(name: String, v: Any, args: Vector[Any], kwargs: Map[String, Any]): Any = name match
      case "length" | "count" => v match
        case s: String => s.length.toLong
        case xs: Vector[?] => xs.length.toLong
        case m: VectorMap[?, ?] => m.size.toLong
        case other => throw TemplateError(s"length of ${str(other)}")
      case "tojson" =>
        val indent = kwargs.get("indent").orElse(args.headOption).filter(_ != null).map(int)
        toJson(v, indent, 0)
      case "trim" => pyStrip(str(v), args.headOption.map(str), true, true)
      case "items" => v match
        case m: VectorMap[?, ?] => m.toVector.map((k, x) => Vector(k, x))
        case Undefined | null => Vector.empty
        case other => throw TemplateError(s"items of ${str(other)}")
      case "string" => str(v)
      case "safe" => v
      case "list" => iterate(v)
      case "join" => iterate(v).map(str).mkString(args.headOption.map(str).getOrElse(""))
      case "lower" => str(v).toLowerCase
      case "upper" => str(v).toUpperCase
      case "capitalize" => val s = str(v); if s.isEmpty then s else s.head.toUpper.toString + s.tail.toLowerCase
      case "first" => iterate(v).headOption.getOrElse(Undefined)
      case "last" => iterate(v).lastOption.getOrElse(Undefined)
      case "default" | "d" =>
        val strict = args.lift(1).exists(truth)
        if v == Undefined || (strict && !truth(v)) then args.headOption.getOrElse("") else v
      case "replace" => str(v).replace(str(args(0)), str(args(1)))
      case "int" => v match
        case n: Long => n
        case d: Double => d.toLong
        case s: String => s.trim.toLongOption.getOrElse(0L)
        case _ => 0L
      case "abs" => v match { case n: Long => math.abs(n); case d: Double => math.abs(d); case o => o }
      case "unique" => iterate(v).distinct
      case "reverse" => v match { case s: String => s.reverse; case o => iterate(o).reverse }
      case other => throw TemplateError(s"a filter not supported: $other")

    private def method(o: Any, m: String, args: Vector[Any], kwargs: Map[String, Any]): Any = (o, m) match
      case (s: String, "startswith") => args.head match
        case xs: Vector[?] => xs.exists(x => s.startsWith(str(x)))
        case x => s.startsWith(str(x))
      case (s: String, "endswith") => args.head match
        case xs: Vector[?] => xs.exists(x => s.endsWith(str(x)))
        case x => s.endsWith(str(x))
      case (s: String, "split") =>
        val max = args.lift(1).orElse(kwargs.get("maxsplit")).map(int).getOrElse(-1L).toInt
        args.headOption.orElse(kwargs.get("sep")).filter(_ != null) match
          case None => s.trim.split("\\s+", if max < 0 then -1 else max + 1).toVector.filter(_.nonEmpty)
          case Some(sep) =>
            val p = java.util.regex.Pattern.quote(str(sep))
            s.split(p, if max < 0 then -1 else max + 1).toVector
      case (s: String, "strip") => pyStrip(s, args.headOption.filter(_ != null).map(str), true, true)
      case (s: String, "lstrip") => pyStrip(s, args.headOption.filter(_ != null).map(str), true, false)
      case (s: String, "rstrip") => pyStrip(s, args.headOption.filter(_ != null).map(str), false, true)
      case (s: String, "lower") => s.toLowerCase
      case (s: String, "upper") => s.toUpperCase
      case (s: String, "title") =>
        s.split(" ").map(w => if w.isEmpty then w else s"${w.head.toUpper}${w.tail.toLowerCase}").mkString(" ")
      case (s: String, "replace") => s.replace(str(args(0)), str(args(1)))
      case (s: String, "find") => s.indexOf(str(args(0))).toLong
      case (s: String, "count") => s.sliding(str(args(0)).length).count(_ == str(args(0))).toLong
      case (m: VectorMap[?, ?], "get") =>
        m.asInstanceOf[VectorMap[String, Any]].getOrElse(str(args(0)), args.lift(1).getOrElse(null))
      case (m: VectorMap[?, ?], "items") => m.toVector.map((k, x) => Vector(k, x))
      case (m: VectorMap[?, ?], "keys") => m.keys.toVector
      case (m: VectorMap[?, ?], "values") => m.values.toVector
      case _ => throw TemplateError(s"a method not supported: ${str(o)}.$m")

  // ---- Python's values

  private def truth(v: Any): Boolean = v match
    case null | Undefined => false
    case b: Boolean => b
    case n: Long => n != 0
    case d: Double => d != 0
    case s: String => s.nonEmpty
    case xs: Vector[?] => xs.nonEmpty
    case m: VectorMap[?, ?] => m.nonEmpty
    case _ => true

  private def num(v: Any): Double = v match
    case n: Long => n.toDouble
    case d: Double => d
    case b: Boolean => if b then 1 else 0
    case other => throw TemplateError(s"not a number: ${str(other)}")
  private def int(v: Any): Long = v match
    case n: Long => n
    case d: Double => d.toLong
    case b: Boolean => if b then 1 else 0
    case other => throw TemplateError(s"not an integer: ${str(other)}")

  private def equal(a: Any, b: Any): Boolean = (a, b) match
    case (x: (Long | Double | Boolean), y: (Long | Double | Boolean)) => num(x) == num(y)
    case (x: Vector[?], y: Vector[?]) => x.length == y.length && x.zip(y).forall(equal)
    case _ => a == b

  private def compareValues(a: Any, b: Any): Int = (a, b) match
    case (x: String, y: String) => x.compareTo(y)
    case (x: (Long | Double | Boolean), y: (Long | Double | Boolean)) => java.lang.Double.compare(num(x), num(y))
    case _ => throw TemplateError(s"cannot compare ${str(a)} and ${str(b)}")

  private def iterate(v: Any): Vector[Any] = v match
    case xs: Vector[?] => xs.asInstanceOf[Vector[Any]]
    case m: VectorMap[?, ?] => m.keys.toVector
    case s: String => s.map(_.toString).toVector
    case Undefined | null => Vector.empty
    case other => throw TemplateError(s"not iterable: ${str(other)}")

  /** Python's str(): None, True, lists and dicts as repr */
  private def str(v: Any): String = v match
    case null => "None"
    case Undefined => ""
    case s: String => s
    case b: Boolean => if b then "True" else "False"
    case n: Long => n.toString
    case d: Double => pyFloat(d)
    case _ => repr(v)
  private def repr(v: Any): String = v match
    case s: String => "'" + s.replace("\\", "\\\\").replace("'", "\\'").replace("\n", "\\n") + "'"
    case xs: Vector[?] => xs.map(repr).mkString("[", ", ", "]")
    case m: VectorMap[?, ?] => m.map((k, x) => repr(k) + ": " + repr(x)).mkString("{", ", ", "}")
    case other => str(other)
  private def pyFloat(d: Double): String =
    if d.isNaN then "nan" else if d.isInfinite then (if d > 0 then "inf" else "-inf")
    else if d == math.rint(d) && math.abs(d) < 1e16 then d.toLong.toString + ".0"
    else d.toString.replace("E", "e+").replace("e+-", "e-")

  private def pyStrip(s: String, chars: Option[String], left: Boolean, right: Boolean): String =
    val drop: Char => Boolean = chars match
      case Some(cs) => c => cs.indexOf(c.toInt) >= 0
      case None => _.isWhitespace
    var a = 0
    var b = s.length
    if left then while a < b && drop(s.charAt(a)) do a += 1
    if right then while b > a && drop(s.charAt(b - 1)) do b -= 1
    s.substring(a, b)

  /** json.dumps: ", " and ": ", not ASCII-escaped; `indent` as Python's */
  private def toJson(v: Any, indent: Option[Long], depth: Int): String =
    def nl(d: Int) = indent.map(n => "\n" + " " * (n.toInt * d)).getOrElse("")
    val (comma, colon) = if indent.isDefined then (",", ": ") else (", ", ": ")
    v match
      case null | Undefined => "null"
      case b: Boolean => if b then "true" else "false"
      case n: Long => n.toString
      case d: Double => pyFloat(d)
      case s: String =>
        Json.quote(s)
      case xs: Vector[?] =>
        if xs.isEmpty then "[]"
        else xs.map(x => nl(depth + 1) + toJson(x, indent, depth + 1)).mkString("[", comma, nl(depth) + "]")
      case m: VectorMap[?, ?] =>
        if m.isEmpty then "{}"
        else m.map((k, x) => nl(depth + 1) + toJson(k, indent, depth + 1) + colon + toJson(x, indent, depth + 1))
          .mkString("{", comma, nl(depth) + "}")
      case ns: Namespace => toJson(VectorMap.from(ns.attrs), indent, depth)
      case other => toJson(str(other), indent, depth)

  private val months = Vector("January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December")
  private val days = Vector("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")

  /** strftime of a time (UTC: Scala Native has no time zones) — the codes chat templates use */
  private[llm] def strftime(f: String, millis: Long): String =
    val secs = Math.floorDiv(millis, 1000L)
    val day = Math.floorDiv(secs, 86400L)
    val sod = Math.floorMod(secs, 86400L)
    // the civil date of a day since 1970-01-01 (Howard Hinnant's algorithm)
    val z = day + 719468
    val era = Math.floorDiv(z, 146097L)
    val doe = z - era * 146097
    val yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365
    val doy = doe - (365 * yoe + yoe / 4 - yoe / 100)
    val mp = (5 * doy + 2) / 153
    val d = doy - (153 * mp + 2) / 5 + 1
    val m = if mp < 10 then mp + 3 else mp - 9
    val y = yoe + era * 400 + (if m <= 2 then 1 else 0)
    val wd = Math.floorMod(day + 3, 7L).toInt       // 1970-01-01 was a Thursday
    val sb = new java.lang.StringBuilder
    var i = 0
    while i < f.length do
      if f.charAt(i) == '%' && i + 1 < f.length then
        f.charAt(i + 1) match
          case 'Y' => sb.append(y)
          case 'm' => sb.append(f"$m%02d")
          case 'd' => sb.append(f"$d%02d")
          case 'B' => sb.append(months(m.toInt - 1))
          case 'b' => sb.append(months(m.toInt - 1).take(3))
          case 'A' => sb.append(days(wd))
          case 'a' => sb.append(days(wd).take(3))
          case 'H' => sb.append(f"${sod / 3600}%02d")
          case 'M' => sb.append(f"${sod / 60 % 60}%02d")
          case 'S' => sb.append(f"${sod % 60}%02d")
          case '%' => sb.append('%')
          case c => sb.append('%').append(c)
        i += 2
      else
        sb.append(f.charAt(i)); i += 1
    sb.toString
