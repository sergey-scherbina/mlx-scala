package okay.mlx.llm

import scala.collection.mutable

/** JSON, read: enough for a model's config.json and tokenizer.json (11 MB: one pass over a char array) */
enum Json:
  case Obj(entries: Map[String, Json])
  case Arr(values: Vector[Json])
  case Str(value: String)
  case Num(value: Double)
  case Bool(value: Boolean)
  case Null

  def apply(key: String): Json = this match
    case Obj(f) => f.getOrElse(key, Null)
    case _ => Null
  def get(key: String): Option[Json] = this match
    case Obj(f) => f.get(key).filter(_ != Null)
    case _ => None
  def str: String = this match { case Str(s) => s; case other => sys.error(s"not a string: $other") }
  def int: Int = this match { case Num(n) => n.toInt; case other => sys.error(s"not a number: $other") }
  def double: Double = this match { case Num(n) => n; case other => sys.error(s"not a number: $other") }
  def bool: Boolean = this match { case Bool(b) => b; case other => sys.error(s"not a boolean: $other") }
  def items: Vector[Json] = this match { case Arr(xs) => xs; case other => sys.error(s"not an array: $other") }
  def fields: Map[String, Json] = this match { case Obj(f) => f; case other => sys.error(s"not an object: $other") }

object Json:
  def parse(text: String): Json = Reader(text.toCharArray).document()

  private final class Reader(cs: Array[Char]):
    private var i = 0
    private val sb = new java.lang.StringBuilder

    def document(): Json =
      val v = value()
      space()
      if i != cs.length then fail("text after the value")
      v

    private def fail(what: String): Nothing = throw IllegalArgumentException(s"JSON at $i: $what")
    private def space(): Unit =
      while i < cs.length && (cs(i) == ' ' || cs(i) == '\n' || cs(i) == '\r' || cs(i) == '\t') do i += 1
    private def expect(c: Char): Unit =
      space()
      if i >= cs.length || cs(i) != c then fail(s"'$c' expected")
      i += 1

    private def value(): Json =
      space()
      if i >= cs.length then fail("a value expected")
      cs(i) match
        case '{' => obj()
        case '[' => arr()
        case '"' => Str(string())
        case 't' => word("true", Bool(true))
        case 'f' => word("false", Bool(false))
        case 'n' => word("null", Null)
        case _ => num()

    private def word(w: String, v: Json): Json =
      if !cs.slice(i, i + w.length).sameElements(w) then fail(s"$w expected")
      i += w.length
      v

    private def obj(): Json =
      i += 1
      val m = mutable.LinkedHashMap.empty[String, Json]
      space()
      if cs(i) == '}' then { i += 1; return Obj(Map.empty) }
      var more = true
      while more do
        space()
        val k = string()
        expect(':')
        m(k) = value()
        space()
        if cs(i) == ',' then i += 1 else { expect('}'); more = false }
      Obj(m.toMap)

    private def arr(): Json =
      i += 1
      val b = Vector.newBuilder[Json]
      space()
      if cs(i) == ']' then { i += 1; return Arr(Vector.empty) }
      var more = true
      while more do
        b += value()
        space()
        if cs(i) == ',' then i += 1 else { expect(']'); more = false }
      Arr(b.result())

    private def string(): String =
      if cs(i) != '"' then fail("a string expected")
      i += 1
      sb.setLength(0)
      while cs(i) != '"' do
        val c = cs(i)
        if c == '\\' then
          i += 1
          cs(i) match
            case 'n' => sb.append('\n')
            case 't' => sb.append('\t')
            case 'r' => sb.append('\r')
            case 'b' => sb.append('\b')
            case 'f' => sb.append('\f')
            case 'u' =>
              sb.append(Integer.parseInt(String(cs, i + 1, 4), 16).toChar)
              i += 4
            case other => sb.append(other)
        else sb.append(c)
        i += 1
      i += 1
      sb.toString

    private def num(): Json =
      val start = i
      while i < cs.length && "+-0123456789.eE".indexOf(cs(i).toInt) >= 0 do i += 1
      if start == i then fail("a value expected")
      Num(String(cs, start, i - start).toDouble)
