package mlx.llm

import java.nio.charset.StandardCharsets.UTF_8

/** A JSON TEXT WALKED, not built: objects and arrays visited in order, each value read by the caller as it comes
 * (`str`, `int`), skipped (`skip`) or taken as its text (`raw`) — for a big file whose few parts are wanted as tables,
 * not as a tree of objects (`Tokenizer.fromFile`). Strict enough for files a program wrote; an error says where */
private[llm] final class JsonScan(bytes: Array[Byte]):
  private var at = 0

  private def fail(what: String): Nothing = throw IllegalArgumentException(s"JSON at $at: $what expected")

  private def ws(): Unit =
    while at < bytes.length && (bytes(at) == ' ' || bytes(at) == '\n' || bytes(at) == '\r' || bytes(at) == '\t') do
      at += 1

  /** the next character, past white space */
  def peek: Char =
    ws()
    if at < bytes.length then bytes(at).toChar else fail("a value")

  private def expect(c: Char): Unit =
    if peek != c then fail(s"'$c'")
    at += 1

  /** an object: `field` called with each key, the reader at its value (which `field` must read or skip) */
  def obj(field: String => Unit): Unit =
    expect('{')
    if peek == '}' then at += 1
    else
      var more = true
      while more do
        val key = str()
        expect(':')
        field(key)
        peek match
          case ',' => at += 1
          case '}' => at += 1; more = false
          case _ => fail("',' or '}'")

  /** an array: `item` called at each value (which it must read or skip) */
  def arr(item: () => Unit): Unit =
    expect('[')
    if peek == ']' then at += 1
    else
      var more = true
      while more do
        item()
        peek match
          case ',' => at += 1
          case ']' => at += 1; more = false
          case _ => fail("',' or ']'")

  /** a string, its escapes undone (\uXXXX pairs included: appended as chars, a surrogate pair stays one) */
  def str(): String =
    expect('"')
    // the common case, no escape: one String straight from the bytes
    var end = at
    while bytes(end) != '"' && bytes(end) != '\\' do end += 1
    if bytes(end) == '"' then
      val s = String(bytes, at, end - at, UTF_8)
      at = end + 1
      return s
    val sb = java.lang.StringBuilder()
    var start = at
    while bytes(at) != '"' do
      if bytes(at) == '\\' then
        if at > start then sb.append(String(bytes, start, at - start, UTF_8))
        val e = bytes(at + 1).toChar
        at += 2
        e match
          case 'n' => sb.append('\n')
          case 't' => sb.append('\t')
          case 'r' => sb.append('\r')
          case 'b' => sb.append('\b')
          case 'f' => sb.append('\f')
          case 'u' =>
            sb.append(Integer.parseInt(String(bytes, at, 4, UTF_8), 16).toChar)
            at += 4
          case c => sb.append(c) // \" \\ \/
        start = at
      else at += 1
    if at > start then sb.append(String(bytes, start, at - start, UTF_8))
    at += 1
    sb.toString

  /** an integer */
  def int(): Int =
    ws()
    val start = at
    if at < bytes.length && bytes(at) == '-' then at += 1
    while at < bytes.length && bytes(at) >= '0' && bytes(at) <= '9' do at += 1
    if at == start then fail("an integer")
    String(bytes, start, at - start, UTF_8).toInt

  /** any value, passed over */
  def skip(): Unit = peek match
    case '"' =>
      // passed over, not built: past the closing quote, an escaped one included
      at += 1
      while bytes(at) != '"' do at += (if bytes(at) == '\\' then 2 else 1)
      at += 1
    case '{' => obj(_ => skip())
    case '[' => arr(() => skip())
    case _ =>
      while at < bytes.length && !",}] \n\r\t".contains(bytes(at).toChar) do at += 1

  /** any value, as its text */
  def raw(): String =
    ws()
    val start = at
    skip()
    String(bytes, start, at - start, UTF_8)
