package mlx.llm

import okay.effects.json.Json

/** NFC against Python's unicodedata (Unicode 16): decomposed Latin, marks out of canonical order, Hangul jamo,
 * exclusions, singletons, and 300 random strings of such (resources/nfc-cases.json, by scripts/gen-nfc.py's
 * Python) */
class NfcSuite extends munit.FunSuite:
  private val cases =
    val text = scala.io.Source.fromInputStream(getClass.getResourceAsStream("/nfc-cases.json"), "UTF-8").mkString
    Json.fromText(text).list.map(c => c.list(0).string -> c.list(1).string)

  test("what Python's NFC gives"):
    for (in, out) <- cases do assertEquals(Nfc(in), out, in.codePoints().toArray.map(_.toHexString).mkString(" "))

  test("ASCII as it is"):
    val s = "plain text, 123"
    assert(Nfc(s) eq s)
