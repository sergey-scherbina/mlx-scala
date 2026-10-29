package mlx.llm

import okay.effects.json.Json

/** Qwen3's and Qwen3.6's chat templates against jinja2 as transformers sets it up (trim_blocks, lstrip_blocks, its
 * tojson): system and user turns, thinking in an assistant's content and apart, tools and tool calls, tool
 * responses, content as parts, Unicode — resources/templates/cases.json, by jinja2 3.1 */
class ChatTemplateSuite extends munit.FunSuite:
  private def resource(name: String): String =
    scala.io.Source.fromInputStream(getClass.getResourceAsStream(s"/templates/$name"), "UTF-8").mkString
  private val templates = Map("qwen3" -> ChatTemplate.parse(resource("qwen3.jinja")),
    "qwen36" -> ChatTemplate.parse(resource("qwen36.jinja")))
  private val cases = Json.fromText(resource("cases.json")).list

  for (c, k) <- cases.zipWithIndex do
    test(s"${c("template").string} #$k"):
      val context = c("context").record.map((n, v) => n -> ChatTemplate.fromJson(v)).toMap
      val expected = c("expected").string
      if expected.startsWith("ERROR") then
        intercept[RuntimeException](templates(c("template").string).render(context))
      else assertEquals(templates(c("template").string).render(context), expected)

  test("strftime_now's date: the civil calendar"):
    assertEquals(ChatTemplate.strftime("%A %d %B %Y %H:%M:%S", 1793268000000L), "Thursday 29 October 2026 10:00:00")
    assertEquals(ChatTemplate.strftime("%Y-%m-%d %a %b", 951782400000L), "2000-02-29 Tue Feb")
