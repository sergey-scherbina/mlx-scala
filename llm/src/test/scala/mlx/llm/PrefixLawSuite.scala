package mlx.llm

import okay.effects.json.Json

/** THE PREFIX LAW: an agent's next request begins with its last one — up to where the model's turn began (the last
 * `<|im_start|>`, the prefix cache's mark), token for token. Appending a turn must not change what came before it:
 * else every turn computes its prompt again (Claude Code's turns: 0.2 s cached, 16 s not) */
class PrefixLawSuite extends munit.FunSuite:
  private def s(t: String) = Json.Str(t)
  private def msg(role: String, content: String) = Json.obj("role" -> s(role), "content" -> s(content))
  private def call(name: String, args: Json) = Json.obj("role" -> s("assistant"), "content" -> s(""),
    "tool_calls" -> Json.arr(Json.obj("type" -> s("function"), "function" -> Json.obj("name" -> s(name),
      "arguments" -> args))))
  private val tools = Vector(Json.obj("type" -> s("function"), "function" -> Json.obj("name" -> s("read"),
    "description" -> s("Read a file"), "parameters" -> Json.obj("type" -> s("object"),
      "properties" -> Json.obj("path" -> Json.obj("type" -> s("string"))), "required" -> Json.arr(s("path"))))))

  /** an agent's conversation; `thought`: its turns carry reasoning */
  private def conversation(thought: Boolean): Vector[Json] =
    def withThought(m: Json, t: String) = if !thought then m else m match
      case o: Json.Obj => Json.Obj(o.fields :+ ("reasoning_content" -> s(t)))
      case other => other
    Vector(msg("system", "You are a coding agent."), msg("user", "What is in README.md?"),
      withThought(call("read", Json.obj("path" -> s("README.md"))), "I should read it."),
      Json.obj("role" -> s("tool"), "content" -> s("# demo\nA demo project.")),
      withThought(msg("assistant", "It describes a demo project."), "Summarise it."),
      msg("user", "And LICENSE?"),
      withThought(call("read", Json.obj("path" -> s("LICENSE"))), "Read that too."),
      Json.obj("role" -> s("tool"), "content" -> s("MIT")))

  /** the requests: where the last message is a person's or a tool's */
  private def requests(c: Vector[Json]) =
    (1 to c.length).filter(k => Set("user", "tool")(c(k - 1)("role").string)).map(c.take)

  /** where each request breaks the law: (request, the first token it differs at, the mark it had to keep) */
  private def breaks(chat: Chat, c: Vector[Json], thinking: Boolean, extra: Map[String, Any] = Map.empty)
      : Vector[String] =
    val start = chat.tokenizer.id("<|im_start|>")
    val prompts = requests(c).map(m => chat.prompt(m, tools, Map("enable_thinking" -> thinking) ++ extra))
    prompts.sliding(2).toVector.zipWithIndex.flatMap { case (Seq(a, b), i) =>
      val mark = a.lastIndexWhere(_ == start)
      val same = a.indices.takeWhile(j => j < b.length && a(j) == b(j)).length
      if same > mark then None
      else Some(s"request ${i + 2}: differs at token $same of ${a.length}, the mark at $mark — " +
        s"'${chat.tokenizer.decode(a.slice(same, same + 30).toSeq)}' became " +
        s"'${chat.tokenizer.decode(b.slice(same, same + 30).toSeq)}'")
    }

  /** where the law holds, asked to keep the think blocks: (repo, without thinking, with it) — Qwen3's template drops
   * past reasoning (no way to keep it), Qwen3.5's past think blocks (none either); Qwen3.6's keeps them when asked */
  private val holds = Seq(("mlx-community/Qwen3-4B-4bit", true, false),
    ("mlx-community/Qwen3.5-9B-MLX-4bit", false, false), ("mlx-community/Qwen3.6-35B-A3B-4bit", true, true))

  for (repo, plain, thinking) <- holds do
    test(s"$repo: each request begins with the last, to its mark — as its template allows"):
      val dir = Models.snapshot(repo)
      assume(dir.isDefined, s"$repo is not in the HuggingFace cache")
      val chat = Chat.load(dir.get)
      val keep = Map[String, Any]("preserve_thinking" -> true)
      for (thought, t) <- Seq((false, false), (true, true)) do
        val b = breaks(chat, conversation(thought), t, keep)
        println(s"$repo, thinking $t: " + (if b.isEmpty then "the law holds" else b.mkString("\n  ", "\n  ", "")))
      assertEquals(breaks(chat, conversation(false), false, keep).isEmpty, plain)
      assertEquals(breaks(chat, conversation(true), true, keep).isEmpty, thinking)
