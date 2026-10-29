package mlx.llm

import okay.effects.json.Json

import java.nio.file.{Files, Path}

/** A MODEL'S CHAT: its tokenizer and its chat template, from its directory — a conversation in, the prompt's tokens
 * out; the tokens that end a turn */
final class Chat(val tokenizer: Tokenizer, val template: ChatTemplate, val stops: Set[Int]):
  /** messages as JSON (`role`, `content`, …), the tools, the template's own variables (`enable_thinking`, …) */
  def prompt(messages: Seq[Json], tools: Seq[Json] = Nil, extra: Map[String, Any] = Map.empty): Array[Int] =
    tokenizer.encode(template(messages, tools, addGenerationPrompt = true, extra))

  /** where to keep a prompt's state for the next one (`PrefixCache`): before the second message (the system prompt
   * and the tools behind it) and before the last (the conversation so far) — a turn's prompt begins the next's up
   * to its last message, whatever the template does with the answer that follows */
  def marks(prompt: Array[Int]): Seq[Int] =
    val start = tokenizer.id("<|im_start|>")
    val at = prompt.indices.filter(i => prompt(i) == start)
    (at.lift(1).toSeq ++ at.lastOption.toSeq).filter(_ > 0).distinct

  /** one user turn */
  def ask(question: String, extra: Map[String, Any] = Map.empty): Array[Int] =
    prompt(Seq(Json.obj("role" -> Json.Str("user"), "content" -> Json.Str(question))), Nil, extra)

object Chat:
  def load(dir: Path): Chat =
    val tokenizer = Tokenizer.fromJson(Json.fromText(Files.readString(dir.resolve("tokenizer.json"))))
    val config = Json.fromText(Files.readString(dir.resolve("config.json")))
    val eos = config("eos_token_id") match
      case Json.Arr(ids) => ids.map(_.int).toSet
      case Json.Null => Set.empty[Int]
      case id => Set(id.int)
    val stops = eos ++ Set("<|im_end|>", "<|endoftext|>").map(tokenizer.id).filter(_ >= 0)
    Chat(tokenizer, ChatTemplate.load(dir), stops)
