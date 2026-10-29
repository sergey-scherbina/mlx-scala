package mlx.llm

import java.nio.file.{Files, Path}

/** A MODEL'S CHAT: its tokenizer and its chat template, from its directory — a conversation in, the prompt's tokens
 * out; the tokens that end a turn */
final class Chat(val tokenizer: Tokenizer, val template: ChatTemplate, val stops: Set[Int]):
  /** messages as JSON (`role`, `content`, …), the tools, the template's own variables (`enable_thinking`, …) */
  def prompt(messages: Seq[Json], tools: Seq[Json] = Nil, extra: Map[String, Any] = Map.empty): Array[Int] =
    tokenizer.encode(template(messages, tools, addGenerationPrompt = true, extra))

  /** one user turn */
  def ask(question: String, extra: Map[String, Any] = Map.empty): Array[Int] =
    prompt(Seq(Json.obj("role" -> Json.Str("user"), "content" -> Json.Str(question))), Nil, extra)

object Chat:
  def load(dir: Path): Chat =
    val tokenizer = Tokenizer.fromJson(Json.parse(Files.readString(dir.resolve("tokenizer.json"))))
    val config = Json.parse(Files.readString(dir.resolve("config.json")))
    val eos = config("eos_token_id") match
      case Json.Arr(ids) => ids.map(_.int).toSet
      case Json.Null => Set.empty[Int]
      case id => Set(id.int)
    val stops = eos ++ Set("<|im_end|>", "<|endoftext|>").map(tokenizer.id).filter(_ >= 0)
    Chat(tokenizer, ChatTemplate.load(dir), stops)
