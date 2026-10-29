package mlx.llm

import okay.effects.json.Json

/** a prompt's prefix reused: the same tokens as computing it all, the next turn from where the last one's messages
 * end — on a model of attention alone (Qwen3) and a hybrid one (Qwen3.5: linear layers' recurrent state) */
class PrefixCacheSuite extends munit.FunSuite:
  override val munitTimeout = scala.concurrent.duration.Duration(10, "min")

  private def msg(role: String, text: String) = Json.obj("role" -> Json.Str(role), "content" -> Json.Str(text))
  private val system = msg("system", "You are a terse assistant. " + "Answer in one short sentence. " * 40)

  for repo <- Seq("mlx-community/Qwen3-4B-4bit", "mlx-community/Qwen3.5-9B-MLX-4bit") do
    test(s"$repo: a prefix kept, reused, the same tokens"):
      val dir = Models.snapshot(repo)
      assume(dir.isDefined, s"$repo is not in the HuggingFace cache")
      val chat = Chat.load(dir.get)
      val model = Model.load(dir.get)
      val cache = PrefixCache()
      val off = Map[String, Any]("enable_thinking" -> false)
      def run(prompt: Array[Int], c: PrefixCache | Null): (Vector[Int], Timing) =
        val s = Session(model, prompt, 12, chat.stops, c, chat.marks(prompt))
        val out = Iterator.continually(s.next()).takeWhile(_.isDefined).flatten.toVector
        val t = s.timing
        s.close()
        (out, t)
      val turn1 = Seq(system, msg("user", "What is the capital of France?"))
      val p1 = chat.prompt(turn1, Nil, off)
      val (a, ta) = run(p1, cache)
      assertEquals(ta.cachedTokens, 0)
      assertEquals(cache.size, chat.marks(p1).size)
      // the same prompt again: from the snapshot before its last message, the same answer
      val (b, tb) = run(p1, cache)
      assertEquals(b, a)
      assertEquals(tb.cachedTokens, chat.marks(p1).last)
      // the next turn: the conversation so far reused up to where the last turn's question began
      val reply = chat.tokenizer.decode(a)
      val p2 = chat.prompt(turn1 ++ Seq(msg("assistant", reply), msg("user", "And of Italy?")), Nil, off)
      val (c, tc) = run(p2, cache)
      assert(tc.cachedTokens >= chat.marks(p1).head, s"${tc.cachedTokens} cached of ${p2.length}")
      println(s"$repo: ${chat.tokenizer.decode(a).trim} / ${chat.tokenizer.decode(c).trim} — " +
        s"cached ${tb.cachedTokens} of ${p1.length}, then ${tc.cachedTokens} of ${p2.length}; ${cache.bytes >> 20} MB")
      cache.clear()
      model.close()
