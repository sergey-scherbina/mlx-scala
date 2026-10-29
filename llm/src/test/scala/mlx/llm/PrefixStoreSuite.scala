package mlx.llm

import okay.effects.json.Json

/** prefixes on disk: written on the first run, found by a cache that has none in memory (as after a restart), the
 * same answer — on Qwen3 (KV) and Qwen3.5 (KV and linear layers' state) */
class PrefixStoreSuite extends munit.FunSuite:
  override val munitTimeout = scala.concurrent.duration.Duration(10, "min")
  private def msg(role: String, text: String) = Json.obj("role" -> Json.Str(role), "content" -> Json.Str(text))
  private val system = msg("system", "You are a terse assistant who answers in one short sentence. " * 120)

  for repo <- Seq("mlx-community/Qwen3-4B-4bit", "mlx-community/Qwen3.5-9B-MLX-4bit") do
    test(s"$repo: a prefix from disk, the same answer"):
      val dir = Models.snapshot(repo)
      assume(dir.isDefined, s"$repo is not in the HuggingFace cache")
      val chat = Chat.load(dir.get)
      val model = Model.load(dir.get)
      val root = java.nio.file.Files.createTempDirectory("prefixes")
      def run(cache: PrefixCache, prompt: Array[Int]) =
        val s = Session(model, prompt, 12, chat.stops, cache, chat.marks(prompt))
        val out = Iterator.continually(s.next()).takeWhile(_.isDefined).flatten.toVector
        val t = s.timing
        s.close()
        (out, t)
      val prompt = chat.prompt(Seq(system, msg("user", "What is the capital of Japan?")), Nil,
        Map("enable_thinking" -> false))
      val first = PrefixCache(store = PrefixStore.forModel(root, dir.get, model))
      val (a, ta) = run(first, prompt)
      assertEquals(ta.cachedTokens, 0)
      first.clear()
      val restarted = PrefixCache(store = PrefixStore.forModel(root, dir.get, model))
      assert(restarted.store.nn.size >= 1, "nothing written")
      val (b, tb) = run(restarted, prompt)
      assertEquals(b, a)
      assert(tb.cachedTokens >= 1024, s"${tb.cachedTokens} cached")
      println(s"$repo: ${chat.tokenizer.decode(a).trim} — ${prompt.length} tokens, ${tb.cachedTokens} from disk; " +
        f"prefill ${ta.prefillNanos / 1e6}%.0f ms, then ${tb.prefillNanos / 1e6}%.0f ms")
      restarted.clear()
      model.close()
