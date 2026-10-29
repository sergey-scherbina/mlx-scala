package mlx.llm

import okay.effects.json.Json

/** how long tokenizer.json takes: read, parsed, the tokenizer built */
@main def tokLoad(repo: String): Unit =
  val dir = Models.snapshot(repo).get
  for _ <- 1 to 4 do
    val t0 = System.nanoTime()
    val text = java.nio.file.Files.readString(dir.resolve("tokenizer.json"))
    val t1 = System.nanoTime()
    val j = Json.fromText(text)
    val t2 = System.nanoTime()
    Tokenizer.fromJson(j)
    val t3 = System.nanoTime()
    println(f"read ${(t1 - t0) / 1e6}%.0f ms, parse ${(t2 - t1) / 1e6}%.0f ms, build ${(t3 - t2) / 1e6}%.0f ms")
