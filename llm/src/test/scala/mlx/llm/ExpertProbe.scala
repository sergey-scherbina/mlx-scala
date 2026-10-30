package mlx.llm

import scala.collection.mutable

/** HOW HOT THE EXPERTS RUN: the experts a decode picks, layer by layer, and how often a cache of C per layer (least
 * recently used out) would have had them — `llmJVM/Test/runMain mlx.llm.expertProbe <repo> <prompt file> [tokens]`.
 * Also: the decode's speed with the routing read every layer (what a cache must pay to know its misses) */
@main def expertProbe(repo: String, promptFile: String, tokens: Int = 256): Unit =
  val dir = Models.snapshot(repo).get
  val model = Model.load(dir, mapped = true)
  val chat = Chat.load(dir)
  val text = java.nio.file.Files.readString(java.nio.file.Path.of(promptFile))
  val prompt = chat.ask(text, Map("enable_thinking" -> false))
  def decode(): Double =
    val s = Session(model, prompt, tokens, _ => false)
    Iterator.continually(s.next()).takeWhile(_.isDefined).foreach(_ => ())
    s.close()
    s.timing.decodePerSecond
  decode()
  val plain = (1 to 2).map(_ => decode()).max
  // decode steps: n = 1 rows; the prefill's rows (n > k ids) kept apart
  val steps = mutable.Map.empty[Int, mutable.ArrayBuffer[Array[Int]]]
  var prefill = 0
  Qwen35.routes = (layer, ids) =>
    if ids.length > 8 then prefill += 1 else steps.getOrElseUpdate(layer, mutable.ArrayBuffer()) += ids
  val traced = decode()
  Qwen35.routes = null
  println(f"DECODE plain $plain%.1f tok/s, the routing read every layer $traced%.1f tok/s")
  val layers = steps.keys.toVector.sorted
  val k = steps(layers.head).head.length
  println(s"layers ${layers.size}, steps ${steps(layers.head).size}, k $k")
  for c <- Seq(8, 16, 32, 64, 96, 128) do
    var hits = 0L
    var all = 0L
    for l <- layers do
      val lru = mutable.LinkedHashSet.empty[Int]
      for ids <- steps(l) do
        for e <- ids do
          all += 1
          if lru.remove(e) then hits += 1
          lru += e
          if lru.size > c then lru -= lru.head
    val missesPerToken = (all - hits).toDouble / steps(layers.head).size
    println(f"C $c%3d of 256: hits ${100.0 * hits / all}%5.1f%%, misses a token $missesPerToken%6.1f")
  val distinct = layers.map(l => steps(l).flatten.distinct.size)
  println(s"distinct experts a layer over the decode: min ${distinct.min} max ${distinct.max}")
  model.close()
