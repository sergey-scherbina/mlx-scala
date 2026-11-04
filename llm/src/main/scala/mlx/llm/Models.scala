package mlx.llm

import java.nio.file.{Files, Path}

/** where a model's files are: the HuggingFace cache's snapshot of a repository (the newest) */
object Models:
  def hubCache: Path =
    sys.env.get("HF_HUB_CACHE").map(Path.of(_))
      .orElse(sys.env.get("HF_HOME").map(Path.of(_).resolve("hub")))
      .getOrElse(Path.of(sys.props("user.home"), ".cache", "huggingface", "hub"))

  def snapshot(repo: String, cache: Path = hubCache): Option[Path] =
    val dir = cache.resolve("models--" + repo.replace("/", "--")).resolve("snapshots")
    if !Files.isDirectory(dir) then None
    else
      val all = Files.list(dir).toArray.map(_.asInstanceOf[Path]).filter(Files.isDirectory(_))
      all.sortBy(p => -Files.getLastModifiedTime(p).toMillis).headOption

  /** the repositories in the cache whose model `Model.load` loads (by config.json's model_type), by name — not
   * an embedding model (Qwen3-Embedding's config is a causal LM's too) */
  def cached(cache: Path = hubCache): Vector[String] =
    if !Files.isDirectory(cache) then Vector.empty
    else
      Files.list(cache).toArray.map(_.asInstanceOf[Path].getFileName.toString).toVector
        .filter(n => n.startsWith("models--") && !n.contains("Embedding")).map(_.stripPrefix("models--").replaceFirst("--", "/")).sorted
        .filter(r => snapshot(r, cache).exists(Model.loads))

  // ---- FETCHED without huggingface-cli: the repository's files listed by the Hub's API, downloaded into the cache's
  // layout (models--org--name/snapshots/main/<file>), so `snapshot` finds them. `sbt "llmJVM/runMain mlx.llm.Fetch
  // mlx-community/Qwen3-0.6B-4bit"`
  def fetch(repo: String, cache: Path = hubCache, say: String => Unit = println): Path =
    import java.net.http.{HttpClient, HttpRequest, HttpResponse}
    val client = HttpClient.newBuilder().followRedirects(HttpClient.Redirect.ALWAYS).build()
    val token = sys.env.get("HF_TOKEN").map("Bearer " + _)
    def req(url: String) =
      val b = HttpRequest.newBuilder(java.net.URI.create(url))
      token.fold(b)(t => b.header("Authorization", t)).build()
    val meta = client.send(req(s"https://huggingface.co/api/models/$repo"), HttpResponse.BodyHandlers.ofString())
    require(meta.statusCode() == 200, s"$repo: ${meta.statusCode()} from the Hub")
    val files = "\"rfilename\"\\s*:\\s*\"([^\"]+)\"".r.findAllMatchIn(meta.body()).map(_.group(1)).toVector
      .filterNot(f => f.startsWith(".") || f.endsWith(".md") || f.endsWith(".png"))
    val dir = cache.resolve("models--" + repo.replace("/", "--")).resolve("snapshots").resolve("main")
    Files.createDirectories(dir)
    for f <- files do
      val target = dir.resolve(f)
      if Files.exists(target) && Files.size(target) > 0 then say(s"  $f: there")
      else
        Files.createDirectories(target.getParent)
        say(s"  $f …")
        val r = client.send(req(s"https://huggingface.co/$repo/resolve/main/$f"), HttpResponse.BodyHandlers.ofFile(target))
        require(r.statusCode() == 200, s"$f: ${r.statusCode()}")
    say(s"$repo: ${files.size} files in $dir")
    dir

object Fetch:
  def main(args: Array[String]): Unit = args.foreach(r => Models.fetch(r))
