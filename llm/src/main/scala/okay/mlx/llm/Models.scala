package okay.mlx.llm

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
