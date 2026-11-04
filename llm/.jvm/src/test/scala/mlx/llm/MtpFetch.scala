package mlx.llm

import okay.effects.json.Json

import java.net.URI
import java.net.http.{HttpClient, HttpRequest, HttpResponse}
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files, Path}

/** THE DRAFT LAYER'S PACK for a model in HuggingFace's cache (`Mtp`): the original checkpoint's `mtp.*` tensors
 * read by their byte ranges (no shard downloaded whole: 230 MB of Qwen3.5-4B's, 1.6 GB of Qwen3.6-35B-A3B's), and
 * its final norm (to tell whether the norms are shifted), into a bf16 file; then quantized as the model is and
 * written where `Qwen35.load` looks (`Mtp.path`).
 *
 * `sbt "llmJVM/Test/runMain mlx.llm.mtpFetch mlx-community/Qwen3.5-4B-MLX-4bit Qwen/Qwen3.5-4B"` */
@main def mtpFetch(repo: String, original: String): Unit =
  val dir = Models.snapshot(repo).getOrElse(sys.error(s"$repo is not in the HuggingFace cache"))
  val out = Mtp.path(dir)
  val source = out.resolveSibling("mtp-bf16.safetensors")
  Files.createDirectories(out.getParent)
  HfRanges.tensors(original, source, k => k.startsWith("mtp.") || k == "model.language_model.norm.weight" ||
    k == "model.norm.weight", k => if k.startsWith("mtp.") then k else "check.norm")
  Mtp.convert(source, dir, out)
  Files.delete(source)
  System.err.println(s"[mtp] $out: ${Files.size(out) >> 20} MB")

object HfRanges:
  private val http = HttpClient.newBuilder().followRedirects(HttpClient.Redirect.ALWAYS).build()
  private def url(repo: String, file: String) = s"https://huggingface.co/$repo/resolve/main/$file"
  private def get(u: String, from: Long, to: Long): Array[Byte] =
    val r = http.send(HttpRequest.newBuilder(URI.create(u)).header("Range", s"bytes=$from-$to").build(),
      HttpResponse.BodyHandlers.ofByteArray())
    require(r.statusCode == 206 || r.statusCode == 200, s"$u: HTTP ${r.statusCode}")
    r.body

  /** the tensors of a repository's checkpoint that `wanted` says, by their byte ranges, into a safetensors file
   * `out` under the names `rename` gives */
  def tensors(repo: String, out: Path, wanted: String => Boolean, rename: String => String): Unit =
    val index = Json.fromText(new String(get(url(repo, "model.safetensors.index.json"), 0, Long.MaxValue - 1), UTF_8))
    val names = index("weight_map").record.map((k, v) => k -> v.string).filter((k, _) => wanted(k))
    require(names.nonEmpty, s"$repo: no tensor wanted")
    // each shard's header, then each tensor's bytes
    val ts = names.groupBy(_._2).toVector.flatMap { (file, ts) =>
      val u = url(repo, file)
      val n = ByteBuffer.wrap(get(u, 0, 7)).order(ByteOrder.LITTLE_ENDIAN).getLong
      val header = Json.fromText(new String(get(u, 8, 7 + n), UTF_8))
      ts.map(_._1).sorted.map { k =>
        val t = header(k)
        val Vector(from, to) = t("data_offsets").list.map(_.long): @unchecked
        System.err.println(s"[mtp] $k ${t("shape").list.map(_.int).mkString("x")} ${(to - from) >> 20} MB")
        (rename(k), t("dtype").string, t("shape").list.map(_.int), get(u, 8 + n + from, 8 + n + to - 1))
      }
    }
    // a safetensors file: its header's length, the header, the data
    var at = 0L
    val entries = ts.map { (k, dt, shape, bytes) =>
      val shapes = shape.mkString(",")
      val e = s"\"$k\":{\"dtype\":\"$dt\",\"shape\":[$shapes],\"data_offsets\":[$at,${at + bytes.length}]}"
      at += bytes.length
      e
    }
    val head = entries.mkString("{", ",", "}").getBytes(UTF_8)
    val pad = head ++ Array.fill[Byte]((8 - head.length % 8) % 8)(' '.toByte)
    val o = Files.newOutputStream(out)
    try
      o.write(ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(pad.length.toLong).array())
      o.write(pad)
      ts.foreach(t => o.write(t._4))
    finally o.close()
