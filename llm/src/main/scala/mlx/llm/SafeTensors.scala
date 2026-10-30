package mlx.llm

import java.io.RandomAccessFile
import java.nio.{ByteBuffer, ByteOrder}
import java.nio.channels.FileChannel
import java.nio.charset.StandardCharsets.UTF_8
import java.nio.file.{Files, Path, StandardCopyOption, StandardOpenOption}
import okay.effects.json.Json
import mlx.*

/** A MODEL'S WEIGHTS FROM ITS .safetensors: read into MLX's own memory (`Mlx.loadSafetensors`), or MAPPED — each
 * tensor over the file's pages (`Mlx.mapTensor`): no copy at load, the pages read as first touched, and the OS's to
 * give back when memory is short (they are the file's). Qwen3.6-35B: loaded in 0.15 s instead of 3 s, the process
 * 0.8 GB instead of 19, its decode the same.
 *
 * A tensor maps only where its bytes sit aligned for its type; MLX's writer does not pad the header, so in
 * mlx-community's files they sit off by the header's length, and MLX copies a tensor it cannot use where it is. Such
 * a file is ALIGNED — the same file, its header padded with spaces (as the format allows): IN PLACE (MLX_SCALA_ALIGN=
 * in-place: written beside it, then moved over it — HuggingFace's blob itself, readable as before by any reader,
 * no disk spent), else as a COPY written once under MLX_SCALA_ALIGNED, else ~/.cache/mlx-scala/aligned */
object SafeTensors:
  /** mapped when nothing says otherwise: MLX_SCALA_MMAP=1 */
  def mappedByDefault: Boolean = sys.env.get("MLX_SCALA_MMAP").contains("1")

  def shards(dir: Path): Vector[Path] =
    Files.list(dir).toArray.map(_.asInstanceOf[Path]).filter(_.toString.endsWith(".safetensors")).sorted.toVector

  /** `cold`: tensors mapped without reading ahead — their pages read only as the GPU uses them */
  def load(path: Path, mapped: Boolean, inPlace: Boolean = alignInPlace, cold: String => Boolean = _ => false)
      (using Scope): Map[String, Arr] =
    if mapped then map(if inPlace then alignInPlace(path) else alignedOf(path), cold)
    else Mlx.loadSafetensors(path.toString)

  /** WHERE A TENSOR IS: its file (aligned, as `load` uses it), its bytes, type and shape */
  final case class Located(path: Path, offset: Long, nbytes: Long, dtype: Dtype, shape: Vector[Int])

  def locate(path: Path, inPlace: Boolean = alignInPlace): Map[String, Located] =
    val file = if inPlace then alignInPlace(path) else alignedOf(path)
    val h = header(file)
    h.tensors.map { (name, t) =>
      val Vector(from, to) = t("data_offsets").list.map(_.long): @unchecked
      name -> Located(file, h.base + from, to - from, dtype(t("dtype").string), t("shape").list.map(_.int))
    }.toMap

  /** unaligned files aligned where they are (MLX_SCALA_ALIGN=in-place), not copied */
  def alignInPlace: Boolean = sys.env.get("MLX_SCALA_ALIGN").contains("in-place")

  /** the file aligned where it is (the file a link points to: HuggingFace's blob) — written beside it and moved
   * over it, so a reader sees the old file or the new one, whole — and it returned */
  def alignInPlace(path: Path): Path =
    if !header(path).aligned then
      val real = path.toRealPath()
      System.err.println(s"[Info] $path: its tensors sit unaligned — aligned in place, once ($real)")
      realign(real, real)
    path

  /** a file's header: its tensors (dtype, shape, byte range after the header) and where the data starts */
  final case class Header(json: Json, base: Long):
    def tensors: Vector[(String, Json)] = json.record.filter(_._1 != "__metadata__")
    def aligned: Boolean = tensors.forall { (_, t) =>
      (base + t("data_offsets").list.head.long) % size(dtype(t("dtype").string)) == 0
    }

  def header(path: Path): Header =
    val f = RandomAccessFile(path.toFile, "r")
    try
      val len = Array.ofDim[Byte](8)
      f.readFully(len)
      val n = ByteBuffer.wrap(len).order(ByteOrder.LITTLE_ENDIAN).getLong
      val text = Array.ofDim[Byte](n.toInt)
      f.readFully(text)
      Header(Json.fromText(String(text, UTF_8)), 8 + n)
    finally f.close()

  /** each tensor over the file's pages */
  def map(path: Path, cold: String => Boolean = _ => false)(using Scope): Map[String, Arr] =
    val h = header(path)
    h.tensors.map { (name, t) =>
      val Vector(from, to) = t("data_offsets").list.map(_.long): @unchecked
      name -> Mlx.mapTensor(path.toString, h.base + from, to - from, dtype(t("dtype").string),
        t("shape").list.map(_.int), readAhead = !cold(name))
    }.toMap

  def alignedRoot: Path = sys.env.get("MLX_SCALA_ALIGNED").map(Path.of(_))
    .getOrElse(Path.of(sys.props("user.home"), ".cache", "mlx-scala", "aligned"))

  /** the file itself if its tensors sit aligned, else its aligned copy — written now if there is none yet (named
   * by the file the cache's link points to: HuggingFace's blob, its content's hash) */
  def alignedOf(path: Path, root: Path = alignedRoot): Path =
    if header(path).aligned then path
    else
      val real = path.toRealPath()
      val copy = root.resolve(s"${real.getFileName}-${Files.size(real)}.safetensors")
      if !Files.exists(copy) then
        Files.createDirectories(root)
        System.err.println(s"[Info] $path: its tensors sit unaligned — an aligned copy, once, at $copy")
        realign(real, copy)
      copy

  /** `src` to `dst`, the header padded with spaces so the data starts at a multiple of 64 */
  def realign(src: Path, dst: Path): Unit =
    val h = header(src)
    val n = h.base - 8
    val pad = ((-(h.base)) % 64 + 64) % 64
    val tmp = dst.resolveSibling(dst.getFileName.toString + ".aligning")
    val in = FileChannel.open(src, StandardOpenOption.READ)
    val out = FileChannel.open(tmp, StandardOpenOption.CREATE, StandardOpenOption.WRITE,
      StandardOpenOption.TRUNCATE_EXISTING)
    try
      val head = ByteBuffer.allocate(8).order(ByteOrder.LITTLE_ENDIAN).putLong(n + pad).flip()
      out.write(head)
      val text = ByteBuffer.allocate((n + pad).toInt)
      text.limit(n.toInt)
      while text.hasRemaining do in.read(text, 8 + text.position())
      text.limit((n + pad).toInt)
      while text.hasRemaining do text.put(' '.toByte)
      out.write(text.flip())
      var at = h.base
      val end = in.size()
      while at < end do at += in.transferTo(at, end - at, out)
    finally
      in.close()
      out.close()
    Files.move(tmp, dst, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING)

  private def size(d: Dtype): Int = d match
    case Dtype.Bool | Dtype.UInt8 | Dtype.Int8 => 1
    case Dtype.UInt16 | Dtype.Int16 | Dtype.Float16 | Dtype.BFloat16 => 2
    case Dtype.UInt32 | Dtype.Int32 | Dtype.Float32 => 4
    case _ => 8

  private def dtype(s: String): Dtype = s match
    case "BOOL" => Dtype.Bool
    case "U8" => Dtype.UInt8
    case "U16" => Dtype.UInt16
    case "U32" => Dtype.UInt32
    case "U64" => Dtype.UInt64
    case "I8" => Dtype.Int8
    case "I16" => Dtype.Int16
    case "I32" => Dtype.Int32
    case "I64" => Dtype.Int64
    case "F16" => Dtype.Float16
    case "F32" => Dtype.Float32
    case "F64" => Dtype.Float64
    case "BF16" => Dtype.BFloat16
    case other => throw MlxError(s"a safetensors dtype not known: $other")
