package mlx.llm

import java.nio.file.Path
import mlx.*
import mlx.Mlx.*

/** TEXT EMBEDDINGS (Qwen3-Embedding, any model with `hiddenStates`): a text's tokens ended by `<|endoftext|>`, the
 * hidden state of that last token after the final norm, L2-normed — as Qwen3-Embedding's own pooling (last token).
 * A QUERY is given its task ("Instruct: <task>\nQuery:<text>", no space: as the model card writes it), a document as
 * it is. On the CPU when `cpu` (MLX's CPU backend), the GPU left to a big model; one text a pass (no padding to
 * mask), at most `maxTokens` of it */
final class Embedder(val model: Model, tokenizer: Tokenizer, val cpu: Boolean = false, maxTokens: Int = 512):
  private val end = tokenizer.id("<|endoftext|>")
  require(end >= 0, "the tokenizer has no <|endoftext|>")

  def dim: Int = vectors(Seq("a")).head.length

  /** the texts' embeddings, each of unit length */
  def vectors(texts: Seq[String]): Vector[Array[Float]] =
    def run = texts.map { t =>
      val ids = tokens(t)
      val st = model.newState()
      val s = new Scope
      try
        given Scope = s
        val h = astype(model.hiddenStates(ints(ids, 1, ids.length), st, all = false), Dtype.Float32)
        val v = h.toFloats
        val norm = math.sqrt(v.map(x => x.toDouble * x).sum).toFloat
        v.map(_ / math.max(norm, 1e-12f))
      finally
        s.close()
        st.close()
    }.toVector
    if cpu then Mlx.onCpu(run) else run

  /** a text's tokens as embedded: at most `maxTokens`, `<|endoftext|>` last */
  def tokens(text: String): Array[Int] = tokenizer.encode(text).take(maxTokens - 1) :+ end

  /** a query as embedded: its task stated as Qwen3-Embedding was trained with */
  def queryText(task: String, text: String): String = s"Instruct: $task\nQuery:$text"

  /** a query's embedding: its task stated as Qwen3-Embedding was trained with */
  def query(task: String, text: String): Array[Float] = vectors(Seq(queryText(task, text))).head

object Embedder:
  /** the model of a directory; for the CPU its weights unpacked to float32 (BLAS, not MLX's slow quantized kernel) */
  /** with an `adapter` (`EmbedTrain`'s, LoRA): put on, for good (its arrays the model's from now on) */
  def load(dir: Path, cpu: Boolean = false, adapter: Path | Null = null): Embedder =
    val m = Model.load(dir)
    if cpu then m match
      case q: Qwen3 => q.densify()
      case _ =>
    if adapter != null then m match
      case a: Adaptable => Lora.attach(a, adapter.toString): Unit
      case _ => sys.error(s"$dir's model takes no adapters")
    Embedder(m, Chat.load(dir).tokenizer, cpu)

  /** the code-search adapter `embedTrainBench` saves for a model of HuggingFace's cache, if there */
  def codeSearch(dir: Path): Option[Path] =
    val p = java.nio.file.Paths.get(sys.props("user.home"), ".cache", "mlx-scala", "embed",
      dir.getParent.getParent.getFileName.toString, "code-search.safetensors")
    Option.when(java.nio.file.Files.exists(p))(p)

  def cosine(a: Array[Float], b: Array[Float]): Float =
    var s = 0f
    var i = 0
    while i < a.length do
      s += a(i) * b(i)
      i += 1
    s
