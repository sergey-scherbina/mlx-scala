package mlx.llm

import mlx.*
import mlx.Mlx.*

/** Qwen3's attention by ops (while training) gives the fused kernel's logits */
class AttentionOpsSuite extends munit.FunSuite:
  override val munitTimeout = scala.concurrent.duration.Duration(5, "min")
  test("attention by ops == the fused kernel"):
    val dir = Models.snapshot("mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ")
    assume(dir.isDefined, "Qwen3-Embedding-0.6B is not in the HuggingFace cache")
    val m = Model.load(dir.get).asInstanceOf[Qwen3]
    val ids = Chat.load(dir.get).tokenizer.encode("def parse(s: String): Option[Int] = s.toIntOption // short")
    def last(train: Boolean) =
      m.training(train)
      val sc = new Scope
      given Scope = sc
      val st = m.newState()
      val h = astype(m.hiddenStates(ints(ids, 1, ids.length), st, all = true), Dtype.Float32).toFloats
      st.close()
      sc.close()
      h
    val (fused, ops) = (last(false), last(true))
    m.training(false)
    // the cosine of the two at each position
    val w = fused.length / ids.length
    val cos = ids.indices.map { p =>
      val (a, b) = (fused.slice(p * w, p * w + w), ops.slice(p * w, p * w + w))
      a.zip(b).map((x, y) => x.toDouble * y).sum /
        math.sqrt(a.map(x => x.toDouble * x).sum * b.map(x => x.toDouble * x).sum)
    }
    println(f"attention by ops vs fused: the cosine at each position, at least ${cos.min}%.6f; |x| up to " +
      f"${fused.map(math.abs).max}%.1f")
    assert(cos.min > 0.999, s"${cos.min}")
    m.close()
