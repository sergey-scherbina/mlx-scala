package mlx.llm

import mlx.*
import mlx.Mlx.*

/** Qwen3.5-4B trainable: the delta rule by ops gives the kernel's logits, and LoRA steps lower an example's loss */
class Qwen35LoraSuite extends munit.FunSuite:
  override val munitTimeout = scala.concurrent.duration.Duration(30, "min")

  test("Qwen3.5-4B: the ops path as the kernel; LoRA lowers the loss"):
    val dir = Models.snapshot("mlx-community/Qwen3.5-4B-MLX-4bit")
    assume(dir.isDefined, "Qwen3.5-4B is not in the HuggingFace cache")
    val chat = Chat.load(dir.get)
    val model = Model.load(dir.get, mapped = true).asInstanceOf[Qwen35]
    val prompt = chat.ask("Write `def isImpure(symbol: String): Boolean` with symdex's `Equivalence.impure`.",
      Map("enable_thinking" -> false))
    val answer = chat.tokenizer.encode(
      "```scala\ndef isImpure(symbol: String): Boolean = Equivalence.impure.exists(symbol.startsWith)\n```")
    val ids = prompt ++ answer
    def logits(train: Boolean): Array[Float] =
      model.training(train)
      val s = new Scope
      val st = model.newState()
      try astype(model.logitsAll(ints(ids, 1, ids.length)(using s), st)(using s), Dtype.Float32)(using s).toFloats
      finally { st.close(); s.close(); model.training(false) }
    val (fast, slow) = (logits(false), logits(true))
    val v = fast.length / ids.length
    val argmaxSame = ids.indices.count { p =>
      def am(xs: Array[Float]) = (0 until v).maxBy(j => xs(p * v + j))
      am(fast) == am(slow)
    }
    val diff = fast.indices.map(i => math.abs(fast(i) - slow(i))).max
    println(s"ops against the kernel: max |Δlogit| $diff, argmax equal at $argmaxSame of ${ids.length} positions")
    assert(argmaxSame >= ids.length * 95 / 100, s"$argmaxSame of ${ids.length}")
    val lora = Lora(model, rank = 8, projs = Seq("o", "out", "down"), lr = 1e-3f)
    println(s"adapters: ${lora.names.size}, ${lora.size} parameters")
    val first = lora.evaluate(ids, prompt.length)
    val t0 = System.nanoTime
    val losses = (1 to 10).map(_ => lora.step(ids, prompt.length))
    val after = lora.evaluate(ids, prompt.length)
    println(f"loss: before $first%.4f, steps ${losses.map(l => f"$l%.3f").mkString(" ")}, after $after%.4f " +
      f"(${(System.nanoTime - t0) / 1e9 / 10}%.2f s a step)")
    assert(after < first / 2, s"$first → $after")
    lora.close()
    model.close()
