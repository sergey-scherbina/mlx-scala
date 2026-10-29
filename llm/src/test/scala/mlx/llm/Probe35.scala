package mlx.llm

import okay.effects.json.Json

import mlx.*
import mlx.Mlx.*

/** Qwen3.5/3.6 against mlx_lm: the prefill's top logits, the hidden state after the first layers */
@main def probe35(repo: String, refFile: String): Unit =
  val model = Model.load(Models.snapshot(repo).get).asInstanceOf[Qwen35]
  val ref = Json.fromText(java.nio.file.Files.readString(java.nio.file.Path.of(refFile)))
  val prompt = ref("prompt").list.map(_.int).toArray
  val s = new Scope
  given Scope = s
  val ids = ints(prompt, 1, prompt.length)
  val lg = astype(reshape(model.logits(ids, model.newState()), -1), Dtype.Float32)
  val top = argsort(negative(lg), 0)
  val best = (0 until 8).map(i => slice(top, Seq(i), Seq(i + 1)).eval().itemInt)
  println("prefill " + best.map(i => (i, slice(lg, Seq(i), Seq(i + 1)).eval().itemFloat)).mkString(", "))
  for ((sm, sq), i) <- model.probe(ids, 4).zipWithIndex do println(f"layer $i $sm%.4f $sq%.2f")
