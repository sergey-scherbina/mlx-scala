package mlx.llm

import mlx.*

/** the models in the HuggingFace cache that load; a model closed gives its memory back */
class ModelsSuite extends munit.FunSuite:
  Mlx.init()

  test("the cache's models that load: no embedding model"):
    val all = Models.cached()
    assume(all.nonEmpty, "no model in the HuggingFace cache")
    assert(all.forall(!_.contains("Embedding")), all.toString)
    assert(all.forall(r => Models.snapshot(r).exists(Model.loads)))

  test("a model closed, the cache cleared: its memory given back"):
    val dir = Models.snapshot("mlx-community/Qwen3-4B-4bit")
    assume(dir.isDefined, "mlx-community/Qwen3-4B-4bit is not in the HuggingFace cache")
    Mlx.clearCache()
    val before = Mlx.activeMemory
    val model = Model.load(dir.get)
    val loaded = Mlx.activeMemory
    model.close()
    Mlx.clearCache()
    assert(loaded - before > (1L << 30), s"loaded: ${(loaded - before) >> 20} MB")
    assert(Mlx.activeMemory - before < (64L << 20), s"after close: ${(Mlx.activeMemory - before) >> 20} MB")
