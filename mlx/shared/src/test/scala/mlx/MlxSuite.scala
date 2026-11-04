package mlx

import Mlx.*

class MlxSuite extends munit.FunSuite:
  Mlx.init()

  test("ops on the GPU, the same on both platforms"):
    val s = new Scope
    given Scope = s
    val a = ints(Array(1, 2, 3, 4, 5, 6), 2, 3)
    assertEquals(a.shape, Vector(2, 3))
    val f = astype(a, Dtype.Float32)
    val m = matmul(f, transpose(f, 1, 0))              // [2,3]·[3,2]
    assertEquals(m.shape, Vector(2, 2))
    assertEquals(slice(m, Seq(1, 1), Seq(2, 2)).eval().itemFloat, 77f)   // 4·4 + 5·5 + 6·6
    assertEquals(argmax(reshape(a, 6), 0).eval().itemInt, 5)
    val c = concatenate(Seq(a, a), 0)
    assertEquals(c.shape, Vector(4, 3))
    val u = sliceUpdate(zeros(Seq(4, 3), Dtype.Int32), a, Seq(2, 0), Seq(4, 3))
    assertEquals(slice(u, Seq(3, 2), Seq(4, 3)).eval().itemInt, 6)
    s.close()

  test("floats in and out, as they were"):
    val s = new Scope
    given Scope = s
    val data = Array(0f, 1.5f, -2.25f, 131072f, 1e-30f, 0.1f)
    val a = floats(data, 2, 3)
    assertEquals(a.shape, Vector(2, 3))
    assertEquals(a.dtype, Dtype.Float32)
    assertEquals(a.toFloats.toVector, data.toVector)
    assertEquals(add(a, a).toFloats.toVector, data.map(_ * 2).toVector)
    s.close()

  test("a scope frees what it made; a result moved out is kept"):
    val outer = new Scope
    val kept = Scope(outer) { add(scalar(1f), scalar(2f)) }
    assertEquals(outer.size, 1)
    assertEquals(kept.eval().itemFloat, 3f)
    outer.close()

  test("the buffers MLX keeps for reuse: bounded by the limit set, given back when cleared"):
    val before = Mlx.setCacheLimit(0L)
    Scope(new Scope) { add(scalar(1f), scalar(2f)) }.eval()
    assertEquals(Mlx.cacheMemory, 0L)          // a limit of nothing: a freed buffer given back at once
    assertEquals(Mlx.setCacheLimit(before), 0L) // the old limit back, this one's told
    Mlx.clearCache()
    assertEquals(Mlx.cacheMemory, 0L)
