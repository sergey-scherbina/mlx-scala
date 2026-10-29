package okay.mlx

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

  test("a scope frees what it made; a result moved out is kept"):
    val outer = new Scope
    val kept = Scope(outer) { add(scalar(1f), scalar(2f)) }
    assertEquals(outer.size, 1)
    assertEquals(kept.eval().itemFloat, 3f)
    outer.close()
