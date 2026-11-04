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

  test("autograd: value and gradients of a loss, as calculus says; descent finds the line"):
    val s = new Scope
    given Scope = s
    val xs = Array(0f, 1f, 2f, 3f)
    val ys = xs.map(x => 2 * x + 1)
    val (x, y) = (floats(xs, 4), floats(ys, 4))
    // mean((w x + b − y)²)
    def loss(using Scope)(args: Seq[Arr]): Arr =
      val e = subtract(add(multiply(args(0), x), args(1)), y)
      divide(sum(multiply(e, e), 0), scalar(4f))
    val (w0, b0) = (0.5f, 0f)
    val (v, g) = valueAndGrad(Seq(scalar(w0), scalar(b0)), Seq(0, 1))(loss)
    val err = xs.indices.map(i => w0 * xs(i) + b0 - ys(i))
    assertEqualsFloat(v.itemFloat, err.map(e => e * e).sum / 4, 1e-4f)
    assertEqualsFloat(g(0).itemFloat, xs.indices.map(i => 2 * err(i) * xs(i)).sum / 4, 1e-4f)
    assertEqualsFloat(g(1).itemFloat, err.map(2 * _).sum / 4, 1e-4f)
    var (w, b) = (w0, b0)
    for _ <- 1 to 500 do
      val t = new Scope
      val (_, gr) = valueAndGrad(Seq(scalar(w)(using t), scalar(b)(using t)), Seq(0, 1))(loss)(using t)
      w -= 0.1f * gr(0).itemFloat
      b -= 0.1f * gr(1).itemFloat
      t.close()
    assertEqualsFloat(w, 2f, 1e-2f)
    assertEqualsFloat(b, 1f, 1e-2f)
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
