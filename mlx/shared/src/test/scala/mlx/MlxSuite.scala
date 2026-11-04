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

  test("conv2d: a 3 x 3 convolution, padded, as the sum says; its gradient flows"):
    val s = new Scope
    given Scope = s
    val (h, w, ci, co) = (4, 5, 2, 3)
    val xs = Array.tabulate(h * w * ci)(i => ((i * 7) % 11 - 5) / 4f)
    val ws = Array.tabulate(co * 9 * ci)(i => ((i * 5) % 13 - 6) / 8f)
    val y = conv2d(floats(xs, 1, h, w, ci), floats(ws, co, 3, 3, ci), padding = 1).toFloats
    for r <- 0 until h; c <- 0 until w; o <- 0 until co do
      var e = 0f
      for dr <- 0 until 3; dc <- 0 until 3; i <- 0 until ci do
        val (rr, cc) = (r + dr - 1, c + dc - 1)
        if rr >= 0 && rr < h && cc >= 0 && cc < w then
          e += xs((rr * w + cc) * ci + i) * ws(((o * 3 + dr) * 3 + dc) * ci + i)
      assertEqualsFloat(y((r * w + c) * co + o), e, 1e-4f)
    val (_, g) = valueAndGrad(Seq(floats(ws, co, 3, 3, ci)), Seq(0))(a =>
      sum(reshape(conv2d(floats(xs, 1, h, w, ci), a(0), padding = 1), h * w * co), 0))
    assertEquals(g(0).shape, Vector(co, 3, 3, ci))
    // d(sum)/d w[o, 1, 1, i] = the sum of channel i over the board (the centre tap sees every point)
    assertEqualsFloat(g(0).toFloats((1 * 3 + 1) * ci), (0 until h * w).map(p => xs(p * ci)).sum, 1e-4f)
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
