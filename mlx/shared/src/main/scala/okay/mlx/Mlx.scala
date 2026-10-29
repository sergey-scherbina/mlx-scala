package okay.mlx

import scala.collection.mutable.ArrayBuffer

/** an array's element type: mlx_dtype's numbering */
enum Dtype:
  case Bool, UInt8, UInt16, UInt32, UInt64, Int8, Int16, Int32, Int64, Float16, Float32, Float64, BFloat16, Complex64

/** AN MLX ARRAY: a handle, lazy until evaluated, owned by a scope — freed when it closes, unless moved to another */
final class Arr private[mlx] (private[mlx] val h: Long, private[mlx] var owner: Scope):
  def ndim: Int = Okm.ndim(h)
  def dim(i: Int): Int = Okm.dim(h, if i < 0 then ndim + i else i)
  def shape: Vector[Int] = Vector.tabulate(ndim)(dim)
  def dtype: Dtype = Dtype.fromOrdinal(Okm.dtype(h))
  /** computed now */
  def eval(): this.type = { check(Okm.eval(h)); this }
  /** computed in the background: a later `eval` or `item` waits for it */
  def evalAsync(): this.type = { check(Okm.asyncEval(h)); this }
  def itemInt: Int = Okm.itemI32(h)
  def itemFloat: Float = Okm.itemF32(h)
  /** moved to `to`: kept when its scope closes */
  def moveTo(to: Scope): this.type =
    owner.forget(this); to.adopt(this); this
  override def toString: String = s"Arr(${shape.mkString("x")}, $dtype)"

private def check(code: Int): Unit = if code != 0 then throw MlxError(Okm.error())

/** an op MLX refused, with its message */
final class MlxError(message: String) extends RuntimeException(message)

/** A SCOPE: the arrays made in it, freed together when it closes */
final class Scope:
  private val arrays = ArrayBuffer.empty[Arr]
  private[mlx] def adopt(a: Arr): Unit = { a.owner = this; arrays += a }
  private[mlx] def forget(a: Arr): Unit =
    val i = arrays.lastIndexOf(a)
    if i >= 0 then arrays.remove(i)
  private[mlx] def own(h: Long): Arr =
    if h == 0L then throw MlxError(Okm.error())
    val a = Arr(h, this)
    arrays += a
    a
  /** freed now */
  def release(a: Arr): Unit =
    forget(a)
    Okm.free(a.h)
  def size: Int = arrays.size
  def close(): Unit =
    var i = 0
    while i < arrays.length do
      Okm.free(arrays(i).h); i += 1
    arrays.clear()

object Scope:
  /** `f` in a new scope, closed after it — what `f` returns is moved out first, to `out` */
  def apply[A](out: Scope)(f: Scope ?=> A): A =
    val s = new Scope
    try
      val a = f(using s)
      a match
        case arr: Arr if arr.owner eq s => arr.moveTo(out)
        case _ =>
      a
    finally s.close()

/** THE OPS: each makes its result in the scope given. Shapes and axes as in MLX */
object Mlx:
  private var started = false
  /** MLX ready: its error handler and the GPU stream — once */
  def init(): Unit = synchronized { if !started then { check(Okm.init()); started = true } }

  def ints(data: Array[Int], shape: Int*)(using s: Scope): Arr = s.own(Okm.fromI32(data, shape.toArray))
  def scalar(v: Float)(using s: Scope): Arr = s.own(Okm.scalarF32(v))
  def zeros(shape: Seq[Int], dtype: Dtype)(using s: Scope): Arr = s.own(Okm.zeros(shape.toArray, dtype.ordinal))

  def add(a: Arr, b: Arr)(using s: Scope): Arr = s.own(Okm.add(a.h, b.h))
  def multiply(a: Arr, b: Arr)(using s: Scope): Arr = s.own(Okm.multiply(a.h, b.h))
  def matmul(a: Arr, b: Arr)(using s: Scope): Arr = s.own(Okm.matmul(a.h, b.h))
  def sigmoid(a: Arr)(using s: Scope): Arr = s.own(Okm.sigmoid(a.h))
  /** x · sigmoid(x) */
  def silu(a: Arr)(using s: Scope): Arr = multiply(a, sigmoid(a))
  def astype(a: Arr, dtype: Dtype)(using s: Scope): Arr = s.own(Okm.astype(a.h, dtype.ordinal))
  def reshape(a: Arr, shape: Int*)(using s: Scope): Arr = s.own(Okm.reshape(a.h, shape.toArray))
  def transpose(a: Arr, axes: Int*)(using s: Scope): Arr = s.own(Okm.transpose(a.h, axes.toArray))
  def argmax(a: Arr, axis: Int)(using s: Scope): Arr = s.own(Okm.argmax(a.h, axis))
  def softmax(a: Arr, axis: Int)(using s: Scope): Arr = s.own(Okm.softmax(a.h, axis))
  def take(a: Arr, indices: Arr, axis: Int)(using s: Scope): Arr = s.own(Okm.take(a.h, indices.h, axis))
  def slice(a: Arr, start: Seq[Int], stop: Seq[Int])(using s: Scope): Arr =
    s.own(Okm.slice(a.h, start.toArray, stop.toArray))
  def sliceUpdate(src: Arr, update: Arr, start: Seq[Int], stop: Seq[Int])(using s: Scope): Arr =
    s.own(Okm.sliceUpdate(src.h, update.h, start.toArray, stop.toArray))
  def concatenate(as: Seq[Arr], axis: Int)(using s: Scope): Arr = s.own(Okm.concatenate(as.map(_.h).toArray, axis))

  /** x · wᵀ (`transpose`) with w quantized: packed weights, a scale and a bias per group */
  def quantizedMatmul(x: Arr, w: Arr, scales: Arr, biases: Arr, transpose: Boolean, groupSize: Int, bits: Int)(
    using s: Scope): Arr = s.own(Okm.quantizedMatmul(x.h, w.h, scales.h, biases.h, transpose, groupSize, bits))
  def dequantize(w: Arr, scales: Arr, biases: Arr, groupSize: Int, bits: Int)(using s: Scope): Arr =
    s.own(Okm.dequantize(w.h, scales.h, biases.h, groupSize, bits))

  // MLX's fused kernels
  def rmsNorm(x: Arr, weight: Arr, eps: Float)(using s: Scope): Arr = s.own(Okm.rmsNorm(x.h, weight.h, eps))
  def rope(x: Arr, dims: Int, traditional: Boolean, base: Float, scale: Float, offset: Int)(using s: Scope): Arr =
    s.own(Okm.rope(x.h, dims, traditional, base, scale, offset))
  /** softmax(q·kᵀ·scale)·v over [batch, heads, length, head_dim]; the causal mask when asked */
  def attention(q: Arr, k: Arr, v: Arr, scale: Float, causal: Boolean)(using s: Scope): Arr =
    s.own(Okm.sdpa(q.h, k.h, v.h, scale, causal))

  /** every tensor of a .safetensors file, by name, in the scope given */
  def loadSafetensors(path: String)(using s: Scope): Map[String, Arr] =
    val f = Okm.loadOpen(path)
    if f == 0L then throw MlxError(s"$path: ${Okm.error()}")
    try
      val b = Map.newBuilder[String, Arr]
      var next = Okm.loadNext(f)
      while next != null do
        val (name, h) = next.asInstanceOf[(String, Long)]
        b += name -> s.own(h)
        next = Okm.loadNext(f)
      b.result()
    finally Okm.loadClose(f)

  /** all computed now, together */
  def evalAll(as: Seq[Arr]): Unit = check(Okm.evalAll(as.map(_.h).toArray))

  def activeMemory: Long = Okm.activeMemory()
  def peakMemory: Long = Okm.peakMemory()
