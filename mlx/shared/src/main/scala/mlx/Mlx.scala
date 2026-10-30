package mlx

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
  /** all of it, computed now, as ints (row-major) */
  def toInts: Array[Int] =
    val n = shape.product
    val r = Okm.toI32(h, n)
    if r == null then throw MlxError(Okm.error()) else r
  def itemFloat: Float = Okm.itemF32(h)
  /** a handle of its own on the same array, in `to`: the array kept as long as either is (arrays are immutable — a
   * snapshot is a handle) */
  def retain(to: Scope): Arr = to.own(Okm.retain(h))
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
  // COMPILED (MLX's compile, shapeless): the elementwise functions mlx_lm compiles — the same ops in the same order,
  // the same numbers, one fused kernel each
  /** x · sigmoid(x) */
  def silu(a: Arr)(using s: Scope): Arr = s.own(Okm.silu(a.h))
  /** silu(gate) · x */
  def swiglu(gate: Arr, x: Arr)(using s: Scope): Arr = s.own(Okm.swiglu(gate.h, x.h))
  /** (silu(gate) · x) in float32, as `h`'s type */
  def preciseSwiglu(h: Arr, gate: Arr, x: Arr)(using s: Scope): Arr = s.own(Okm.preciseSwiglu(h.h, gate.h, x.h))
  /** exp(−exp(A_log) · softplus(a + dt_bias)) in float32: Gated DeltaNet's decay */
  def computeG(aLog: Arr, a: Arr, dtBias: Arr)(using s: Scope): Arr = s.own(Okm.computeG(aLog.h, a.h, dtBias.h))
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

  def subtract(a: Arr, b: Arr)(using s: Scope): Arr = s.own(Okm.subtract(a.h, b.h))
  def divide(a: Arr, b: Arr)(using s: Scope): Arr = s.own(Okm.divide(a.h, b.h))
  def floorDivide(a: Arr, b: Arr)(using s: Scope): Arr = s.own(Okm.floorDivide(a.h, b.h))
  def exp(a: Arr)(using s: Scope): Arr = s.own(Okm.exp(a.h))
  def negative(a: Arr)(using s: Scope): Arr = s.own(Okm.negative(a.h))
  def logaddexp(a: Arr, b: Arr)(using s: Scope): Arr = s.own(Okm.logaddexp(a.h, b.h))
  /** log(1 + eˣ) */
  def softplus(a: Arr)(using s: Scope): Arr = logaddexp(a, astype(scalar(0f), a.dtype))
  def contiguous(a: Arr)(using s: Scope): Arr = s.own(Okm.contiguous(a.h))
  def sum(a: Arr, axis: Int, keepdims: Boolean = false)(using s: Scope): Arr = s.own(Okm.sum(a.h, axis, keepdims))
  def expandDims(a: Arr, axis: Int)(using s: Scope): Arr = s.own(Okm.expandDims(a.h, axis))
  def squeeze(a: Arr, axis: Int)(using s: Scope): Arr = s.own(Okm.squeeze(a.h, axis))
  def argpartition(a: Arr, kth: Int, axis: Int)(using s: Scope): Arr = s.own(Okm.argpartition(a.h, kth, axis))
  def argsort(a: Arr, axis: Int)(using s: Scope): Arr = s.own(Okm.argsort(a.h, axis))
  def takeAlong(a: Arr, indices: Arr, axis: Int)(using s: Scope): Arr = s.own(Okm.takeAlong(a.h, indices.h, axis))
  /** a 1-D convolution: x [batch, length, channels], w [out, kernel, in / groups] */
  def conv1d(x: Arr, w: Arr, stride: Int = 1, padding: Int = 0, dilation: Int = 1, groups: Int = 1)(
    using s: Scope): Arr = s.own(Okm.conv1d(x.h, w.h, stride, padding, dilation, groups))
  def int(v: Int)(using s: Scope): Arr = s.own(Okm.scalarI32(v))

  /** experts' quantized weights by index: x · w[indices]ᵀ — a mixture of experts' layer */
  def gatherQmm(x: Arr, w: Arr, scales: Arr, biases: Arr, indices: Arr, transpose: Boolean, groupSize: Int, bits: Int,
      sorted: Boolean = false)(using s: Scope): Arr =
    s.own(Okm.gatherQmm(x.h, w.h, scales.h, biases.h, indices.h, transpose, groupSize, bits, sorted))

  // sampling
  def cumsum(a: Arr, axis: Int)(using s: Scope): Arr = s.own(Okm.cumsum(a.h, axis))
  def greater(a: Arr, b: Arr)(using s: Scope): Arr = s.own(Okm.greater(a.h, b.h))
  /** `a` where `cond`, else `b` */
  def where(cond: Arr, a: Arr, b: Arr)(using s: Scope): Arr = s.own(Okm.where(cond.h, a.h, b.h))
  /** an index drawn along `axis`, each with the probability softmax(logits) gives it */
  def categorical(logits: Arr, axis: Int)(using s: Scope): Arr = s.own(Okm.categorical(logits.h, axis))
  /** the random generator's seed: the draws after it the same each run */
  def randomSeed(seed: Long): Unit = check(Okm.randomSeed(seed))

  /** x · wᵀ (`transpose`) with w quantized: packed weights, a scale and a bias per group */
  def quantizedMatmul(x: Arr, w: Arr, scales: Arr, biases: Arr, transpose: Boolean, groupSize: Int, bits: Int)(
    using s: Scope): Arr = s.own(Okm.quantizedMatmul(x.h, w.h, scales.h, biases.h, transpose, groupSize, bits))
  def dequantize(w: Arr, scales: Arr, biases: Arr, groupSize: Int, bits: Int)(using s: Scope): Arr =
    s.own(Okm.dequantize(w.h, scales.h, biases.h, groupSize, bits))

  // MLX's fused kernels
  /** `weight` may be `none` */
  def rmsNorm(x: Arr, weight: Arr, eps: Float)(using s: Scope): Arr = s.own(Okm.rmsNorm(x.h, weight.h, eps))
  def rope(x: Arr, dims: Int, traditional: Boolean, base: Float, scale: Float, offset: Int)(using s: Scope): Arr =
    s.own(Okm.rope(x.h, dims, traditional, base, scale, offset))
  /** softmax(q·kᵀ·scale)·v over [batch, heads, length, head_dim]; the causal mask when asked */
  def attention(q: Arr, k: Arr, v: Arr, scale: Float, causal: Boolean)(using s: Scope): Arr =
    s.own(Okm.sdpa(q.h, k.h, v.h, scale, causal))

  /** arrays into a .safetensors file, by name, with string metadata */
  def saveSafetensors(path: String, arrays: Seq[(String, Arr)], metadata: Map[String, String] = Map.empty): Unit =
    check(Okm.saveSafetensors(path, arrays.map(_._1).toArray, arrays.map(_._2.h).toArray, metadata.keys.toArray,
      metadata.values.toArray))

  /** the null array: an optional argument not given (a norm without a weight) */
  val none: Arr = Arr(0L, null)

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

  /** bytes [offset, offset + nbytes) of a file as an array of `dtype` and `shape` over the file's own pages, mapped
   * — no copy (Metal wraps them): read when first touched (and ahead, in the background: `readAhead`), the OS's to
   * give back when memory is short */
  def mapTensor(path: String, offset: Long, nbytes: Long, dtype: Dtype, shape: Seq[Int], readAhead: Boolean = true)
      (using s: Scope): Arr =
    val h = Okm.mapTensor(path, offset, nbytes, dtype.ordinal, shape.toArray, readAhead)
    if h == 0L then throw MlxError(Okm.error())
    s.own(h)

  /** zeros of `shape`, a contiguous buffer of its own, computed: memory `readInto` may fill */
  def buffer(shape: Seq[Int], dtype: Dtype)(using s: Scope): Arr =
    val h = Okm.buffer(shape.toArray, dtype.ordinal)
    if h == 0L then throw MlxError(Okm.error())
    s.own(h)

  /** a file open to read from (`readInto`) */
  def openFd(path: String): Int =
    val fd = Okm.openFd(path)
    if fd < 0 then throw MlxError(Okm.error()) else fd
  def closeFd(fd: Int): Unit = Okm.closeFd(fd)

  /** n bytes of a file at `fileOffset` into `dst`'s memory (a `buffer`) at `dstOffset` — the GPU done with that part */
  def readInto(fd: Int, dst: Arr, dstOffset: Long, fileOffset: Long, n: Long): Unit =
    check(Okm.readInto(fd, dst.h, dstOffset, fileOffset, n))

  /** all computed now, together */
  def evalAll(as: Seq[Arr]): Unit = check(Okm.evalAll(as.map(_.h).toArray))

  /** the graph that computes `a`, Graphviz text, into a file: what a step is made of (not evaluated) */
  def exportDot(a: Arr, path: String): Unit = check(Okm.exportDot(a.h, path))

  def activeMemory: Long = Okm.activeMemory()
  /** the buffers MLX keeps for reuse freed: after a model is closed, its memory given back */
  def clearCache(): Unit = check(Okm.clearCache())
  def peakMemory: Long = Okm.peakMemory()

/** A CUSTOM METAL KERNEL: its source, run on arrays — MLX's `fast.metal_kernel`. Applied with its outputs' shapes and
 * types, the grid and threadgroup, template arguments (ints, dtypes, bools) */
final class Kernel(name: String, inputs: Seq[String], outputs: Seq[String], source: String, header: String = ""):
  private lazy val h = Okm.kernelNew(name, inputs.toArray, outputs.toArray, source, header)
  def apply(args: Seq[Arr], outShapes: Seq[Seq[Int]], outTypes: Seq[Dtype], grid: (Int, Int, Int),
      group: (Int, Int, Int), template: Seq[(String, Int | Dtype | Boolean)])(using s: Scope): Vector[Arr] =
    val kinds = template.map(_._2 match { case _: Int => 0; case _: Dtype => 1; case _: Boolean => 2 })
    val values = template.map(_._2 match
      case i: Int => i
      case d: Dtype => d.ordinal
      case b: Boolean => if b then 1 else 0)
    val outs = Okm.kernelApply(h, args.map(_.h).toArray, outShapes.flatten.toArray, outShapes.map(_.length).toArray,
      outTypes.map(_.ordinal).toArray, Array(grid._1, grid._2, grid._3), Array(group._1, group._2, group._3),
      template.map(_._1).toArray, kinds.toArray, values.toArray)
    if outs.isEmpty then throw MlxError(Okm.error())
    outs.toVector.map(s.own)
