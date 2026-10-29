package mlx

import java.lang.foreign.*
import java.lang.foreign.ValueLayout.*
import java.lang.invoke.MethodHandle

/** THE FLAT LAYER on the JVM: libokm's functions through FFM. A handle (a C pointer) is a `long` on both sides — the
 * same register in the C ABI — so a call allocates nothing; int arrays go through a buffer of the thread's own */
private[mlx] object Okm:
  /** -Dokm.lib=<libokm.dylib>, else MLX_SCALA_LIB=<its directory> */
  private val lib = SymbolLookup.libraryLookup(java.nio.file.Path.of(sys.props.get("okm.lib")
    .orElse(sys.env.get("MLX_SCALA_LIB").map(_ + "/libokm.dylib"))
    .getOrElse(sys.error("libokm not found: set -Dokm.lib=<libokm.dylib> or MLX_SCALA_LIB=<its directory>"))),
    Arena.global())
  private val linker = Linker.nativeLinker()
  private def fn(name: String, ret: MemoryLayout | Null, args: MemoryLayout*): MethodHandle =
    val d = if ret == null then FunctionDescriptor.ofVoid(args*) else FunctionDescriptor.of(ret, args*)
    linker.downcallHandle(lib.find(name).orElseThrow(), d)

  private val J = JAVA_LONG
  private val I = JAVA_INT
  private val F = JAVA_FLOAT

  /** four int arrays of 16 and two words per thread: an op's shapes, start/stop, and the out-parameters of a load */
  private val scratch = ThreadLocal.withInitial(() => Arena.global().allocate(4 * 16 * 4 + 16, 8))
  private inline def ints(slot: Int, xs: Array[Int]): Long =
    val s = scratch.get()
    val base = slot * 64L
    var i = 0
    while i < xs.length do
      s.set(JAVA_INT, base + i * 4L, xs(i)); i += 1
    s.address() + base

  private val hInit = fn("okm_init", I)
  private val hError = fn("okm_error", J)
  private val hFree = fn("okm_free", null, J)
  private val hNdim = fn("okm_ndim", I, J)
  private val hDim = fn("okm_dim", I, J, I)
  private val hDtype = fn("okm_dtype", I, J)
  private val hEval = fn("okm_eval", I, J)
  private val hAsyncEval = fn("okm_async_eval", I, J)
  private val hEvalAll = fn("okm_eval_all", I, J, I)
  private val hItemI32 = fn("okm_item_i32", I, J)
  private val hItemF32 = fn("okm_item_f32", F, J)
  private val hFromI32 = fn("okm_from_i32", J, J, J, I)
  private val hScalarF32 = fn("okm_scalar_f32", J, F)
  private val hAdd = fn("okm_add", J, J, J)
  private val hMultiply = fn("okm_multiply", J, J, J)
  private val hMatmul = fn("okm_matmul", J, J, J)
  private val hSigmoid = fn("okm_sigmoid", J, J)
  private val hAstype = fn("okm_astype", J, J, I)
  private val hReshape = fn("okm_reshape", J, J, J, I)
  private val hTranspose = fn("okm_transpose", J, J, J, I)
  private val hArgmax = fn("okm_argmax", J, J, I)
  private val hSoftmax = fn("okm_softmax", J, J, I)
  private val hTake = fn("okm_take", J, J, J, I)
  private val hSlice = fn("okm_slice", J, J, J, J, I)
  private val hSliceUpdate = fn("okm_slice_update", J, J, J, J, J, I)
  private val hConcatenate = fn("okm_concatenate", J, J, I, I)
  private val hZeros = fn("okm_zeros", J, J, I, I)
  private val hQmm = fn("okm_quantized_matmul", J, J, J, J, J, I, I, I)
  private val hSubtract = fn("okm_subtract", J, J, J)
  private val hDivide = fn("okm_divide", J, J, J)
  private val hFloorDivide = fn("okm_floor_divide", J, J, J)
  private val hExp = fn("okm_exp", J, J)
  private val hNegative = fn("okm_negative", J, J)
  private val hLogaddexp = fn("okm_logaddexp", J, J, J)
  private val hContiguous = fn("okm_contiguous", J, J)
  private val hSum = fn("okm_sum", J, J, I, I)
  private val hExpandDims = fn("okm_expand_dims", J, J, I)
  private val hSqueeze = fn("okm_squeeze", J, J, I)
  private val hArgpartition = fn("okm_argpartition", J, J, I, I)
  private val hArgsort = fn("okm_argsort", J, J, I)
  private val hTakeAlong = fn("okm_take_along", J, J, J, I)
  private val hConv1d = fn("okm_conv1d", J, J, J, I, I, I, I)
  private val hScalarI32 = fn("okm_scalar_i32", J, I)
  private val hGatherQmm = fn("okm_gather_qmm", J, J, J, J, J, J, I, I, I, I)
  private val hKernelNew = fn("okm_kernel_new", J, J, J, I, J, I, J, J)
  private val hKernelApply = fn("okm_kernel_apply", I, J, J, I, J, J, J, I, J, J, J, J, J, I, J)
  private val hDequantize = fn("okm_dequantize", J, J, J, J, I, I)
  private val hRmsNorm = fn("okm_rms_norm", J, J, J, F)
  private val hRope = fn("okm_rope", J, J, I, I, F, F, I)
  private val hSdpa = fn("okm_sdpa", J, J, J, J, F, I)
  private val hLoadOpen = fn("okm_load_open", J, J)
  private val hLoadNext = fn("okm_load_next", I, J, J, J)
  private val hLoadClose = fn("okm_load_close", null, J)
  private val hActive = fn("okm_active_memory", J)
  private val hPeak = fn("okm_peak_memory", J)

  def init(): Int = hInit.invokeExact()
  def error(): String =
    val p: Long = hError.invokeExact()
    MemorySegment.ofAddress(p).reinterpret(1024).getString(0)
  def free(a: Long): Unit = hFree.invokeExact(a)
  def ndim(a: Long): Int = hNdim.invokeExact(a)
  def dim(a: Long, i: Int): Int = hDim.invokeExact(a, i)
  def dtype(a: Long): Int = hDtype.invokeExact(a)
  def eval(a: Long): Int = hEval.invokeExact(a)
  def asyncEval(a: Long): Int = hAsyncEval.invokeExact(a)
  def evalAll(as: Array[Long]): Int =
    val arena = Arena.ofConfined()
    try
      val p = arena.allocateFrom(JAVA_LONG, as*)
      val r: Int = hEvalAll.invokeExact(p.address(), as.length)
      r
    finally arena.close()
  def itemI32(a: Long): Int = hItemI32.invokeExact(a)
  def itemF32(a: Long): Float = hItemF32.invokeExact(a)
  def fromI32(data: Array[Int], shape: Array[Int]): Long =
    val arena = Arena.ofConfined()
    try
      val d = arena.allocateFrom(JAVA_INT, data*)
      val r: Long = hFromI32.invokeExact(d.address(), ints(0, shape), shape.length)
      r
    finally arena.close()
  def scalarF32(v: Float): Long = hScalarF32.invokeExact(v)
  def add(a: Long, b: Long): Long = hAdd.invokeExact(a, b)
  def multiply(a: Long, b: Long): Long = hMultiply.invokeExact(a, b)
  def matmul(a: Long, b: Long): Long = hMatmul.invokeExact(a, b)
  def sigmoid(a: Long): Long = hSigmoid.invokeExact(a)
  def astype(a: Long, dtype: Int): Long = hAstype.invokeExact(a, dtype)
  def reshape(a: Long, shape: Array[Int]): Long = hReshape.invokeExact(a, ints(0, shape), shape.length)
  def transpose(a: Long, axes: Array[Int]): Long = hTranspose.invokeExact(a, ints(0, axes), axes.length)
  def argmax(a: Long, axis: Int): Long = hArgmax.invokeExact(a, axis)
  def softmax(a: Long, axis: Int): Long = hSoftmax.invokeExact(a, axis)
  def take(a: Long, idx: Long, axis: Int): Long = hTake.invokeExact(a, idx, axis)
  def slice(a: Long, start: Array[Int], stop: Array[Int]): Long =
    hSlice.invokeExact(a, ints(0, start), ints(1, stop), start.length)
  def sliceUpdate(src: Long, upd: Long, start: Array[Int], stop: Array[Int]): Long =
    hSliceUpdate.invokeExact(src, upd, ints(0, start), ints(1, stop), start.length)
  def concatenate(as: Array[Long], axis: Int): Long =
    val s = scratch.get()
    // handles in slots 2..3 (32 words)
    var i = 0
    while i < as.length do
      s.set(JAVA_LONG, 128L + i * 8L, as(i)); i += 1
    hConcatenate.invokeExact(s.address() + 128L, as.length, axis)
  def zeros(shape: Array[Int], dtype: Int): Long = hZeros.invokeExact(ints(0, shape), shape.length, dtype)
  def quantizedMatmul(x: Long, w: Long, scales: Long, biases: Long, transpose: Boolean, groupSize: Int, bits: Int)
      : Long = hQmm.invokeExact(x, w, scales, biases, if transpose then 1 else 0, groupSize, bits)
  def dequantize(w: Long, scales: Long, biases: Long, groupSize: Int, bits: Int): Long =
    hDequantize.invokeExact(w, scales, biases, groupSize, bits)
  def rmsNorm(x: Long, w: Long, eps: Float): Long = hRmsNorm.invokeExact(x, w, eps)
  def rope(x: Long, dims: Int, traditional: Boolean, base: Float, scale: Float, offset: Int): Long =
    hRope.invokeExact(x, dims, if traditional then 1 else 0, base, scale, offset)
  def sdpa(q: Long, k: Long, v: Long, scale: Float, causal: Boolean): Long =
    hSdpa.invokeExact(q, k, v, scale, if causal then 1 else 0)

  def subtract(a: Long, b: Long): Long = hSubtract.invokeExact(a, b)
  def divide(a: Long, b: Long): Long = hDivide.invokeExact(a, b)
  def floorDivide(a: Long, b: Long): Long = hFloorDivide.invokeExact(a, b)
  def exp(a: Long): Long = hExp.invokeExact(a)
  def negative(a: Long): Long = hNegative.invokeExact(a)
  def logaddexp(a: Long, b: Long): Long = hLogaddexp.invokeExact(a, b)
  def contiguous(a: Long): Long = hContiguous.invokeExact(a)
  def sum(a: Long, axis: Int, keepdims: Boolean): Long = hSum.invokeExact(a, axis, if keepdims then 1 else 0)
  def expandDims(a: Long, axis: Int): Long = hExpandDims.invokeExact(a, axis)
  def squeeze(a: Long, axis: Int): Long = hSqueeze.invokeExact(a, axis)
  def argpartition(a: Long, kth: Int, axis: Int): Long = hArgpartition.invokeExact(a, kth, axis)
  def argsort(a: Long, axis: Int): Long = hArgsort.invokeExact(a, axis)
  def takeAlong(a: Long, idx: Long, axis: Int): Long = hTakeAlong.invokeExact(a, idx, axis)
  def conv1d(x: Long, w: Long, stride: Int, padding: Int, dilation: Int, groups: Int): Long =
    hConv1d.invokeExact(x, w, stride, padding, dilation, groups)
  def scalarI32(v: Int): Long = hScalarI32.invokeExact(v)
  def gatherQmm(x: Long, w: Long, scales: Long, biases: Long, rhs: Long, transpose: Boolean, groupSize: Int,
      bits: Int, sorted: Boolean): Long =
    val (t, so) = (if transpose then 1 else 0, if sorted then 1 else 0)
    hGatherQmm.invokeExact(x, w, scales, biases, rhs, t, groupSize, bits, so)

  private def strings(arena: Arena, xs: Array[String]): MemorySegment =
    val p = arena.allocate(JAVA_LONG, xs.length.max(1).toLong)
    for (x, i) <- xs.zipWithIndex do p.setAtIndex(JAVA_LONG, i.toLong, arena.allocateFrom(x).address())
    p
  def kernelNew(name: String, inputs: Array[String], outputs: Array[String], source: String, header: String): Long =
    val arena = Arena.ofConfined()
    try
      val r: Long = hKernelNew.invokeExact(arena.allocateFrom(name).address(), strings(arena, inputs).address(),
        inputs.length, strings(arena, outputs).address(), outputs.length, arena.allocateFrom(source).address(),
        arena.allocateFrom(header).address())
      r
    finally arena.close()
  def kernelApply(kernel: Long, inputs: Array[Long], shapes: Array[Int], ndims: Array[Int], dtypes: Array[Int],
      grid: Array[Int], group: Array[Int], tNames: Array[String], tKinds: Array[Int], tValues: Array[Int])
      : Array[Long] =
    val arena = Arena.ofConfined()
    try
      def ints(xs: Array[Int]) = arena.allocateFrom(JAVA_INT, (if xs.isEmpty then Array(0) else xs)*).address()
      val outs = arena.allocate(JAVA_LONG, ndims.length.max(1).toLong)
      val in = arena.allocateFrom(JAVA_LONG, inputs*)
      val e: Int = hKernelApply.invokeExact(kernel, in.address(), inputs.length, ints(shapes), ints(ndims),
        ints(dtypes), ndims.length, ints(grid), ints(group), strings(arena, tNames).address(), ints(tKinds),
        ints(tValues), tNames.length, outs.address())
      if e != 0 then Array.empty else Array.tabulate(ndims.length)(i => outs.getAtIndex(JAVA_LONG, i.toLong))
    finally arena.close()

  def loadOpen(path: String): Long =
    val arena = Arena.ofConfined()
    try
      val p = arena.allocateFrom(path)
      val r: Long = hLoadOpen.invokeExact(p.address())
      r
    finally arena.close()
  /** the next tensor of an open file: its name and handle, or null at the end */
  def loadNext(file: Long): (String, Long) | Null =
    val s = scratch.get()
    val at = 4 * 16 * 4L
    val got: Int = hLoadNext.invokeExact(file, s.address() + at, s.address() + at + 8)
    if got == 0 then null
    else
      val key = MemorySegment.ofAddress(s.get(JAVA_LONG, at)).reinterpret(4096).getString(0)
      (key, s.get(JAVA_LONG, at + 8))
  def loadClose(file: Long): Unit = hLoadClose.invokeExact(file)
  def activeMemory(): Long = hActive.invokeExact()
  def peakMemory(): Long = hPeak.invokeExact()
