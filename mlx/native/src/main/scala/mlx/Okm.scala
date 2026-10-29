package mlx

import scala.scalanative.unsafe.*

/** libokm's functions as C declares them; a handle (a C pointer) declared a 64-bit integer: the same register in
 * the C ABI */
@extern @link("okm")
private object C:
  def okm_init(): CInt = extern
  def okm_error(): CString = extern
  def okm_free(a: Long): Unit = extern
  def okm_ndim(a: Long): CInt = extern
  def okm_dim(a: Long, i: CInt): CInt = extern
  def okm_dtype(a: Long): CInt = extern
  def okm_eval(a: Long): CInt = extern
  def okm_async_eval(a: Long): CInt = extern
  def okm_eval_all(as: Ptr[Long], n: CInt): CInt = extern
  def okm_item_i32(a: Long): CInt = extern
  def okm_item_f32(a: Long): CFloat = extern
  def okm_from_i32(data: Ptr[CInt], shape: Ptr[CInt], ndim: CInt): Long = extern
  def okm_scalar_f32(v: CFloat): Long = extern
  def okm_add(a: Long, b: Long): Long = extern
  def okm_multiply(a: Long, b: Long): Long = extern
  def okm_matmul(a: Long, b: Long): Long = extern
  def okm_sigmoid(a: Long): Long = extern
  def okm_astype(a: Long, dtype: CInt): Long = extern
  def okm_reshape(a: Long, shape: Ptr[CInt], n: CInt): Long = extern
  def okm_transpose(a: Long, axes: Ptr[CInt], n: CInt): Long = extern
  def okm_argmax(a: Long, axis: CInt): Long = extern
  def okm_softmax(a: Long, axis: CInt): Long = extern
  def okm_take(a: Long, idx: Long, axis: CInt): Long = extern
  def okm_slice(a: Long, start: Ptr[CInt], stop: Ptr[CInt], n: CInt): Long = extern
  def okm_slice_update(src: Long, upd: Long, start: Ptr[CInt], stop: Ptr[CInt], n: CInt): Long = extern
  def okm_concatenate(as: Ptr[Long], n: CInt, axis: CInt): Long = extern
  def okm_zeros(shape: Ptr[CInt], n: CInt, dtype: CInt): Long = extern
  def okm_quantized_matmul(x: Long, w: Long, s: Long, b: Long, transpose: CInt, gs: CInt, bits: CInt): Long =
    extern
  def okm_dequantize(w: Long, s: Long, b: Long, gs: CInt, bits: CInt): Long = extern
  def okm_rms_norm(x: Long, w: Long, eps: CFloat): Long = extern
  def okm_rope(x: Long, dims: CInt, traditional: CInt, base: CFloat, scale: CFloat, offset: CInt): Long = extern
  def okm_sdpa(q: Long, k: Long, v: Long, scale: CFloat, causal: CInt): Long = extern
  def okm_load_open(path: CString): Long = extern
  def okm_load_next(file: Long, key: Ptr[CString], value: Ptr[Long]): CInt = extern
  def okm_load_close(file: Long): Unit = extern
  def okm_active_memory(): Long = extern
  def okm_peak_memory(): Long = extern

/** THE FLAT LAYER on Scala Native: the same functions as the JVM's `Okm`, called directly; int arrays copied to the
 * stack of the call */
private[mlx] object Okm:
  private inline def withInts[A](xs: Array[Int])(inline f: Ptr[CInt] => A): A =
    val p = stackalloc[CInt](xs.length.max(1))
    var i = 0
    while i < xs.length do
      p(i) = xs(i); i += 1
    f(p)

  def init(): Int = C.okm_init()
  def error(): String = fromCString(C.okm_error())
  def free(a: Long): Unit = C.okm_free(a)
  def ndim(a: Long): Int = C.okm_ndim(a)
  def dim(a: Long, i: Int): Int = C.okm_dim(a, i)
  def dtype(a: Long): Int = C.okm_dtype(a)
  def eval(a: Long): Int = C.okm_eval(a)
  def asyncEval(a: Long): Int = C.okm_async_eval(a)
  def evalAll(as: Array[Long]): Int = C.okm_eval_all(as.at(0), as.length)
  def itemI32(a: Long): Int = C.okm_item_i32(a)
  def itemF32(a: Long): Float = C.okm_item_f32(a)
  def fromI32(data: Array[Int], shape: Array[Int]): Long =
    withInts(shape)(s => C.okm_from_i32(data.at(0), s, shape.length))
  def scalarF32(v: Float): Long = C.okm_scalar_f32(v)
  def add(a: Long, b: Long): Long = C.okm_add(a, b)
  def multiply(a: Long, b: Long): Long = C.okm_multiply(a, b)
  def matmul(a: Long, b: Long): Long = C.okm_matmul(a, b)
  def sigmoid(a: Long): Long = C.okm_sigmoid(a)
  def astype(a: Long, dtype: Int): Long = C.okm_astype(a, dtype)
  def reshape(a: Long, shape: Array[Int]): Long = withInts(shape)(s => C.okm_reshape(a, s, shape.length))
  def transpose(a: Long, axes: Array[Int]): Long = withInts(axes)(s => C.okm_transpose(a, s, axes.length))
  def argmax(a: Long, axis: Int): Long = C.okm_argmax(a, axis)
  def softmax(a: Long, axis: Int): Long = C.okm_softmax(a, axis)
  def take(a: Long, idx: Long, axis: Int): Long = C.okm_take(a, idx, axis)
  def slice(a: Long, start: Array[Int], stop: Array[Int]): Long =
    withInts(start)(s => withInts(stop)(e => C.okm_slice(a, s, e, start.length)))
  def sliceUpdate(src: Long, upd: Long, start: Array[Int], stop: Array[Int]): Long =
    withInts(start)(s => withInts(stop)(e => C.okm_slice_update(src, upd, s, e, start.length)))
  def concatenate(as: Array[Long], axis: Int): Long =
    val p = stackalloc[Long](as.length.max(1))
    var i = 0
    while i < as.length do
      p(i) = as(i); i += 1
    C.okm_concatenate(p, as.length, axis)
  def zeros(shape: Array[Int], dtype: Int): Long = withInts(shape)(s => C.okm_zeros(s, shape.length, dtype))
  def quantizedMatmul(x: Long, w: Long, scales: Long, biases: Long, transpose: Boolean, groupSize: Int, bits: Int)
      : Long = C.okm_quantized_matmul(x, w, scales, biases, if transpose then 1 else 0, groupSize, bits)
  def dequantize(w: Long, scales: Long, biases: Long, groupSize: Int, bits: Int): Long =
    C.okm_dequantize(w, scales, biases, groupSize, bits)
  def rmsNorm(x: Long, w: Long, eps: Float): Long = C.okm_rms_norm(x, w, eps)
  def rope(x: Long, dims: Int, traditional: Boolean, base: Float, scale: Float, offset: Int): Long =
    C.okm_rope(x, dims, if traditional then 1 else 0, base, scale, offset)
  def sdpa(q: Long, k: Long, v: Long, scale: Float, causal: Boolean): Long =
    C.okm_sdpa(q, k, v, scale, if causal then 1 else 0)

  def loadOpen(path: String): Long = Zone.acquire(implicit z => C.okm_load_open(toCString(path)))
  def loadNext(file: Long): (String, Long) | Null =
    val key = stackalloc[CString]()
    val value = stackalloc[Long]()
    if C.okm_load_next(file, key, value) == 0 then null else (fromCString(!key), !value)
  def loadClose(file: Long): Unit = C.okm_load_close(file)
  def activeMemory(): Long = C.okm_active_memory()
  def peakMemory(): Long = C.okm_peak_memory()
