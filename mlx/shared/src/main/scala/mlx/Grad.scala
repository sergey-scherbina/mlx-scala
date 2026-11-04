package mlx

/** the functions autograd calls back, by id: given their inputs' handles (borrowed), their outputs' handles (each a
 * new one, given up to the caller) */
private[mlx] object Grad:
  private val fns = java.util.concurrent.ConcurrentHashMap[java.lang.Long, Array[Long] => Array[Long]]()
  private val next = java.util.concurrent.atomic.AtomicLong()
  def register(f: Array[Long] => Array[Long]): Long =
    val id = next.incrementAndGet()
    fns.put(id, f)
    id
  def forget(id: Long): Unit = fns.remove(id): Unit
  def call(id: Long, ins: Array[Long]): Array[Long] = fns.get(id)(ins)
