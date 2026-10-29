// okm: mlx-c flattened for foreign callers (the JVM's FFM, Scala Native). Every argument a pointer, an int or a float —
// no struct by value: an array is its handle (mlx_array's ctx), an op returns the result's handle, NULL on an error
// (okm_error() then says which). One GPU stream for every op.
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#include "mlx/c/mlx.h"

static mlx_stream S, CPU;
static char err[1024];

static void on_error(const char* msg, void* data) { (void)data; strncpy(err, msg, sizeof err - 1); }

int okm_init(void) {
  mlx_set_error_handler(on_error, NULL, NULL);
  S = mlx_default_gpu_stream_new();
  CPU = mlx_default_cpu_stream_new();
  return 0;
}
const char* okm_error(void) { return err; }

#define A(h) ((mlx_array){.ctx = (h)})
#define RET(call) \
  do { mlx_array r = mlx_array_new(); if (call) { mlx_array_free(r); return NULL; } return r.ctx; } while (0)
static mlx_optional_int oi(int v) { return (mlx_optional_int){.value = v, .has_value = v > 0}; }

// arrays
void okm_free(void* a) { mlx_array_free(A(a)); }
// a handle of its own on the same array (arrays are immutable: a snapshot is a handle)
void* okm_retain(void* a) { mlx_array r = mlx_array_new(); mlx_array_set(&r, A(a)); return r.ctx; }
int okm_ndim(void* a) { return (int)mlx_array_ndim(A(a)); }
int okm_dim(void* a, int i) { return mlx_array_shape(A(a))[i]; }
int okm_dtype(void* a) { return (int)mlx_array_dtype(A(a)); }
int okm_eval(void* a) {
  mlx_vector_array v = mlx_vector_array_new_data((mlx_array[]){A(a)}, 1);
  int r = mlx_eval(v); mlx_vector_array_free(v); return r;
}
int okm_async_eval(void* a) {
  mlx_vector_array v = mlx_vector_array_new_data((mlx_array[]){A(a)}, 1);
  int r = mlx_async_eval(v); mlx_vector_array_free(v); return r;
}
int okm_eval_all(void** as, int n) {
  mlx_vector_array v = mlx_vector_array_new();
  for (int i = 0; i < n; i++) mlx_vector_array_append_value(v, A(as[i]));
  int r = mlx_eval(v); mlx_vector_array_free(v); return r;
}
int okm_item_i32(void* a) { int32_t r = 0; mlx_array_item_int32(&r, A(a)); return r; }
float okm_item_f32(void* a) { float r = 0; mlx_array_item_float32(&r, A(a)); return r; }
void* okm_from_i32(const int* data, const int* shape, int ndim) {
  return mlx_array_new_data(data, shape, ndim, MLX_INT32).ctx;
}
void* okm_scalar_f32(float v) { return mlx_array_new_float(v).ctx; }

// ops
void* okm_add(void* a, void* b) { RET(mlx_add(&r, A(a), A(b), S)); }
void* okm_multiply(void* a, void* b) { RET(mlx_multiply(&r, A(a), A(b), S)); }
void* okm_matmul(void* a, void* b) { RET(mlx_matmul(&r, A(a), A(b), S)); }
void* okm_sigmoid(void* a) { RET(mlx_sigmoid(&r, A(a), S)); }
void* okm_astype(void* a, int dtype) { RET(mlx_astype(&r, A(a), (mlx_dtype)dtype, S)); }
void* okm_reshape(void* a, const int* shape, int n) { RET(mlx_reshape(&r, A(a), shape, n, S)); }
void* okm_transpose(void* a, const int* axes, int n) { RET(mlx_transpose_axes(&r, A(a), axes, n, S)); }
void* okm_argmax(void* a, int axis) { RET(mlx_argmax_axis(&r, A(a), axis, false, S)); }
void* okm_softmax(void* a, int axis) { RET(mlx_softmax_axis(&r, A(a), axis, true, S)); }
void* okm_take(void* a, void* idx, int axis) { RET(mlx_take_axis(&r, A(a), A(idx), axis, S)); }
void* okm_slice(void* a, const int* start, const int* stop, int n) {
  int strides[16]; for (int i = 0; i < n; i++) strides[i] = 1;
  RET(mlx_slice(&r, A(a), start, n, stop, n, strides, n, S));
}
void* okm_slice_update(void* src, void* upd, const int* start, const int* stop, int n) {
  int strides[16]; for (int i = 0; i < n; i++) strides[i] = 1;
  RET(mlx_slice_update(&r, A(src), A(upd), start, n, stop, n, strides, n, S));
}
void* okm_concatenate(void** as, int n, int axis) {
  mlx_vector_array v = mlx_vector_array_new();
  for (int i = 0; i < n; i++) mlx_vector_array_append_value(v, A(as[i]));
  mlx_array r = mlx_array_new(); int e = mlx_concatenate_axis(&r, v, axis, S); mlx_vector_array_free(v);
  if (e) { mlx_array_free(r); return NULL; } return r.ctx;
}
void* okm_zeros(const int* shape, int n, int dtype) { RET(mlx_zeros(&r, shape, n, (mlx_dtype)dtype, S)); }

void* okm_subtract(void* a, void* b) { RET(mlx_subtract(&r, A(a), A(b), S)); }
void* okm_divide(void* a, void* b) { RET(mlx_divide(&r, A(a), A(b), S)); }
void* okm_floor_divide(void* a, void* b) { RET(mlx_floor_divide(&r, A(a), A(b), S)); }
void* okm_exp(void* a) { RET(mlx_exp(&r, A(a), S)); }
void* okm_negative(void* a) { RET(mlx_negative(&r, A(a), S)); }
void* okm_logaddexp(void* a, void* b) { RET(mlx_logaddexp(&r, A(a), A(b), S)); }
void* okm_contiguous(void* a) { RET(mlx_contiguous(&r, A(a), false, S)); }
void* okm_sum(void* a, int axis, int keepdims) { RET(mlx_sum_axis(&r, A(a), axis, keepdims, S)); }
void* okm_expand_dims(void* a, int axis) { RET(mlx_expand_dims(&r, A(a), axis, S)); }
void* okm_squeeze(void* a, int axis) { RET(mlx_squeeze_axis(&r, A(a), axis, S)); }
void* okm_argpartition(void* a, int kth, int axis) { RET(mlx_argpartition_axis(&r, A(a), kth, axis, S)); }
void* okm_argsort(void* a, int axis) { RET(mlx_argsort_axis(&r, A(a), axis, S)); }
void* okm_take_along(void* a, void* idx, int axis) { RET(mlx_take_along_axis(&r, A(a), A(idx), axis, S)); }
void* okm_conv1d(void* x, void* w, int stride, int padding, int dilation, int groups) {
  RET(mlx_conv1d(&r, A(x), A(w), stride, padding, dilation, groups, S));
}
void* okm_scalar_i32(int v) { return mlx_array_new_int(v).ctx; }

// sampling
void* okm_cumsum(void* a, int axis) {
  RET(mlx_cumsum_axis(&r, A(a), axis, false, true, (mlx_optional_dtype){.has_value = false}, S));
}
void* okm_greater(void* a, void* b) { RET(mlx_greater(&r, A(a), A(b), S)); }
void* okm_where(void* c, void* a, void* b) { RET(mlx_where(&r, A(c), A(a), A(b), S)); }
void* okm_categorical(void* logits, int axis) { RET(mlx_random_categorical(&r, A(logits), axis, (mlx_array){0}, S)); }
int okm_random_seed(long seed) { return mlx_random_seed((uint64_t)seed); }

// quantized: `bits` and `group_size` of the affine mode
void* okm_quantized_matmul(void* x, void* w, void* scales, void* biases, int transpose, int group_size, int bits) {
  RET(mlx_quantized_matmul(&r, A(x), A(w), A(scales), A(biases), transpose, oi(group_size), oi(bits), "affine", S));
}
void* okm_dequantize(void* w, void* scales, void* biases, int group_size, int bits) {
  RET(mlx_dequantize(&r, A(w), A(scales), A(biases), oi(group_size), oi(bits), "affine", (mlx_array){0},
    (mlx_optional_dtype){.has_value = false}, S));
}

// rows of quantized experts by index: x [.., 1, D] · w[idx]ᵀ
void* okm_gather_qmm(void* x, void* w, void* scales, void* biases, void* rhs, int transpose, int group_size, int bits,
    int sorted) {
  RET(mlx_gather_qmm(&r, A(x), A(w), A(scales), A(biases), (mlx_array){0}, A(rhs), transpose, oi(group_size),
    oi(bits), "affine", (mlx_array){0}, sorted, S));
}

// fast
void* okm_rms_norm(void* x, void* w, float eps) { RET(mlx_fast_rms_norm(&r, A(x), A(w), eps, S)); }
void* okm_rope(void* x, int dims, int traditional, float base, float scale, int offset) {
  RET(mlx_fast_rope(&r, A(x), dims, traditional, (mlx_optional_float){.value = base, .has_value = true}, scale, offset,
    (mlx_array){0}, S));
}
// `causal` 1: the causal mask; else none
void* okm_sdpa(void* q, void* k, void* v, float scale, int causal) {
  RET(mlx_fast_scaled_dot_product_attention(&r, A(q), A(k), A(v), scale, causal ? "causal" : "", (mlx_array){0},
    (mlx_array){0}, false, S));
}

// a custom Metal kernel: made once, applied with its outputs' shapes and types, its grid, its template arguments
// (kind 0: an int, 1: a dtype, 2: a bool)
void* okm_kernel_new(const char* name, const char** inputs, int n_in, const char** outputs, int n_out,
    const char* source, const char* header) {
  mlx_vector_string in = mlx_vector_string_new(), out = mlx_vector_string_new();
  for (int i = 0; i < n_in; i++) mlx_vector_string_append_value(in, inputs[i]);
  for (int i = 0; i < n_out; i++) mlx_vector_string_append_value(out, outputs[i]);
  mlx_fast_metal_kernel k = mlx_fast_metal_kernel_new(name, in, out, source, header, true, false);
  mlx_vector_string_free(in); mlx_vector_string_free(out);
  return k.ctx;
}
int okm_kernel_apply(void* kernel, void** inputs, int n_in, const int* shapes, const int* ndims, const int* dtypes,
    int n_out, const int* grid, const int* group, const char** t_names, const int* t_kinds, const int* t_values,
    int n_t, void** outs) {
  mlx_fast_metal_kernel_config c = mlx_fast_metal_kernel_config_new();
  const int* sh = shapes;
  for (int i = 0; i < n_out; i++) {
    mlx_fast_metal_kernel_config_add_output_arg(c, sh, ndims[i], dtypes[i]);
    sh += ndims[i];
  }
  mlx_fast_metal_kernel_config_set_grid(c, grid[0], grid[1], grid[2]);
  mlx_fast_metal_kernel_config_set_thread_group(c, group[0], group[1], group[2]);
  for (int i = 0; i < n_t; i++) {
    if (t_kinds[i] == 0) mlx_fast_metal_kernel_config_add_template_arg_int(c, t_names[i], t_values[i]);
    else if (t_kinds[i] == 1) mlx_fast_metal_kernel_config_add_template_arg_dtype(c, t_names[i], t_values[i]);
    else mlx_fast_metal_kernel_config_add_template_arg_bool(c, t_names[i], t_values[i] != 0);
  }
  mlx_vector_array in = mlx_vector_array_new();
  for (int i = 0; i < n_in; i++) mlx_vector_array_append_value(in, A(inputs[i]));
  mlx_vector_array res = mlx_vector_array_new();
  mlx_fast_metal_kernel k = {.ctx = kernel};
  int e = mlx_fast_metal_kernel_apply(&res, k, in, c, S);
  if (!e) for (int i = 0; i < n_out; i++) {
    mlx_array a = mlx_array_new(); mlx_vector_array_get(&a, res, i); outs[i] = a.ctx;
  }
  mlx_vector_array_free(in); mlx_vector_array_free(res); mlx_fast_metal_kernel_config_free(c);
  return e;
}

// COMPILED (mlx_compile, shapeless): the elementwise functions mlx_lm compiles, the same ops in the same order — so the
// same numbers, one fused kernel each instead of two to five
static mlx_array in_at(const mlx_vector_array in, int i) {
  mlx_array a = mlx_array_new(); mlx_vector_array_get(&a, in, i); return a;
}
static mlx_array weak(float v, mlx_array like) {      // a Python float in an op: the array's own type
  mlx_array f = mlx_array_new_float(v), r = mlx_array_new();
  mlx_astype(&r, f, mlx_array_dtype(like), S); mlx_array_free(f); return r;
}
static int done(mlx_vector_array* res, mlx_array r) { mlx_vector_array_set_value(res, r); mlx_array_free(r); return 0; }

// silu(x) = x · sigmoid(x)
static int f_silu(mlx_vector_array* res, const mlx_vector_array in) {
  mlx_array x = in_at(in, 0), s = mlx_array_new(), r = mlx_array_new();
  mlx_sigmoid(&s, x, S); mlx_multiply(&r, x, s, S);
  mlx_array_free(x); mlx_array_free(s); return done(res, r);
}
// swiglu(gate, x) = silu(gate) · x
static int f_swiglu(mlx_vector_array* res, const mlx_vector_array in) {
  mlx_array g = in_at(in, 0), x = in_at(in, 1), s = mlx_array_new(), a = mlx_array_new(), r = mlx_array_new();
  mlx_sigmoid(&s, g, S); mlx_multiply(&a, g, s, S); mlx_multiply(&r, a, x, S);
  mlx_array_free(g); mlx_array_free(x); mlx_array_free(s); mlx_array_free(a); return done(res, r);
}
// precise_swiglu(h, gate, x) = (silu(gate as f32) · (x as f32)) as h's type
static int f_precise_swiglu(mlx_vector_array* res, const mlx_vector_array in) {
  mlx_array h = in_at(in, 0), g = in_at(in, 1), x = in_at(in, 2);
  mlx_array g32 = mlx_array_new(), x32 = mlx_array_new(), s = mlx_array_new(), a = mlx_array_new(),
    p = mlx_array_new(), r = mlx_array_new();
  mlx_astype(&g32, g, MLX_FLOAT32, S); mlx_sigmoid(&s, g32, S); mlx_multiply(&a, g32, s, S);
  mlx_astype(&x32, x, MLX_FLOAT32, S); mlx_multiply(&p, a, x32, S); mlx_astype(&r, p, mlx_array_dtype(h), S);
  mlx_array_free(h); mlx_array_free(g); mlx_array_free(x); mlx_array_free(g32); mlx_array_free(x32);
  mlx_array_free(s); mlx_array_free(a); mlx_array_free(p); return done(res, r);
}
// compute_g(A_log, a, dt_bias) = exp(−exp(A_log as f32) · softplus(a + dt_bias)), softplus(x) = logaddexp(x, 0)
static int f_compute_g(mlx_vector_array* res, const mlx_vector_array in) {
  mlx_array al = in_at(in, 0), a = in_at(in, 1), dt = in_at(in, 2);
  mlx_array al32 = mlx_array_new(), e = mlx_array_new(), ne = mlx_array_new(), sum = mlx_array_new(),
    sp = mlx_array_new(), m = mlx_array_new(), r = mlx_array_new();
  mlx_astype(&al32, al, MLX_FLOAT32, S); mlx_exp(&e, al32, S); mlx_negative(&ne, e, S);
  mlx_add(&sum, a, dt, S);
  mlx_array zero = weak(0.0f, sum); mlx_logaddexp(&sp, sum, zero, S);
  mlx_multiply(&m, ne, sp, S); mlx_exp(&r, m, S);
  mlx_array_free(al); mlx_array_free(a); mlx_array_free(dt); mlx_array_free(al32); mlx_array_free(e);
  mlx_array_free(ne); mlx_array_free(sum); mlx_array_free(zero); mlx_array_free(sp); mlx_array_free(m);
  return done(res, r);
}

static mlx_closure compiled(int (*f)(mlx_vector_array*, const mlx_vector_array)) {
  mlx_closure c = mlx_closure_new_func(f), r = mlx_closure_new();
  mlx_compile(&r, c, true); mlx_closure_free(c); return r;
}
static void* apply(mlx_closure* slot, int (*f)(mlx_vector_array*, const mlx_vector_array), void** args, int n) {
  if (!slot->ctx) *slot = compiled(f);
  mlx_vector_array in = mlx_vector_array_new();
  for (int i = 0; i < n; i++) mlx_vector_array_append_value(in, A(args[i]));
  mlx_vector_array out = mlx_vector_array_new();
  int e = mlx_closure_apply(&out, *slot, in);
  mlx_vector_array_free(in);
  if (e) { mlx_vector_array_free(out); return NULL; }
  mlx_array r = mlx_array_new(); mlx_vector_array_get(&r, out, 0); mlx_vector_array_free(out);
  return r.ctx;
}
static mlx_closure c_silu, c_swiglu, c_precise, c_g;
void* okm_silu(void* x) { return apply(&c_silu, f_silu, (void*[]){x}, 1); }
void* okm_swiglu(void* gate, void* x) { return apply(&c_swiglu, f_swiglu, (void*[]){gate, x}, 2); }
void* okm_precise_swiglu(void* h, void* gate, void* x) {
  return apply(&c_precise, f_precise_swiglu, (void*[]){h, gate, x}, 3);
}
void* okm_compute_g(void* a_log, void* a, void* dt_bias) {
  return apply(&c_g, f_compute_g, (void*[]){a_log, a, dt_bias}, 3);
}

// safetensors: opened, walked name by name, closed
typedef struct {
  mlx_map_string_to_array m; mlx_map_string_to_string meta; mlx_map_string_to_array_iterator it;
} okm_file;
void* okm_load_open(const char* path) {
  okm_file* f = malloc(sizeof *f);
  f->m = mlx_map_string_to_array_new(); f->meta = mlx_map_string_to_string_new();
  if (mlx_load_safetensors(&f->m, &f->meta, path, CPU)) { free(f); return NULL; }
  f->it = mlx_map_string_to_array_iterator_new(f->m);
  return f;
}
// the next tensor: its name into *key, a handle of its own into *value; 0 at the end
int okm_load_next(void* file, const char** key, void** value) {
  okm_file* f = file; mlx_array v = mlx_array_new();
  if (mlx_map_string_to_array_iterator_next(key, &v, f->it)) { mlx_array_free(v); return 0; }
  mlx_array own = mlx_array_new(); mlx_array_set(&own, v); mlx_array_free(v);
  *value = own.ctx; return 1;
}
// arrays saved, by name, with string metadata
int okm_save_safetensors(const char* path, const char** names, void** arrays, int n, const char** keys,
    const char** values, int m) {
  mlx_map_string_to_array map = mlx_map_string_to_array_new();
  mlx_map_string_to_string meta = mlx_map_string_to_string_new();
  for (int i = 0; i < n; i++) mlx_map_string_to_array_insert(map, names[i], A(arrays[i]));
  for (int i = 0; i < m; i++) mlx_map_string_to_string_insert(meta, keys[i], values[i]);
  int e = mlx_save_safetensors(path, map, meta);
  mlx_map_string_to_array_free(map); mlx_map_string_to_string_free(meta);
  return e;
}
void okm_load_close(void* file) {
  okm_file* f = file; mlx_map_string_to_array_iterator_free(f->it); mlx_map_string_to_array_free(f->m);
  mlx_map_string_to_string_free(f->meta); free(f);
}

// the graph that computes an array, as Graphviz text, to a file: what a step is made of
int okm_export_dot(void* a, const char* path) {
  FILE* f = fopen(path, "w"); if (!f) return 1;
  mlx_node_namer namer = mlx_node_namer_new();
  mlx_vector_array v = mlx_vector_array_new_data((mlx_array[]){A(a)}, 1);
  int e = mlx_export_to_dot(f, namer, v);
  mlx_vector_array_free(v); mlx_node_namer_free(namer); fclose(f); return e;
}

// memory
long okm_active_memory(void) { size_t r = 0; mlx_get_active_memory(&r); return (long)r; }
long okm_peak_memory(void) { size_t r = 0; mlx_get_peak_memory(&r); return (long)r; }
// the buffers MLX keeps for reuse, given back (after a model is closed: its memory free for the next)
int okm_clear_cache(void) { return mlx_clear_cache(); }

// A TENSOR OF A FILE, MAPPED: bytes [offset, offset + nbytes) of the file, an array of `dtype` and `shape` over the
// file's own pages — no copy: Metal wraps page-aligned memory as it is (newBuffer bytesNoCopy), the tensor a view
// into it (slice, view, reshape; evaluated, they share the buffer). The pages read when first touched, and theirs
// to give back when memory is short (they are the file's, clean); unmapped when the array is freed
typedef struct { void* p; size_t n; } okm_map;
static void okm_unmap(void* payload) { okm_map* m = payload; munmap(m->p, m->n); free(m); }
void* okm_map_tensor(const char* path, long offset, long nbytes, int dtype, const int* shape, int ndim) {
  long page = getpagesize();
  long start = offset / page * page, end = (offset + nbytes + page - 1) / page * page;
  int fd = open(path, O_RDONLY);
  if (fd < 0) { snprintf(err, sizeof err, "%s: %s", path, strerror(errno)); return NULL; }
  void* p = mmap(NULL, (size_t)(end - start), PROT_READ, MAP_SHARED, fd, start);  // read-only: never a page of its own
  close(fd);
  if (p == MAP_FAILED) { snprintf(err, sizeof err, "mmap %s: %s", path, strerror(errno)); return NULL; }
  okm_map* m = malloc(sizeof *m); m->p = p; m->n = (size_t)(end - start);
  int len = (int)(end - start), b = (int)(offset - start), e = b + (int)nbytes, one = 1;
  mlx_array bytes = mlx_array_new_data_managed_payload(p, &len, 1, MLX_UINT8, m, okm_unmap);
  mlx_array s = mlx_array_new(), v = mlx_array_new(), r = mlx_array_new();
  int bad = mlx_slice(&s, bytes, &b, 1, &e, 1, &one, 1, S) || mlx_view(&v, s, (mlx_dtype)dtype, S) ||
    mlx_reshape(&r, v, shape, ndim, S);
  mlx_array_free(bytes); mlx_array_free(s); mlx_array_free(v);
  if (bad) { mlx_array_free(r); return NULL; }
  if (getenv("OKM_MAP_DEBUG")) {
    mlx_array_eval(r);
    char* d = (char*)mlx_array_data_uint8(r);
    if (d < (char*)p || d >= (char*)p + (end - start))
      fprintf(stderr, "okm_map_tensor COPIED dtype %d nbytes %ld offset %ld (mod 16: %ld) ndim %d\n", dtype, nbytes,
        offset, offset % 16, ndim);
  }
  return r.ctx;
}
