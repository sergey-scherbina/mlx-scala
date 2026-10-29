// okm: mlx-c flattened for foreign callers (the JVM's FFM, Scala Native). Every argument a pointer, an int or a float —
// no struct by value: an array is its handle (mlx_array's ctx), an op returns the result's handle, NULL on an error
// (okm_error() then says which). One GPU stream for every op.
#include <stdlib.h>
#include <string.h>
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

// quantized: `bits` and `group_size` of the affine mode
void* okm_quantized_matmul(void* x, void* w, void* scales, void* biases, int transpose, int group_size, int bits) {
  RET(mlx_quantized_matmul(&r, A(x), A(w), A(scales), A(biases), transpose, oi(group_size), oi(bits), "affine", S));
}
void* okm_dequantize(void* w, void* scales, void* biases, int group_size, int bits) {
  RET(mlx_dequantize(&r, A(w), A(scales), A(biases), oi(group_size), oi(bits), "affine", (mlx_array){0},
    (mlx_optional_dtype){.has_value = false}, S));
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
void okm_load_close(void* file) {
  okm_file* f = file; mlx_map_string_to_array_iterator_free(f->it); mlx_map_string_to_array_free(f->m);
  mlx_map_string_to_string_free(f->meta); free(f);
}

// memory
long okm_active_memory(void) { size_t r = 0; mlx_get_active_memory(&r); return (long)r; }
long okm_peak_memory(void) { size_t r = 0; mlx_get_peak_memory(&r); return (long)r; }
