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
