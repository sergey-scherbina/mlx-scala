# mlx-scala — notes for agents

Apple's MLX from Scala 3 on two platforms, one model code: the JVM through FFM (Java 22+), Scala Native (0.5).

- `native-lib/`: `build.sh` builds mlx-c (v0.8.0, MLX v0.32.4) as shared libraries into `native-lib/out` (not in
  git) and compiles `okm.c` — mlx-c FLATTENED: every argument a pointer, an int or a float, an array its handle
  (mlx_array's ctx), an op returning the result's handle (NULL on an error, `okm_error()` says which). No struct by
  value crosses the boundary, so FFM and Scala Native bind the same functions the same simple way.
- `mlx/` (crossProject Full): `Okm` per platform — jvm: FFM `MethodHandle`s, a handle passed as `long` (the same
  register as a pointer), int arrays through a buffer of the thread's; native: `@extern` with handles as `Long`,
  arrays on the stack. `shared/Mlx.scala`: `Arr`, `Scope`, the ops. Depends on nothing.
- `llm/`: `Json` (a reader), `Tokenizer` (byte-level BPE; Qwen's pre-tokenizer regex written out by hand — its
  `(?!\S)` has no RE2 form), `Qwen3`, `KvCache`, `Generate` (greedy, the next step started before the last token is
  read). Depends on `mlx` alone.
- `app/`: the world through okay-scala's effects — `Llm` (mlx / fake), `Files` (disk / of), `Console` (stdout /
  collect), std `Clock`, `Log`; `Programs.ask`, `Programs.bench`; `Main` wires the handlers.

Rules:
- A HANDLE HELD KEEPS ITS BUFFER: MLX frees an intermediate once the graph is done with it only if no handle holds
  it. Each layer runs in a `Scope` of its own, its output alone moved out (with a pass's intermediates held, a
  4096-token prompt peaked at 48.7 GB instead of 3.6).
- MLX reads safetensors on the CPU stream (`okm_load_open`); the weights are evaluated at load.
- Tests that need the model skip when mlx-community/Qwen3-4B-4bit is not in the HuggingFace cache
  (`rozum models pull mlx-community:Qwen3-4B-4bit`, or any HF download).

Build: `native-lib/build.sh` once; `sbt test` (JVM and Native); `sbt "appJVM/run ask <question>"`,
`sbt "appNative/run bench 3"`. Lines of code at most 120 characters.
