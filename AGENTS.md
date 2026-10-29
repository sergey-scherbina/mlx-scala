# mlx-scala — notes for agents

Apple's MLX from Scala 3 on two platforms, one model code: the JVM through FFM (Java 22+), Scala Native (0.5).

- `native-lib/`: `build.sh` builds mlx-c (v0.8.0, MLX v0.32.4) as shared libraries into `native-lib/out` (not in
  git) and compiles `okm.c` — mlx-c FLATTENED: every argument a pointer, an int or a float, an array its handle
  (mlx_array's ctx), an op returning the result's handle (NULL on an error, `okm_error()` says which). No struct by
  value crosses the boundary, so FFM and Scala Native bind the same functions the same simple way.
- `mlx/` (crossProject Full): `Okm` per platform — jvm: FFM `MethodHandle`s, a handle passed as `long` (the same
  register as a pointer), int arrays through a buffer of the thread's; native: `@extern` with handles as `Long`,
  arrays on the stack. `shared/Mlx.scala`: `Arr`, `Scope`, the ops. Depends on nothing.
- `llm/`: JSON is okay-simple's value module, `okay-effects-json-value` (no dependency; `j("key")`, `.string`,
  `.list`, `Json.fromText`), the one JSON of mlx-scala, okay-llm and okay-mlx. `Tokenizer` (byte-level BPE; Qwen's
  pre-tokenizer regex written out by hand — its `(?!\S)` has no RE2 form; NFC when tokenizer.json says), `Nfc`
  (Unicode's data in the resource mlx/llm/nfc.txt, by `scripts/gen-nfc.py`; Native apps embed resources),
  `ChatTemplate` (a Jinja interpreter: transformers' jinja2 settings, trim_blocks and lstrip_blocks; checked against
  jinja2 in `resources/templates/cases.json` — regenerate with jinja2 when adding a template), `Chat` (tokenizer +
  template + stop tokens), `Qwen3`, `KvCache`, `Generate`. Depends on `mlx` alone.
- No effects here: the model as an effect, the gateway, the app are okay-llm and okay-mlx, repositories of their
  own. The one thing shared with them is the JSON value.

- `PrefixCache`: `Model.State.snapshot()` (handles of their own on the same arrays: `okm_retain`; MLX copies on
  the next update) at `Chat.marks` (the second and the last `<|im_start|>`); `Session(..., cache, marks)` restores the
  longest kept prefix of the prompt. Snapshots only at marks: a linear layer's state is not cut back. `PrefixStore`:
  the first mark's snapshot written as safetensors (tokens in the metadata; the index is the headers), 4 GB budget —
  NOT every turn's (a 60k conversation wrote 31 GB in one run).
- `Sampling` (greedy / temperature, top_k by argpartition, top_p by cumsum, categorical, seed); `Sampling.of(dir)`
  reads generation_config.json. Decode: `Session.next` builds and starts the next step BEFORE reading the current
  token (else no overlap: 13%); okm.c compiles mlx_lm's compiled functions (silu, swiglu, precise_swiglu,
  compute_g) with `mlx_compile` — same ops, same order, same numbers. Probes: decodeProbe (also exports a step's
  graph: okm_export_dot), prefillCurve, gdnAb.

Rules:
- A HANDLE HELD KEEPS ITS BUFFER: MLX frees an intermediate once the graph is done with it only if no handle holds
  it. Each layer runs in a `Scope` of its own, its output alone moved out (with a pass's intermediates held, a
  4096-token prompt peaked at 48.7 GB instead of 3.6).
- MLX reads safetensors on the CPU stream (`okm_load_open`); the weights are evaluated at load.
- Tests that need the model skip when mlx-community/Qwen3-4B-4bit is not in the HuggingFace cache
  (`rozum models pull mlx-community:Qwen3-4B-4bit`, or any HF download).

Build: `native-lib/build.sh` once; `sbt test` (JVM and Native); `sbt publishLocal` for okay-mlx. Scala Native has
no `java.time`, no `java.text.Normalizer`: write it, as `Nfc` and `ChatTemplate.strftime` do. Tests embed resources
on Native (`withEmbedResources`). Lines of code at most 120 characters.
