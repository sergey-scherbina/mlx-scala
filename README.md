# mlx-scala

Apple's [MLX](https://github.com/ml-explore/mlx) from Scala 3, on two platforms with one model code: the JVM through
FFM (Java 22+), and Scala Native. A Qwen3 runs on it — a byte-level BPE tokenizer, the model, its KV cache, greedy
generation — as fast as rozum's Rust runtime on the same weights.

```
native-lib/build.sh                                   # mlx-c (MLX v0.32.4) and okm, once
sbt "appJVM/run ask What is a KV cache?"              # the JVM
sbt "appNative/run bench 3"                           # Scala Native
```

## How it is built

| module | what | depends on |
|---|---|---|
| `native-lib/okm.c` | mlx-c flattened: every argument a pointer, an int or a float — no struct by value crosses | mlx-c |
| `mlx` | the MLX API: `Arr`, `Scope`, ops, fast kernels, quantized matmul, safetensors. `Okm` per platform: FFM `MethodHandle`s on the JVM, `@extern` on Scala Native — the rest shared | `okm` alone |
| `llm` | `Tokenizer` (HuggingFace's ids, checked), `Qwen3`, `KvCache`, `Generate` | `mlx` alone |
| `app` | the world through effects (okay-scala): `Llm`, `Files`, `Console`, `Clock`, `Log`; `ask`, `bench` | `llm`, okay-scala |

Where nothing outside MLX is involved — arrays, the model, the tokenizer — the code is plain Scala over the MLX API.
Where the app meets the world — which model answers, the files, the console, the clock — it is an effect, and a
handler gives it its meaning: `Llm.mlx` (Qwen3 on the GPU) or `Llm.fake`, `Files.disk` or `Files.of(map)`,
`Console.stdout` or `Console.collect`. The app's programs are tested without a GPU, a model or a disk.

```scala
def ask[R[_]: Has[Llm] : Has[Console]](question: String, maxTokens: Int): Generation ! R =
  for
    g <- Llm.complete(question, maxTokens)
    _ <- Console.line(g.text)
  yield g
```

## Measured

Qwen3-4B (mlx-community, 4-bit), greedy, 256 tokens out, a different prompt each run; tokens a second, the median
of 3 runs; M4 Max, 2026-10-29. mlx-scala in process; rozum through its HTTP API, streamed.

| prompt | mlx-scala, JVM | mlx-scala, Native | rozum (Rust) |
|---|---|---|---|
| 143 tokens: prefill / decode | 850–1158 / 109–117 | 750–925 / 103–110 | 524–615 / 100–109 |
| 4.7k tokens | 850–1027 / 90–91 | 840–921 / 90–91 | 785–827 / 86–90 |
| 9.2k tokens | 780–844 / 73–77 | 779–816 / 72–76 | 767–781 / 64–67 |

The GPU's kernels are MLX's own on all three: the platforms differ in what the host does per token — about 1000
calls a token build the graph. On FFM a call costs ~130 ns (most of it MLX's own node); the next step is built
while the GPU runs the last one, so the host's time hides behind it.
