# mlx-scala

Apple's [MLX](https://github.com/ml-explore/mlx) from Scala 3, on two platforms with one model code: the JVM through
FFM (Java 22+), and Scala Native. A Qwen3 runs on it — a byte-level BPE tokenizer, the model, its KV cache, greedy
generation — as fast as rozum's Rust runtime on the same weights.

```
native-lib/build.sh                                   # mlx-c (MLX v0.32.4) and okm, once
sbt test                                              # the JVM and Scala Native
```

## How it is built

| module | what | depends on |
|---|---|---|
| `native-lib/okm.c` | mlx-c flattened: every argument a pointer, an int or a float — no struct by value crosses | mlx-c |
| `mlx` | the MLX API: `Arr`, `Scope`, ops, fast kernels, quantized matmul, safetensors. `Okm` per platform: FFM `MethodHandle`s on the JVM, `@extern` on Scala Native — the rest shared | `okm` alone |
| `llm` | `Tokenizer` (byte-level BPE, HuggingFace's ids), `Nfc`, `ChatTemplate` (the model's Jinja, rendered as transformers does), `Chat`, `Qwen3`, `Qwen35` (Qwen3.5, Qwen3.6: Gated DeltaNet, MoE), `KvCache`, `Session` | `mlx` alone |

```scala
val dir = Models.snapshot("mlx-community/Qwen3-4B-4bit").get
val chat = Chat.load(dir)                     // tokenizer.json + the chat template
val model = Qwen3.load(dir)
val prompt = chat.ask("What is the capital of France?", Map("enable_thinking" -> false))
Generate.greedy(model, prompt, 64, chat.stops)(id => print(chat.tokenizer.decode(Seq(id))))
```

Checked against the Python reference on the same files: the tokenizer's ids against HuggingFace `tokenizers`; NFC
against `unicodedata` (Unicode 16; Scala Native has no `java.text.Normalizer`, so Unicode's data is a resource,
mlx/llm/nfc.txt by `scripts/gen-nfc.py` — on Native the app builds with `withEmbedResources(true)`); Qwen3's and
Qwen3.6's chat templates against jinja2 as transformers configures it — system turns, thinking, tools, tool calls and
responses, content parts.

On it, with effects: [okay-llm](https://github.com/okay-effects/okay-llm) (the model as an effect, a gateway for
OpenAI's and Anthropic's APIs) and its MLX module [okay-mlx](https://github.com/okay-effects/okay-mlx).

## Measured

Qwen3-4B (mlx-community, 4-bit), greedy, 256 tokens out, a different prompt each run; tokens a second, the median
of 3 runs; M4 Max, 2026-10-29. mlx-scala in process (okay-mlx's `bench`); rozum through its HTTP API, streamed.

| prompt | mlx-scala, JVM | mlx-scala, Native | rozum (Rust) |
|---|---|---|---|
| 143 tokens: prefill / decode | 850–1158 / 109–117 | 750–925 / 103–110 | 524–615 / 100–109 |
| 4.7k tokens | 850–1027 / 90–91 | 840–921 / 90–91 | 785–827 / 86–90 |
| 9.2k tokens | 780–844 / 73–77 | 779–816 / 72–76 | 767–781 / 64–67 |

The GPU's kernels are MLX's own on all three: the platforms differ in what the host does per token — about 1000
calls a token build the graph. On FFM a call costs ~130 ns (most of it MLX's own node); the next step is built
while the GPU runs the last one, so the host's time hides behind it.
