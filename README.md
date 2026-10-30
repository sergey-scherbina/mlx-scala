# mlx-scala

Apple's [MLX](https://github.com/ml-explore/mlx) from Scala 3, on two platforms with one model code: the JVM through
FFM (Java 22+), and Scala Native. A Qwen3 runs on it — a byte-level BPE tokenizer, the model, its KV cache, greedy
generation — as fast as rozum's Rust runtime on the same weights.

```
native-lib/build.sh                                   # mlx-c (MLX v0.32.4) and okm, once
sbt test                                              # the JVM and Scala Native
```

## Get it

```scala
// build.sbt
resolvers ++= Seq("mlx-scala" at "https://sergey-scherbina.github.io/mlx-scala/maven",
  "okay-simple" at "https://okay-effects.github.io/okay-simple/maven")       // its JSON
libraryDependencies += "io.github.sergey-scherbina" %%% "mlx-scala-llm" % "0.1.0"  // or "mlx-scala": the MLX API alone
```

The native libraries — `libokm` (the flat layer), `libmlxc`, `libmlx`, `mlx.metallib` — from the
[release](https://github.com/sergey-scherbina/mlx-scala/releases/tag/v0.1.0) (Apple Silicon, built on macOS 27), or
built: `native-lib/build.sh` (mlx-c v0.8.0, MLX v0.32.4). Then:

- the JVM (Java 22+): `--enable-native-access=ALL-UNNAMED` and `MLX_SCALA_LIB=<their directory>` (or
  `-Dokm.lib=<…>/libokm.dylib`);
- Scala Native: `nativeConfig ~= (_.withEmbedResources(true).withLinkingOptions(Seq("-L<dir>", "-lmlxc",
  "-Wl,-rpath,<dir>")))`.

## How it is built

| module | what | depends on |
|---|---|---|
| `native-lib/okm.c` | mlx-c flattened: every argument a pointer, an int or a float — no struct by value crosses | mlx-c |
| `mlx` | the MLX API: `Arr`, `Scope`, ops, fast kernels, quantized matmul, safetensors. `Okm` per platform: FFM `MethodHandle`s on the JVM, `@extern` on Scala Native — the rest shared | `okm` alone |
| `llm` | `Tokenizer` (byte-level BPE, HuggingFace's ids), `Nfc`, `ChatTemplate` (the model's Jinja, rendered as transformers does), `Chat`, `Qwen3`, `Qwen35` (Qwen3.5, Qwen3.6: Gated DeltaNet, MoE), `KvCache`, `Session`, `Sampling`, `PrefixCache`, `PrefixStore` | `mlx` alone |

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

Decode at mlx_lm's speed: Qwen3.6-35B-A3B 117 tok/s against mlx_lm's 118 (the next step built and started before
the last token is read; the elementwise functions mlx_lm compiles — silu, swiglu, precise_swiglu, compute_g —
compiled by MLX the same way, in okm.c). Tokens drawn as `Sampling` says — greedy, or temperature, top_k, top_p, a
seed; a model's own from its generation_config.json — on the GPU.

A prompt's prefixes are kept between generations (`PrefixCache`): a model's state snapshot at the marks `Chat.marks`
gives — where messages begin; a hybrid model's recurrent state cannot be cut shorter, so a prefix is reused whole —
and a prompt beginning with one is computed from there; a prompt's first mark (the system prompt and the tools)
written to disk too (`PrefixStore`, safetensors, 4 GB by default), so a restart does not compute it again. Claude Code's next turn: 13.6k tokens' prefill 16 s → 0.2–0.3 s.

JSON is okay-simple's value module alone, `okay-effects-json-value` (no dependency) — the same JSON as okay-llm's.

On it, with effects: [okay-llm](https://github.com/okay-effects/okay-llm) (the model as an effect, a gateway for
OpenAI's and Anthropic's APIs) and its MLX module [okay-mlx](https://github.com/okay-effects/okay-mlx).

## Weights mapped

`Model.load(dir, mapped = true)` (or MLX_SCALA_MMAP=1) maps the .safetensors instead of reading them: each tensor an
array over the file's own pages — Metal wraps page-aligned memory as it is, the tensor a view into it — so nothing
is copied at load, a page is read when first touched, and it is the file's (clean) for the OS to give back when
memory is short. Qwen3.6-35B-A3B, M4 Max, 3 alternating rounds: loaded in 0.15 s instead of 2.7–3.6 s, the process's
footprint 0.8 GB instead of 19 GB, the first token sooner (~4.4 s against ~5.7 s, the pages read by the first
prefill), the decode the same (118.3 against 119.9 tok/s, the medians; within the host's noise) — and greedy tokens
still mlx_lm's.

A tensor maps only where its bytes sit aligned for its type, and MLX's writer does not pad the header: in
mlx-community's files every tensor sits off by the header's length (a uint32 off a multiple of 4), and MLX copies
what it cannot use in place. Such a file is aligned — the same file, its header padded with spaces, as the format
allows: in place (`inPlace = true`, MLX_SCALA_ALIGN=in-place: written beside it and moved over it, HuggingFace's blob
itself, read as before by any reader), else as a copy written once under ~/.cache/mlx-scala/aligned
(MLX_SCALA_ALIGNED).

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
