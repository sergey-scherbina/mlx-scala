# BACKLOG — mlx-scala

Reference below: FreeToken (github.com/FlashML-org/FreeToken, Apache-2.0, read at 628b88f; Python/CUDA).

Done 2026-11-04: MTP drafts (`Mtp`, `Session(drafts)`, 35B-A3B k=2 93 → 121 tok/s); the state kept after an answer
and after `<tool_call>` (`Session(anchors)`, `Chat.anchors`).

- **MTP: the draft layer's KV in snapshots**: a state restored from `PrefixCache` starts the draft layer empty (its
  positions 0.., the model's further on). Drafts stay correct (checked), fewer are taken on a long cached prefix —
  measure that first (an agent's turn: most of the prompt cached), then keep the draft's cache in `Snapshot`
  (and `PrefixStore`'s file: a tensor more).
- **MTP: overlap**: a round waits for its picks before the next is built (the drafts depend on them); plain decode
  hides the graph's building behind the GPU. A round's cost now: drafts 1.4 ms + a pass of k + 1 (35B-A3B: 13.4 ms
  for 2 vs 11.2 for 1) + the wait. Ideas: build the next round's pass for "all taken" while waiting.
- **MTP for Qwen3.8-27B / Qwen3.6-27B** (dense: FreeToken's 2.7× at k=4 on CUDA): `mtpFetch
  mlx-community/Qwen3.8-27B-4bit Qwen/Qwen3.8-27B` (~810 MB of bf16), then `draftBench`.
- **MTP with hot experts** (`expertSlots`): a pass of k + 1 wants the union of their experts — not tried; okay-mlx
  turns drafts off on hot experts as it does guesses.
- **Snapshots every 64 tokens of prefill** (FreeToken's ×64 track): less needed with our prefill speed; measure the
  re-prefill an agent's turn still pays first.
- **MTP trained for Scala: tried 2026-11-04, not worth serving** (`MtpTrain`: LoRA on the draft layer, rank 16,
  toward the model's own picks; `mtpTrainBench`, `mtpSelfBench`, Qwen3.5-4B). On human code (okay-* files, held-out
  symdex + okay-datalog): first-draft agreement 66.8 → 72.0% in 1500 steps (prose 63.5 → 66.8, not lost) — but a
  real decode moved 2.29 → 2.31 tokens a pass. On the model's OWN answers (the distribution a decode sees; 500
  continuations of training files, second draft in the loss): agreement is already 85.1% untrained, 86.4% trained
  (overfits after ~750 steps), decode 2.44 → 2.49 tokens a pass (+2%). The draft layer is near its ceiling on the
  model's own Scala; more tokens a pass need a deeper draft (several layers / EAGLE-style), not tuning this one.
- **Code search, next** (2026-11-05, branch embeddings): more pairs (call site ↔ declaration, commit message ↔
  changed member — symdex and git; now only doc comments, 597), the adapter trained on all projects for use; an
  index per project (vectors cached by file hash, on the CPU in the background); then rozum's `rag.search` and
  symdex's search on it (both lexical/semantic today — measure them on the same pairs first).
