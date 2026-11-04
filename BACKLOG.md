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
