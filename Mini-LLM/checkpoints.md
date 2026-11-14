# Checkpoints

Every checkpoint stores its weights, its configuration and metadata (`step`,
`total_steps`, `val_loss`, plus `bits_per_byte` when the run had a `data/meta.json` to
convert the loss). Files in `models/`:

| File                                             | Content                                                                                       |
| ------------------------------------------------ | --------------------------------------------------------------------------------------------- |
| `models/<lineage>/tok.json`                      | The lineage tokenizer: `32k` = 32,768 tokens (40M/150M), `16k` = 16,384 (34M/70M)             |
| `models/16k/mini-llm-base-<size>-<steps>steps.pt` or `models/32k/<size>/mini-llm-base-<size>-<steps>steps.pt` or `models/digit32k/<size>/petal-<version>-<size>-base-<steps>steps.pt` | Base model, written once pretraining is complete (`16k` 34M-12000steps / 70M-22000steps, `32k` 40M-35096steps / 150M-34332steps). New runs use the Petal 1 naming (`petal-<version>-<size>-base-<steps>steps.pt`); existing `mini-llm-*` files remain loadable |
| `models/<lineage>/<size>/petal-<version>-<size>-base-<base+steps>steps.pt` (new) or `mini-llm-base-<size>-<base+steps>steps.pt` (legacy) | Same base line after the format stage (`uv run midtrain 600`): 40M-35696steps / 150M-53975steps, i.e. 600 steps above the pretrained count |
| `models/16k/mini-llm-base-<size>-<steps>steps.state.pt` or `models/32k/<size>/mini-llm-base-<size>-<steps>steps.state.pt` or `petal-<version>-<size>-base-<steps>steps.state.pt` | Resume state from `pretrain` (includes optimizer), updated every 500 steps and deleted when the run completes |
| `models/32k/150M/teacher-150M-base.pt` | 150M base converted from `GautierP/MiniLLM-150M` for distillation (`uv run teacher`); not a deliverable, never selected automatically |
| `models/16k/mini-llm-assistant-<size>.pt` or `models/32k/<size>/mini-llm-assistant-<size>.pt` or `petal-<version>-<size>.pt` | Best assistant so far, by validation loss (`16k` assistant-34M/70M, `32k` assistant-40M/150M). New runs use the plain release file, e.g. `petal-1-150M.pt` |
| `models/16k/mini-llm-assistant-<size>.state.pt` or `models/32k/<size>/mini-llm-assistant-<size>.state.pt` or `petal-<version>-<size>.state.pt` | Resume state of the SFT run, deleted when the run completes |
| `data/train.bin`, `data/val.bin`, `data/val_docs.json`, `data/meta.json` | Training data, already converted to tokens, plus validation document hashes and vocab/tokenizer provenance |

## Automatic selection

How checkpoints are chosen when no path is given:

- `sft` picks the base model with the highest step count in its file name, then the
  most recent file. A quick 300-step test therefore never takes priority over a
  6000-step model. `.state.pt` files are never selected automatically.
- The format-stage output (`petal-<version>-<size>-base-<base+600>steps.pt`, or legacy
  `mini-llm-base-<size>-<base+600>steps.pt`) therefore wins
  over the pretrained checkpoint it was built from, which is the point: `cloud/run.sh
  sft` trains on it as soon as it exists. Pass the pretrained path explicitly to `sft`
  to skip the stage.
- `chat` does the same among the assistant files.
- Checkpoint selection looks for the Petal 1 patterns (`petal-*-base-*.pt`,
  `petal-*-*M.pt`) first and falls back to the legacy `mini-llm-*` names,
  so the models already on disk keep working.
- `midtrain` only looks under `models/digit32k/` for its default `--init`: the stage
  needs the Petal 1 lineage, so a classic base is never picked by accident.
- `midtrain` writes the final checkpoint only, no resume state: 600 steps are cheap to
  rerun, and the pretrained base it starts from is untouched.
- A path can always be passed explicitly, either absolute or relative to `models/`.
- The converted teacher (`teacher-*.pt`) matches neither the Petal 1 base pattern nor
  the legacy one,
  so `sft` and `chat` never pick it; it is only used through `pretrain --teacher`.

The tokenizer is read next to the checkpoint, or in the lineage folder one level up:
each lineage keeps a single `tok.json` (`models/16k/` for 34M/70M, `models/32k/` for
the classic 40M/150M, `models/digit32k/` for the Petal 1 models), and each size stores its
checkpoints in its own subfolder (`models/32k/40M/`, `models/digit32k/150M/`). `prep`
derives the lineage folder from the preset vocab (16,384 → `16k`, 32,768 → `32k`,
unless the size asks for `digit32k`); `pretrain` adds the size tag.

## Compatibility rules

- Checkpoints store the configuration as a plain dictionary. Adding a field with a
  default value to `Cfg` keeps old checkpoints loadable; renaming or removing a field
  does not.
- Checkpoints are loaded with `weights_only=True`. Only legacy checkpoints that
  pickled the `Cfg` object itself fall back to unrestricted loading. Never load a
  `.pt` file from an untrusted source.

## See also

- [Training](training.md) — how these files are produced and resumed.
- [Export to Hugging Face](export-hf.md) — `.pt` in, deployable directory out.
- [Mini-Eval backend](../Mini-Eval/backend.md) — how a checkpoint is evaluated
  directly, with no export step.
