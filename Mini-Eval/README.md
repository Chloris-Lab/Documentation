# Mini-Eval

Evaluation of [Mini-LLM](../Mini-LLM/README.md) models against reference models
(Pythia, ...) with
[lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness). The
repository is separate from the training repository on purpose: it only asks the
models for probabilities, and `lm-eval` computes the scores.

Source repository: [github.com/Chloris-Lab/Mini-Eval](https://github.com/Chloris-Lab/Mini-Eval).

**Server only.** Protocol v3 numbers are produced on the Linux GPU server (CUDA),
from a committed tree. The package refuses to import on macOS: there is no local
path to results, and `uv run pytest` also runs on the server.

## Pages

| Page                          | Content                                                          |
| ----------------------------- | ---------------------------------------------------------------- |
| [Protocol](protocol.md)       | The rules that make every number comparable (v2)                 |
| [Workflow](workflow.md)       | `list`, `fingerprint`, `check`, `run`, `report`; adding a model  |
| [Backend](backend.md)         | How native `.pt` models are evaluated without an export step     |
| [Results](results.md)         | Where runs are stored and what a report contains                 |

## Setup

On the server, requires [uv](https://docs.astral.sh/uv/).

```
uv sync --extra native   # reference-only installs can use plain `uv sync`
cp local.example.yaml local.yaml
```

Edit `local.yaml` so that `models_dir` points to the `models/` folder of the training
repository (or set `MINI_EVAL_MODELS_DIR`). This file stays on your machine and is
ignored by Git.

The native backend is an **optional** dependency (`[project.optional-dependencies]
native` in `pyproject.toml`) pointed at a checkout of the training repository:

```
[tool.uv.sources]
mini-llm = { path = "../Mini-LLM", editable = true }
```

That is convenient while developing. For reported numbers, pin an exact tag instead
(`mini-llm = { git = "<repo url>", tag = "..." }`) and run `uv lock`, so the code that
produces the probabilities is identified by a commit.

## Commands

```
uv run mini-eval list
uv run mini-eval run smoke mini-llm-150M-base pythia-70m-step128 --device cuda
uv run mini-eval run small_lm --all --device cuda --require-clean
uv run mini-eval report small_lm
uv run pytest
```

The line-up is the four HF references plus **two** mini models: `petal-1-150M`
(`kind: native`, `digit32k/150M/petal-1-150M-base-53375steps.pt`) and
`petal-1-150M-assistant` (`digit32k/150M/petal-1-150M.pt`, the SFT-final checkpoint),
both native and SHA-256-pinned in `models.yaml`. `mini-llm-150M-base` (`GautierP/MiniLLM-150M`
under `base/`) remains as the legacy 1.0 reference. The legacy 34M/40M/70M/150M native
models are not evaluated.

See [Workflow](workflow.md) for what each command does, including the regular
workflow rule: `run` always rebuilds **two reports** from the latest runs —
`reports/<suite>_private.md` (all models) and `reports/<suite>_public.md`
(references only) — so a benchmark never ends without images; previous reports go to
`reports/history/<suite>/<date>/` and the interactive `<suite>_private.html` /
`<suite>_public.html` versions are written alongside. Use `--no-report` to skip it,
and `report` alone to rebuild from existing results.
