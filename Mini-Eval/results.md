# Results

```
results/<model>/<suite>/<timestamp>/
    results.json    scores, standard errors, versions, sample counts
    meta.json       model entry, suite, device, versions, git state, fingerprints
    samples/        one line per question (ignored by Git)
```

Runs are never overwritten. `report` uses the latest run of each model and suite.

The report is generated in **two renderings**, from the same runs, both committed:

- `reports/<suite>_private.md` (+ `<suite>_private_vs_tokens.png`, `<suite>_private_vs_params.png`, `<suite>_private.html`) — every registered model, including the mini ones.
- `reports/<suite>_public.md` (+ `<suite>_public_vs_tokens.png`, `<suite>_public_vs_params.png`, `<suite>_public.html`) — references plus the models flagged `public: true` in the registry (the Petal release); mini-llm, its internal alpha, stays private.

Each rebuilt report is archived first: `reports/history/<suite>/<generation date>/`
keeps the previous markdown, plots and HTML, sorted by date, so past evaluations
always stay readable. The interactive versions (hover a point to see the model, its
score and the run date) live in `reports/<suite>_private.html` and
`reports/<suite>_public.html`. See [Workflow](workflow.md) for when they are rebuilt.

## Tests

`uv run pytest` runs the adapter against hand computations, inside the real `lm-eval`
harness on a tiny local task, and the registry, runner and report logic. These tests
use fake models and need no network. They run on the server too: the package refuses
to import on macOS, so there is no local test path either. `tests/test_native.py` is
skipped unless PyTorch and the training package are installed.
