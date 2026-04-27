# TTC eval pipeline

An evaluation pipeline for The Token Company's input-compression API (`bear-1`, `bear-1.1`, `bear-1.2`). It measures whether compressing prompts with TTC before sending them to a downstream LLM hurts accuracy, helps cost, helps latency, and where the sweet spot sits across the family of compressor models. The framing is borrowed from TTC's own
**Pax Historia** case study (compressed prompts *outperformed* uncompressed
in a 268K-vote blind A/B): treat the eval as a measurement instrument for
when compression *helps* and when it *hurts*, not as a hatchet job.

## Sample run

A real run on **OpenAI gpt-4o-mini** (5 tasks, n=6–15 per task, 27-min wall
clock, 0 failures) is committed at
[`results/sample_run.html`](results/sample_run.html) — open it locally to see
actual numbers without re-running the eval. Headlines from that run:

| Task | Sweet spot | Accuracy delta vs baseline | Compression |
| --- | --- | --- | --- |
| QA | `bear-1.2` @ 0.9 | F1 0.867 → 0.956 (**+8.9pp**) | 35.7% |
| Summarization | `bear-1.1` @ 0.9 | ROUGE +1.8pp | 33.5% |
| RAG | `bear-1.2` @ 0.9 | retrieval 1.00 → 1.00 (flat) | 35.8% |
| Long context | `bear-1.2` @ 0.9 | retrieval 1.00 → 1.00 (flat) | **56.2%** |

## Quick start

```bash
pip install -r requirements.txt

# 1) Smoke test with no API keys, end-to-end report in seconds.
python run_eval.py --llm mock --samples 5

# 2) Real run requires OPENAI_API_KEY and TTC_API_KEY.
#    Either export them, or paste them into the .env file (auto-loaded):
cp .env.example .env  # if you haven't yet
# edit .env, then:
python run_eval.py --llm openai --use-embeddings
```
Both runs produce `results/results_<timestamp>.json` (raw audit trail, every
sample) and `results/results_<timestamp>.html` (the human-facing report).
Open the HTML in any browser. It's self-contained, no CDN, no JS.

## Architecture

```
Datasets ─┬──> TTCClient ──┐
          │                ├──> compressed input ──> LLMBackend ──> predictions ─┐
          └────────────────┘                                                      ├──> metrics ──> Aggregator ──> JSON ──> HTML report
                          (baseline path: skip compression, send raw context) ───┘
```

Five tasks run in sequence. For each task:

1. **Baseline**: send raw input to the LLM (no compression).
2. **Sweep**: compress with each `(model, aggressiveness)` cell, score per
   sample, aggregate with bootstrap CIs, pick the Pareto-knee cell.
3. **Per-task sub-experiments** where applicable: `<ttc_safe>` for QA,
   faithfulness scatter for summarization, per-turn-position accuracy for
   conversational, position × aggressiveness heatmap for long-context.

The CLI sweep defaults to `models = [bear-1, bear-1.1, bear-1.2]` ×
`aggressiveness = [0.1, 0.3, 0.5, 0.7, 0.9]`, a 3×5 grid per task.

## What each metric captures and why

| Metric | What it answers | Used by |
| --- | --- | --- |
| **EM** (Exact Match) | "Did we get the answer span verbatim?" SQuAD-normalized: lowercase, strip punctuation + articles. | QA, conversational, long-context (as F1 fallback) |
| **F1 token overlap** | "How much of the answer did we capture?" Standard SQuAD F1 over normalized tokens. | QA, conversational |
| **ROUGE-L** | "Does the summary cover the same content as the reference?" Longest-common-subsequence based. | summarization |
| **Faithfulness** | "Does the summary stay grounded in the **original** source even when the LLM only saw the compressed version?" The hallucination guard. NLI cross-encoder when `--use-embeddings`; lexical noun-phrase fallback otherwise. | summarization |
| **Semantic similarity** | "How close is one passage to another in meaning?" Sentence-transformer cosine when `--use-embeddings`; TF-IDF cosine fallback. | RAG ranking |
| **Retrieval accuracy** | "Did we rank the correct doc / recover the needle?" | RAG, long-context |
| **MRR** | Reciprocal rank of the gold document. Soft signal when the relevant doc isn't #1. | RAG |
| **accuracy_delta** | `compressed_score − baseline_score`. Positive = compression *helped* (the Pax Historia direction). Surfaced everywhere with sign. | all |
| **bootstrap_ci** | Honest uncertainty for n≈10–50. Reported alongside every metric mean. | all |

## Cost model

The product makes money for the *customer* iff:

```
(downstream LLM tokens NOT sent × downstream LLM input price)
   −  (TTC tokens removed × $0.05 / 1M)
   >  0
```

Worked example. 10K-token context compressed 40% to 6K tokens, GPT-4o-mini
($0.15 / 1M input):

```
TTC bill          : 4,000 × $0.05 / 1M     = $0.0002
LLM saved         : 4,000 × $0.15 / 1M     = $0.0006
Net per request   :                        = $0.0004
At 1M req/day     :                        = $400 / day
```

The report's **Net cost analysis** section runs this math against every
sweep cell and projects to 1M, 10M, 100M req/day. The TTC bill is on
*tokens removed*; the downstream bill is on *tokens sent*. Net savings is
the difference, and it's the customer-facing ROI of the product. The HTML
report's executive-summary card leads with this number.

The price table for the major frontier models lives in
[`eval/tokenizers.py`](eval/tokenizers.py): it's a snapshot from late
January 2026 and will go stale. Production would pull from a config file
or pricing API.

## Sweet-spot logic

Highest mean accuracy is not the best setting. That's just `aggressiveness ~ 0`
with no real compression. Highest compression ≠ best either. It tanks
accuracy. Pick the **Pareto-knee** cell:

```
utility(cell) = (cell_accuracy − baseline_accuracy) + λ × cell_compression_ratio
```

with `λ = 0.5` by default. Cells whose accuracy delta is below −2pp are
disqualified. The selection rule lives in
[`eval/tasks/_common.py:select_best_cell`](eval/tasks/_common.py).

`λ` is the tradeoff weight between "more accuracy" and "more compression."
λ=0 means "only count accuracy" (always picks aggressiveness=0). λ->infinity means
"only count compression" (always picks aggressiveness=1). λ=0.5 says a 5pp
accuracy gain is worth the same as a 10-percentage-point increase in
compression ratio. That feels right for most customers; tune it for yours.

## Reading the report

The HTML report has 11 sections, in order:

1. **Executive summary**: net daily savings, sweet-spot compression
   ratio, accuracy delta vs baseline, p95 E2E latency change. Headline numbers.
2. **Sweet-spot recommendation**: names the recommended `(model, aggressiveness)`.
3. **Accuracy vs aggressiveness curves**: one chart per task, one line per
   model, shaded bootstrap CIs, dashed baseline line.
4. **`<ttc_safe>` impact**: bar chart: protected vs unprotected QA accuracy
   at each aggressiveness level. Tests TTC's protected-region feature.
5. **Compression-ratio distribution**: per task per model.
6. **Latency**: p50/p95/p99 for TTC compress, downstream LLM, E2E
   compressed, and the LLM-only baseline.
7. **Net cost analysis**: TTC bill, LLM savings, net daily savings at
   1M / 10M / 100M req/day. The customer-facing ROI table.
8. **Faithfulness vs ROUGE scatter** (summarization): top-right is good;
   bottom-right (high ROUGE, low faithfulness) is **compression-induced
   confabulation** and gets called out by sample id.
9. **Long-context heatmap**: needle-position × aggressiveness. The
   single chart most likely to *sell* the product.
10. **Regression section**: only if `--baseline` was passed. Side-by-side
    diff against a prior run.
11. **Full results table**: every cell, sortable.

How to read the numbers:

* **Accuracy delta is signed.** Positive = compression helped on this
  cell. Embrace it where it shows up that's the Pax Historia framing.
* **A "good" compression ratio depends on the task.** RAG-style retrieval
  generally compresses 30–50% with little accuracy cost; QA on dense facts
  compresses less because the facts themselves are the high-information
  tokens.
* **Faithfulness drop without ROUGE drop** is the signal to watch. ROUGE
  measures overlap with a reference summary; faithfulness measures
  grounding in the source. If ROUGE stays flat while faithfulness drops,
  the compressor is removing details the LLM then confabulates.

## The `<ttc_safe>` experiment

For 20% of QA samples, I wrap the question in `<ttc_safe>...</ttc_safe>`
and prepend it to the context before compression. The TTC docs guarantee
that protected regions are passed through unmodified. Hypothesis:
protected questions resist accuracy loss as aggressiveness rises, since
the words the LLM most needs to see (the question) are guaranteed to
survive. The bar chart in the report puts protected next to unprotected
at every sweep cell so the difference is visible at a glance.

## Regression mode

```bash
python run_eval.py --llm openai --baseline results/results_20260101T000000Z.json
```

Diffs current vs baseline at the metric level (EM/F1/ROUGE-L/retrieval,
compression ratio, faithfulness, p95 E2E latency). Anything moving more
than 1pp (accuracy-style) or 5% (ratio-style) is flagged. Useful for the
internal "we want to ship `bear-1.3` without surprising customers" loop
run the eval, point `--baseline` at the last accepted run, eyeball the
red rows. Lives in [`eval/regression.py`](eval/regression.py).

## Extending

**Add a task.** Subclass `EvalTask` in a new file under `eval/tasks/`.
Drop a JSON dataset in `datasets/`. Register the class in
`run_eval.py`'s `TASK_REGISTRY`. Done, the runner picks it up, the
report renders it, regression mode handles it.

**Swap the downstream LLM.** Subclass `LLMBackend` in
[`eval/llm_backend.py`](eval/llm_backend.py). Add it to `build_backend`.
The interface is three methods: `answer`, `summarize`, `chat`.

**Compression middleware for production calls.** `CompressingBackend`
wraps any `LLMBackend` and inserts the TTC call between your prompt and
the LLM:

```python
from eval.client import TTCClient
from eval.llm_backend import CompressingBackend, OpenAIBackend

ttc = TTCClient(model="bear-1.2")
llm = CompressingBackend(
    OpenAIBackend(model="gpt-4o-mini"),
    ttc,
    aggressiveness=0.1,             
    fail_open=True,                    
    compress_roles={"user"},             
    min_chars_to_compress=200,         
)
answer = llm.answer(context, question)
```

A runnable example lives at
[`examples/compressed_llm_call.py`](examples/compressed_llm_call.py).

**Re-render an existing run.** Iteration on the report layout doesn't
need a re-run:

```bash
python report.py results/results_20260426T000627Z.json
```

## Limitations and honest notes

* **Small-sample CIs.** With n~10-50 per cell the bootstrap CIs are wide.
  This is reported, not papered over, the eval is meant to surface
  directional signal, not declare statistical victory. The included
  datasets are starter sets sized for a take-home; expanding any of the
  JSON files in `datasets/` immediately tightens CIs.
* **Synthetic data, no production traffic.** All passages are
  hand-written, deliberately noisy enough for compression to have signal
  to remove. A production eval should drive this with a sample of real
  customer prompts.
* **Pricing snapshot.** The model price table in `eval/tokenizers.py` is
  pinned to a moment in time. The README's worked example will go stale
  as prices move.
* **OpenAI is the only supported downstream backend.** The pipeline was
  scoped to OpenAI for this submission. Adding a new backend is one
  `LLMBackend` subclass plus one line in `build_backend`, see "Extending."
* **Latency table includes client-side rate-limit pacing.** The TTC
  free-tier limit is 60 req/min; I paced it at 50/min to leave headroom (see
  `--ttc-rpm`). For high-fan-out tasks like RAG (5 compress calls per
  sample), the p95 E2E numbers in the report's latency section are
  dominated by this pacing wait, not by TTC's actual compress latency.
  Run with `--ttc-rpm 0` on a higher-tier quota to see TTC's true
  per-call latency.
* **Mock TTC client.** When `--llm mock` runs without `TTC_API_KEY`, a
  `MockTTCClient` substitutes for the real API so the smoke test can
  produce a complete HTML end-to-end. The mock removes a hand-picked set
  of low-signal tokens at a probability scaled by aggressiveness, it is
  *not* a model of TTC's actual compressor and should not be used to
  evaluate anything beyond "the pipeline runs."
* **Custom client over the official SDK.** TTC ships
  [`tokenc-python-sdk`](https://github.com/TheTokenCompany/tokenc-python-sdk).
  I used a thin custom `requests`-based client instead. The rationale
  (instrumentation, lighter deps for the eval) is the comment at the top
  of [`eval/client.py`](eval/client.py).

## Repo conventions

* Python 3.10+, full type hints, `from __future__ import annotations`
  where it helps.
* `ruff` clean. CI runs `ruff check .` and `pytest -q` on every push.
* Docstrings on classes and public methods. Comments explain *why*, not
  *what*.
* Atomic commits, one logical change per commit, conventional-style messages.
* Structured logging throughout (`eval/logging_config.py`); never `print`
  outside the CLI's user-facing output.
* No hardcoded keys anywhere; env vars only.

## File map

```
ttc-eval/
├── run_eval.py                    # CLI entrypoint
├── report.py                      # Self-contained HTML report
├── eval/
│   ├── client.py                  # TTCClient + MockTTCClient
│   ├── tokenizers.py              # tiktoken / anthropic / fallback + price table
│   ├── metrics.py                 # EM, F1, ROUGE-L, faithfulness, bootstrap CI
│   ├── llm_backend.py             # Mock + OpenAI + CompressingBackend wrapper
│   ├── logging_config.py          # JSON-line + human formatters
│   ├── regression.py              # Compare current vs prior baseline JSON
│   ├── runner.py                  # EvalRunner, idempotence pre-check, JSON output
│   └── tasks/
│       ├── base.py                # EvalTask, SamplePrediction, SweepCell, TaskResult
│       ├── _common.py             # aggregate_with_ci, build_sweep_cell, select_best_cell
│       ├── qa.py                  # QA + <ttc_safe> sub-experiment
│       ├── summarization.py       # ROUGE-L + faithfulness vs original
│       ├── rag.py                 # 5-doc retrieval ranking
│       ├── conversational.py      # CoQA-style multi-turn
│       └── long_context.py        # Needle-in-haystack heatmap
├── datasets/                      # Starter JSON datasets, expand as needed
├── tests/                         # pytest unit tests
├── .github/workflows/test.yml     # ruff + pytest on push
├── pyproject.toml                 # ruff config + project metadata
└── requirements.txt
```
