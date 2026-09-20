# Experience → Task Transfer in Tool-Using LLM Agents

Research code studying whether prior experience from one tool-use task can help, hurt, or have no effect on an LLM agent's performance on a genuinely new task — and whether that effect can be predicted before the new task is even run.

**Target venue:** IEEE ICA 2026 (6pp regular paper)
**Submission deadline:** September 15, 2026
**Status:** Data collection and analysis complete. Experimental protocol is **frozen** (see [`docs/findings_summary.md`](docs/findings_summary.md) §14) — no further changes to conditions, splits, or feature sets. `docs/findings_summary.md` is the authoritative, confidence-graded results writeup this repo's code produced.

---

## Research Question

> Given a specific prior experience `E` and a specific target task `T`, can we predict — **before running `T`** — whether transferring `E` will help, hurt, or do nothing, and by how much?

Everything else in this project (when/why transfer works, whether representation matters, which experience properties predict transfer) is evidence feeding this one predictive question, evaluated at the individual **experience–task pair** level — never pooled across a task family or memory bank.

Full research design, hard constraints, and rationale live in [`CLAUDE.md`](CLAUDE.md). This README is the practical entry point: what was built, what was found, and how to run it.

---

## Key Results

Full detail, statistics, and confidence levels for every item below are in [`docs/findings_summary.md`](docs/findings_summary.md). Numbers here are final (post-freeze).

- **Baseline ceiling effect.** GLM-4.7-Flash solves 91% (254/280) of held-out tasks with no experience at all. Only 26 tasks are baseline-failures — the only pool where a success-flip can be observed at all. This is why the project added a second outcome variable (`Δ_eff`, effort/efficiency) that isn't ceiling-bound.
- **The hero claim (predicting Δ(E,T) magnitude/direction) does not hold.** Neither magnitude regression nor ternary success-flip classification beat the majority-class or "Break It Down" utility-score baselines, confirmed on the final enriched dataset. Reported as a negative result, not reframed as a win.
- **One real, narrower positive result: predicting the *sign* of efficiency transfer.** Given that an experience is already matched-success, whether it makes the task cheaper or more expensive (completion-token direction) is predictable above both baselines — macro-F1 0.394 vs. best baseline 0.237 (v2, final split) — and the margin *widens* with more data rather than shrinking.
- **`distractor_tool` negative transfer — confirmed, not borderline.** Tasks requiring the agent to *decline* a tool call see significantly more negative transfer when injected with an irrelevant experience: 15/131 negative flips vs. 3/192 in the other three shift categories. Fisher's exact test p = 0.00024, odds ratio 8.15, 95% CI [2.22, 44.56] — survives Bonferroni correction across the 4 categories. Mechanistically verified by reading the actual failing trajectories, not just the aggregate stat.
- **Retrieval quality is a minor factor; content transferability dominates.** Automatic (embedding) retrieval accounts for at most ~4% of the oracle-vs-baseline success gap on hard tasks. Growing the experience bank 3.7x (30→110) produced no measurable improvement in no-transfer rate under either oracle or retrieved selection — evidence against "the bank just needs more/better experiences" as the main story.
- **4 of 9 experience properties (abstraction, specificity, composability, information_context) show persistent floor/ceiling clustering**, reconfirmed at 4x scale after three separate remediation attempts. Treated as a documented characteristic of BFCL's short, structurally homogeneous tasks, not a fixable prompting bug — used only as a secondary ablation feature, never silently included as if reliable.
- **Known limitation, reported rather than patched:** the `oracle` condition's selection rule is a heuristic (tool/family-overlap scoring), and is measurably non-monotonic — more candidate experiences can produce a *worse* pick. Documented with a concrete example task, not hidden.

---

## Repository Structure

```text
experience-transfer-agents/
├── CLAUDE.md                    — full research design spec, hard constraints, decision log
├── README.md                    — this file
├── requirements.txt
│
├── configs/
│   ├── models.yaml               — pinned model, base_url, temperature, max_tokens
│   ├── agent.yaml                 — max_steps, retry/backoff, inter-call delay
│   └── experiments.yaml           — per-experiment task source, seed, output path
│
├── src/
│   ├── agent/                     — instrumented agent loop (the only place that calls the API)
│   ├── environment/                — task loading, exact-match evaluation (no LLM judge)
│   ├── experience/                 — experience generation, raw/reflection/procedure, retrieval
│   ├── experiments/                — condition runner (baseline/oracle/retrieved), result logging
│   └── analysis/                   — Δ(E,T) / Δ_eff computation, 9-property features, predictor, stats
│
├── tasks/                          — adapted BFCL tasks; frozen experience-gen/held-out split
├── data/
│   ├── bfcl_raw/                   — original BFCL v4 category files
│   ├── experiences/                — the 110-experience bank (JSONL)
│   └── results/                    — raw, machine-written harness output (never hand-edited)
│
├── scripts/                        — reproducible CLI entry points (data collection + analysis)
├── notebooks/                      — exploration/orchestration (calls into src/, not a second copy)
│
├── results/
│   ├── processed/                  — scored (E,T) pairs, frozen predictor splits (v1/v2)
│   ├── figures/                    — final paper figures
│   └── tables/
│
├── docs/
│   ├── findings_summary.md         — FINAL, confidence-graded results writeup (source of truth)
│   ├── experiment_log.md           — running day-by-day dev/debugging log
│   └── Research Plan.md            — original hypothesis and proposal
│
└── tests/                          — pytest suite (agent, evaluation, experience, features, predictor, statistics, transfer)
```

`data/results/` (raw harness output) and `results/` (derived analysis output) are kept deliberately separate — the harness only ever writes to the former; analysis code reads it and writes to the latter.

---

## Setup

Requires Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the repo root containing your API key:

```text
GLM_API_KEY=<your key>
```

Never commit `.env` or any file containing credentials (already covered by `.gitignore`).

Before running any evaluation, verify the served model matches `configs/models.yaml` — every collection script (`scripts/run_*.py`) does this automatically on its first call and raises `SystemExit` on a mismatch rather than silently trusting drifted results.

---

## Reproducing the Analysis

```bash
./scripts/reproduce_analysis.sh
```

This deterministically rebuilds every downstream analysis artifact — scored `(E,T)` pairs, the predictor report (both targets, both baselines), and all final figures in `results/figures/` — from the raw data already collected (`data/results/*.jsonl`, `data/experiences/*.jsonl`). Frozen splits are skipped, never overwritten. Safe to re-run at any time.

**This does not call the live API.** Data collection (generating experiences; running baseline/oracle/retrieved conditions) required live judgment calls and explicit API-budget gating at each stage. To reproduce that side, run the individual `scripts/*.py` entry points in the order documented in `docs/findings_summary.md`, against the same `configs/*.yaml` and a `.env` with `GLM_API_KEY`.

### Pinned versions

- **Python:** 3.11
- **Model:** `glm-4.7-flash` (Z.ai/Zhipu, open-weight, 30B total / ~3B active MoE) — pinned in `configs/models.yaml` along with `base_url`, `temperature: 0`, `max_tokens`. Single-model policy: the same model runs the agent, generates `reflection`/`procedure` representations, and rates experience properties.
- **Seeds:** `42` everywhere task sampling occurs, set explicitly in `configs/experiments.yaml` / script constants — never left to a library default.
- **Task/dataset versions:** BFCL v4, adapted via `scripts/adapt_bfcl.py`. The frozen experience-generation/held-out task split lives in `tasks/split.json` (never modified after freezing). The frozen predictor-pair split lives in `results/processed/predictor_split.json` (v1) and `predictor_split_v2.json` (v2 — additive extension; v1's assignments are preserved exactly).

---

## Experimental Design (condensed)

**Conditions:** `baseline` (no experience) / `oracle` (deliberately best-matched experience) / `retrieved` (automatic embedding retrieval, MiniLM cosine similarity, fixed before evaluation). Token/inference budget is matched and logged across all three.

**Task shift categories** (increasing novelty relative to experience-generation tasks): `familiar` → `novel_composition` → `novel_tool` → `distractor_tool`. (`ood` is excluded from the current design.)

**Experience representations**, all derived from the same source trajectory: `raw` (full trajectory) / `reflection` (LLM-generated summary) / `procedure` (LLM-generated step-by-step plan).

**Outcome variables:**
- `Δ(E,T) = P(T|E) − P(T|∅)` — success-flip transfer gain, per pair.
- `Δ_eff(E,T)` — efficiency transfer, defined only on completion tokens + steps + tool calls (never raw/input tokens, which trivially rise from injecting context regardless of benefit), computed on matched-success pairs only.

**Experience properties (9 logged per experience):** `length`, `num_steps`, `success_failure`, `task_structure`, `tool_dependence` (reliable) plus `abstraction`, `specificity`, `composability`, `information_context` (clustered — ablation-only, see Key Results).

Full definitions, hard constraints (frozen splits, fixed retrieval, matched budgets, etc.), and the reasoning behind each are in `CLAUDE.md`.

---

## Data Scale

- **Benchmark:** BFCL v4 (Berkeley Function-Calling Leaderboard), adapted into this project's task schema. Ground truth is exact-match against the gold tool call — never an LLM judge.
- **Adapted dataset:** 1,240 tasks → frozen split of 280 experience-generation / 280 held-out test / 680 reserve.
- **Experience bank:** 110 experiences, each with all three representations and all 9 properties.
- **Final scored dataset:** 323 `(E,T)` pairs across baseline/oracle/retrieved, all four shift categories, and the representation-crossing and `distractor_tool`-oversampling follow-ups.

---

## Testing

```bash
pytest
```

Covers transfer-gain computation, property extraction, the representation transformer, evaluation logic, and the predictor/statistics modules — written before spending API budget on the harness itself.

---

## Documentation Map

- [`docs/findings_summary.md`](docs/findings_summary.md) — **start here for results.** Final, confidence-graded writeup; the direct source for the paper's Results and Limitations sections.
- [`docs/experiment_log.md`](docs/experiment_log.md) — running day-by-day log of what was tried, what broke, and how it was fixed. Historical record, not a polished summary.
- [`docs/Research Plan.md`](docs/Research%20Plan.md) — the original hypothesis and proposal that motivated the project.
- [`CLAUDE.md`](CLAUDE.md) — full research design spec: definitions, hard constraints, model/serving decisions, schema contracts.

---

## Research Principles

1. Evaluate transfer at the individual `(E,T)` pair level — never pooled.
2. Keep experience-generation and held-out tasks disjoint.
3. Freeze the test split before evaluation; never modify it afterward.
4. Use exact-match task evaluation, never an LLM judge.
5. Match inference/token budgets across experimental conditions.
6. Fix the retrieval method before evaluating transfer.
7. Separate retrieval failure from genuine non-transfer.
8. Use the same underlying trajectory when comparing experience representations.
9. Keep exploratory analysis separate from confirmatory analysis.
10. Report negative and null transfer, not only successful cases.
11. Do not silently reproduce related work's contributions and present them as novel.
12. Record experimental limitations rather than tuning them away after seeing results.

---

## Citation

The final paper will document the benchmark, related work, experimental methodology, and any external models or tools used. A citation file will be added once the paper and bibliography are finalized.

---

## License

This repository contains research code and experiment artifacts. Repository license and redistribution terms will be finalized alongside the paper release. Benchmark data remains subject to BFCL's original licensing and usage terms.
