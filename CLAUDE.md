# Project: Experience → Task Transfer in Tool-Use LLM Agents

Target venue: IEEE ICA 2026 (regular paper, 6pp, submission deadline Sept 15, 2026).

## Research Question
**Overarching:** Can an open-source LLM agent use experience from previous tasks to perform better on genuinely new tasks? If so, why does transfer work or fail, what makes an experience transferable, does representation affect transfer, and how transferable is a given experience?

**Hero sub-question — this is the paper's headline claim, everything else supports it:**
Given a specific prior experience E and a specific target task T, can we predict — *before running T* — whether transferring E will produce positive, zero, or negative transfer, and by how much? Evaluated against (a) a majority-class / mean-Δ baseline and (b) a re-implementation of the specificity × abstractness utility score from "Break It Down, Pass It On" (arXiv 2608.20274).

All other sub-questions (when/why transfer works, whether representation matters, which properties predict transfer) are evidence and features that feed the hero question — not standalone claims. Any finding that only replicates prior work must be framed as "extends X to the tool-use domain," never as novel. In particular:
- Negative transfer concentrating on hard cases, and abstraction reducing negative transfer, are already shown in "When Continual Learning Moves to Memory" (arXiv 2604.27003) — treat replicating this in tool-use as supporting evidence, not a finding.
- Do not implement anything that silently reinvents the above two papers' contributions and present it as new — if a design choice looks like it's converging on their exact method, flag it to me.

## Core Definitions (use these exact terms everywhere — code, comments, commit messages, docstrings, plots)
- **Experience (E):** a record of the agent's interaction with a task — task context, actions, observations, and outcome.
- **Transfer:** occurs when a prior experience changes agent performance on a new task.
- **Transfer gain / Δ(E,T):** Δ(E,T) = P(T | E) − P(T | ∅), computed per (experience, task) **pair** — NOT pooled across a task family or memory bank. This pairwise granularity is our main point of difference from both concurrent papers above; do not aggregate it away by default.
- **Efficiency transfer / Δ_eff(E,T):** a second outcome variable, defined only on execution-side effort — completion tokens, number of steps, number of tool calls, number of retries — never on raw total/input tokens, which trivially increase from injecting experience context regardless of whether it helps. Computed conditional on matched success (baseline and experience condition both succeed), since efficiency is only cleanly interpretable when the outcome is held constant. Added specifically because Δ(E,T) for success is ceiling-bound (see Experimental Conditions); Δ_eff is not — it can be measured on effectively the entire dataset, not just the baseline-failure subset. Not claimed as independently novel — memory-driven efficiency gains are well-established in prior agent-memory literature — the novel piece is measuring it pairwise, per (E,T), and feeding it into the same predictor as success-transfer.
- **Task shift:** a controlled, tagged category describing how a target task differs from experience-generation tasks. Ordered by increasing novelty: `familiar` → `novel_composition` → `novel_tool` → `distractor_tool` → `ood`.
- **Representation:** how an experience is encoded for reuse — `raw` (full trajectory), `reflection` (LLM-generated summary/lesson), `procedure` (LLM-generated step-by-step plan). All three must be derived from the SAME underlying trajectory for any given experience — never independently collected, or the representation comparison is confounded with content.
- **Experience properties (9, logged per experience):** abstraction, specificity, task_structure, tool_dependence, length, num_steps, success_failure, composability, information_context.

## Experimental Conditions (do not conflate these)
1. `baseline` — agent attempts T with no experience.
2. `oracle` — agent attempts T with the single best-matching experience, deliberately/manually selected.
3. `retrieved` — agent attempts T with the single experience returned by automatic retrieval (embedding similarity).

Hard constraints:
- Token/inference budget must be matched and logged across all three conditions. If oracle/retrieved get more context, Δ(E,T) is confounded with budget, not experience.
- Retrieval method must be fixed BEFORE evaluation begins — no tuning retrieval after seeing results.
- Train/experience-generation vs. held-out/test task split must be frozen before any evaluation and never modified afterward.
- Every held-out test set needs a baseline-success/baseline-fail (hard/easy) split so we can separate genuine negative transfer from retrieval failure.
- **Ceiling check per shift_category (measured, Step 2, 280/280):** aggregate baseline success 91% (254/280). Breakdown: familiar 91% (7 hard), novel_tool 93% (4 hard, CEILING), novel_composition 89% (9 hard), distractor_tool 90% (6 hard, CEILING). Only 26 baseline-failure tasks total across all 280 — confirmed at full scale, not a pilot artifact (reserve pool would yield roughly the same rate; not worth exhausting reserve to chase this further). **Consequence: Δ(E,T) success-flip analysis is now a secondary, small-n (26 hard cases) analysis, reported with explicit statistical hedging — Δ_eff carries the primary quantitative predictor claim, with ~254 matched-success pairs available.**
- **Noise floor (measured, Step 4):** 8% flip rate (1/12 trials: 4 tasks × 3 repeats, one per shift_category). The flip was entirely concentrated in `bfcl_multiple_25` (novel_tool, swap-similar-API decision); the other three tasks were stable across all repeats. Treat this as a hypothesis pending confirmation across more novel_tool pairs during Wednesday's run, not a settled category-level property yet — n=1 task is thin. Interim rule: treat single-trial Δ magnitudes below ~8% as ambiguous, especially on novel_tool pairs.
- **Efficiency ≠ raw token count (added when Δ_eff was introduced):** never define Δ_eff on total/input tokens — prompt-side cost trivially rises whenever experience is injected, independent of whether it actually helps. Use completion tokens, steps, tool calls, and retries only, and compute conditional on matched success.
- **retry_count excluded from Δ_eff's aggregate (found Step 4 pilot):** retry_count as implemented tracks HTTP-layer retries (rate limits, timeouts, transient errors) — shared infrastructure noise, not per-condition agent effort. Confirmed by identical retry deltas across oracle/retrieved on the same task. Keep logging it for endpoint-health monitoring, but Δ_eff's effort metric is completion_tokens + steps + tool_call_count only.
- **Report median alongside mean for Δ_eff (confirmed necessary, Step 4 pilot):** at n=4, oracle's mean completion-token delta (+2.25) and median (−19.50) disagreed in *sign* — a single favorable outlier was masking a net-cost typical case. Always report both; don't trust mean alone at small n, and watch specifically for cases where experience adds cost on already-solvable tasks without a success-rate benefit — a subtler form of negative transfer than a success/failure flip.

## Data scale target
The predictor (hero result) needs a defensible held-out evaluation set. Target **at least 150-200 scored (E,T) pairs** by Thursday, not the ~20-50 experiences generated Tuesday. Size task counts, experience-generation task counts, and the shift-category matrix with this end target in mind from the start — don't build infrastructure that only works for 20-50 total pairs.

## Domain
Tool-use / API-calling tasks. Ground truth = exact match on the gold API call(s) and parameters, not an LLM judge. Adapt an existing tool-use benchmark (BFCL, τ-bench, or API-Bank) rather than building a custom sandbox — a custom sandbox is out of scope for this sprint.

## How the Sub-Questions Connect (read this before building the predictor)
Sub-questions (a) when/why transfer works or fails, (b) what makes an experience transferable, and (c) does representation affect transfer are NOT separate experiments run alongside the predictor — they are the experiments that generate the predictor's training data. No additional experiments are needed for (d).

- **(a)** is answered directly by Wednesday's baseline/oracle/retrieved results, tagged by `shift_category`, plus the hard/easy diagnostic and failure categorization (wrong experience, tool dependence, overgeneralization, conflicting experiences, composition failure).
- **(c)** is answered directly by Thursday's raw/reflection/procedure crossing on the same underlying trajectory.
- **(b)** is answered directly by Thursday's 9-property extraction correlated against Δ(E,T) from the same pairs above.
- **(d)** — the predictor — is a supervised model trained on the union of the above: every (E,T) pair has features `{shift_category, representation, the 9 properties}` and a label `Δ(E,T)`. This is why per-pair granularity (Data Schema Contracts: `Result record`) matters — it's literally the row structure the predictor learns from. Evaluate on a held-out slice of pairs never used in training. **Δ_eff(E,T) is a secondary label the same feature set can also be trained/evaluated against, and is not ceiling-limited the way Δ(E,T) is — useful if the §Data scale target's positive-transfer examples turn out thin.**

## Property Variance — Closed Investigation (Step 6, reconfirmed Day 1 wrap-up at 4x scale)
Three remediation attempts on abstraction/specificity/composability/information_context (biased "be general" prompt → ceiling cluster; neutral prompt → floor cluster, pole flipped not fixed; anchored 1/3/5 rubric re-rating → still clustered) converged on the same conclusion: this is likely a domain characteristic of BFCL's short, structurally similar tool-use tasks bounding achievable variance on these 4 properties, not a fixable prompt/rubric artifact. Reconfirmed on the bank-growth batch (30→110 experiences): same clustering pattern held at 4x the original sample (abstraction 60/80 floor, specificity 62/80 ceiling, composability 64/80 floor, information_context 55/80 ceiling) — this is now a well-supported finding, not a small-sample fluke. Documented in full in `docs/experiment_log.md`. **For the predictor: treat length, num_steps, success_failure, task_structure, tool_dependence, and Δ_eff as the reliable feature set. Either exclude the 4 clustered properties from the primary predictor and report them as a secondary ablation, or explicitly report their near-zero feature importance if included — do not include them silently.**

## Models
**Before running ANY evaluation, including calibration pilots: confirm the agent config actually points to the model below.** This has already drifted once (a local Qwen2.5-3B default surfaced instead of the decided model) — verify with one live test call before trusting any results, calibration included.

**Model change log (for context, not re-litigation):** local HF (unspecified) → GLM-5.2 (744B, self-host infeasible) → DeepSeek-V4-Flash-0731 via NVIDIA (rate-limit risk) → **GLM-4.7-Flash (current)**. **No further model changes before the calibration pilot has actually run once** — a new model is not a reason to delay Step 3 again; a pilot result that reveals a genuine capability problem is.

- **Single-model policy:** the same model both runs the agent AND generates the `reflection`/`procedure` representations. Do not use a different (e.g. larger/stronger) model for representation generation — that would confound "does representation format help" with "is the generator model just smarter," and undermines the representation comparison in Core Definitions.
- **Model: GLM-4.7-Flash** (Z.ai/Zhipu, open-weight, 30B total / ~3B active MoE), optimized for agentic tool-use and coding — scores 79.5 on τ²-Bench (interactive tool invocation), directly relevant to this domain.
- **Serving: hosted API, primary — Z.ai's own free endpoint** (official checkpoint). Redundant free/cheap fallbacks exist if needed: Cloudflare Workers AI (`@cf/zai-org/glm-4.7-flash`) and OpenRouter (DeepInfra/Venice/Novita, ~$0.06/$0.40 per M tokens). **Local self-hosting via vLLM is a genuine fallback this time** — small enough to plausibly run on a single GPU, unlike prior candidates — worth keeping in reserve if the free API tier throttles under Thursday's volume, rather than switching models again.
- **Reproducibility:** pin and log the exact checkpoint/provider used per result in `configs/models.yaml`, since this model is served by several different hosts and provider-level differences (quantization, etc.) are possible.

## Repo Structure — target end-state (the repo grows into this over the sprint; it will not exist all at once on day one)
```
experience-transfer/
│
├── README.md
├── requirements.txt
├── .gitignore
├── pyproject.toml
│
├── configs/
│   ├── models.yaml
│   ├── agent.yaml
│   ├── tasks.yaml
│   └── experiments.yaml
│
├── src/
│   ├── agent/            — existing agent loop, wrapped/instrumented, never rewritten
│   │   ├── agent.py
│   │   ├── prompts.py
│   │   └── config.py
│   ├── tools/
│   │   ├── calculator.py
│   │   ├── registry.py
│   │   └── tool_interface.py
│   ├── environment/
│   │   ├── environment.py
│   │   ├── tasks.py
│   │   ├── task_families.py
│   │   └── evaluation.py     — exact gold-call match, not LLM judge
│   ├── experience/
│   │   ├── schema.py
│   │   ├── generator.py
│   │   ├── representations.py   — raw / reflection / procedure, same source trajectory
│   │   ├── storage.py
│   │   └── retrieval.py
│   ├── experiments/
│   │   ├── runner.py
│   │   ├── conditions.py     — baseline / oracle / retrieved, matched budget
│   │   └── logging.py
│   └── analysis/
│       ├── transfer.py       — Δ(E,T) per pair
│       ├── features.py       — the 9 experience properties
│       ├── predictor.py      — hero-result predictor + both baselines
│       └── statistics.py
│
├── tasks/
│   ├── task_definitions/
│   ├── experience_tasks/
│   ├── test_tasks/           — frozen held-out split lives here, never modified after freezing
│   └── metadata/
│
├── data/
│   ├── experiences/
│   ├── trajectories/
│   └── results/               — raw per-(task,condition) result records, machine-written by the harness only
│
├── notebooks/                 — exploration + orchestration; should mostly call src/ functions, not reimplement logic
│   ├── 01_agent_sanity_check.ipynb
│   ├── 02_generate_experiences.ipynb
│   ├── 03_baseline.ipynb
│   ├── 04_transfer_experiment.ipynb
│   ├── 05_representation_experiment.ipynb
│   ├── 06_transfer_analysis.ipynb
│   └── 07_predict_transfer.ipynb
│
├── scripts/                   — reproducible CLI entry points; the source of truth notebooks call into
│   ├── generate_experiences.py
│   ├── run_experiment.py
│   └── run_analysis.py
│
├── results/
│   ├── raw/                   — exports assembled from data/results/ for a specific analysis pass
│   ├── processed/              — cleaned/aggregated tables
│   ├── figures/
│   └── tables/
│
├── docs/
│   ├── experimental_design.md  — human-readable version of this file's definitions/schemas
│   ├── related_work.md         — positioning vs. Break It Down (2608.20274) and the CL-memory paper (2604.27003)
│   ├── experiment_log.md
│   └── paper_outline.md
│
└── tests/
    ├── test_agent.py
    ├── test_tools.py
    ├── test_tasks.py
    ├── test_experience.py
    └── test_evaluation.py
```
Note the naming overlap: `data/results/` is raw harness output, never hand-edited; `results/` is what analysis code derives from it for the paper. Keep that distinction explicit rather than writing to both interchangeably.

## Data Schema Contracts (fix these before building both sides of any pipeline)
- **Task record:** `{task_id, tools_available, goal, gold_call, shift_category, split}`
- **Experience record:** `{experience_id, source_task_id, trajectory, representations: {raw, reflection, procedure}, properties: {...9 fields...}}`
- **Result record:** `{task_id, experience_id_or_null, condition, success, prompt_tokens, completion_tokens, steps, tool_call_count, retry_count, trajectory, timestamp}` — tokens split into prompt/completion (not one combined `tokens_used` field) specifically so Δ_eff can be computed on completion-side effort only.

## Engineering Practices
- Notebooks (`notebooks/`) are for exploration and orchestration, not a second copy of the logic. A notebook cell should mostly read `from src.X import Y; Y(...)`. If logic in a notebook becomes something you'd rerun, promote it into `src/` before moving on — otherwise the notebook and the "real" pipeline silently drift apart.
- Add a dry-run/mock mode for the harness so plumbing can be tested without spending API budget.
- Write unit tests for: transfer-gain computation, property extraction, representation transformer — before spending real API budget on the harness.
- Every experiment run must be reproducible from a config file + seed; log model version and dataset version.
- Keep exploratory analysis code separate from confirmatory analysis code — don't let exploration silently become the reported result.