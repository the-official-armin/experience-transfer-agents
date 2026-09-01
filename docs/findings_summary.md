# Findings Summary — Experience → Task Transfer in Tool-Use Agents

**Status: FINAL. Protocol frozen as of Day 3 (see §14).** This is the primary source for the paper's Results and Limitations sections. Confidence level is stated explicitly per finding — some are well-established, some are candidate/exploratory, and that distinction is preserved throughout rather than flattened. §1-9 are the Day 1/2 record (preserved as originally written, with inline pointers where Day 3 work updates or resolves them); §10-14 are Day 3.

---

## 1. Infrastructure & Calibration

- **Model:** GLM-4.7-Flash (Z.ai/Zhipu, open-weight, 30B/~3B active MoE), single-model policy across agent, representation generation, and property scoring. Served via Z.ai's API, with Cloudflare Workers AI and OpenRouter as redundant fallbacks (same weights, no confound if switched).
- **Non-determinism at temperature=0 is real but not fully characterized.** Initial measurement: 8% flip rate (1/12 trials), concentrated in one `novel_tool` task (`bfcl_multiple_25`, a swap-similar-API decision). A follow-up recheck (15 more trials, including 3 more on that same task) found 0/15 flips. Combined: 1/6 total trials on that task flipped. **Honest status: inconclusive at this sample size** — not confirmed as a stable, reproducible instability tied to that task or decision type. Do not treat 8% as a settled noise floor in the paper; state the recheck result alongside it.
- **Grader required fixing twice**, both times found to be BFCL-fidelity issues, not the model's fault: notation normalization (`^` vs `**` in expressions) and correct handling of parameter fields that accept a list of valid values rather than one exact value.

---

## 2. The Ceiling Effect (full-scale baseline, 280/280 held-out tasks)

Aggregate baseline success: **91% (254/280)**. By shift_category:

| Category | Success | Hard (baseline-fail) |
|---|---|---|
| familiar | 91% | 7 |
| novel_tool | 93% (ceiling-flagged) | 4 |
| novel_composition | 89% | 9 |
| distractor_tool | 90% (ceiling-flagged) | 6 |

Only **26 hard tasks total across all 280.** Since Δ(E,T) ≤ 0 whenever baseline already succeeds, positive transfer is only observable on this thin subset — a real methodological constraint, confirmed at full scale, not a pilot artifact. No meaningful difficulty gradient was found across familiar/novel_tool/novel_composition (89-93%, statistically indistinguishable) — the shift taxonomy's assumed novelty ordering is not validated by this model's actual performance across these three categories. This finding directly motivated introducing **Δ_eff** (§6) as a second outcome variable not bound by the same ceiling.

---

## 3. Property Variance — Closed Investigation

Of the 9 experience properties, 5 (length, num_steps, success_failure, task_structure, tool_dependence) behave as expected. The remaining 4 (abstraction, specificity, composability, information_context) showed severe, persistent clustering across three independent remediation attempts:

1. Biased "be general/reusable" prompt → ceiling cluster (abstraction, specificity, composability all near 5/5).
2. Neutral prompt (removing "be general") → pole flipped to floor, not fixed (abstraction/composability collapsed to 1-2/5; specificity stayed at ceiling — got worse, not better).
3. Anchored 1/3/5 rubric re-rating → still clustered.

**Reconfirmed at 4x scale** when the experience bank grew 30→110: same pattern held (abstraction 60/80 floor, specificity 62/80 ceiling, composability 64/80 floor, information_context 55/80 ceiling). This is now a well-supported conclusion, not a small-sample artifact: **likely a genuine characteristic of BFCL's short, structurally homogeneous tool-use tasks bounding achievable variance on these 4 properties, not a fixable prompt/rubric issue.**

**Consequence:** these 4 properties are excluded from the primary predictor's feature set and treated as a secondary ablation, not included silently.

---

## 4. Coverage vs. Genuine Non-Transferability — Resolved

**Question:** the oracle-vs-retrieved gap analysis (25 hard tasks) found 64% (16/25) show no transfer even under oracle selection. Is this because the 30-experience bank lacks good candidates (coverage), or because experience genuinely doesn't help (transferability)?

**Test:** grew the bank 30→110 (3.7x), re-ran both oracle and retrieved selection on the identical 25 tasks.

**Result:** oracle 64%→68% no-transfer (+4pp), retrieved 68%→64% (−4pp). Combined: 34/50 → 33/50 — essentially no net change. **Neither condition showed the substantial improvement a real coverage effect would predict from 3.7x more candidates.** This is evidence against the coverage explanation and for genuine non-transferability being the primary driver of residual no-transfer on these hard tasks.

**Precise framing (not "two measurements confirmed each other"):** oracle's movement has a known, specific cause (see §5, not generic noise); retrieved's single-task movement is too small at n=25 to be more than suggestive on its own. The correct claim is "neither condition showed the coverage effect being tested for," not "independent triangulation."

---

## 5. Oracle Selection Rule: Non-Monotonicity (limitation)

The oracle condition uses a heuristic scoring rule (tool/family-overlap ranking), not a true empirical optimum. A genuine oracle should be monotonic — more candidates should never produce a worse pick. This one isn't: `bfcl_multiple_76`, originally used as the paper's "clean oracle-vs-retrieved divergence" illustration (oracle succeeds, retrieved fails, at bank=30), **regressed** at bank=110 — a different, tied-score experience won selection and failed where the original pick succeeded.

**Framing for the paper:** state this limitation explicitly, using this exact task as the illustration (the same task serving two different illustrative roles — divergence example, then non-monotonicity example — is itself worth noting for continuity). The correct interpretive claim, accounting for this: *"even acknowledging the oracle rule's documented non-monotonicity, the no-transfer rate did not improve with 3.7x more candidates, suggesting the limitation is more likely inherent to available experience content than to selection method."*

---

## 6. Main Transfer Results — Δ(E,T) (203 pairs / 103 unique tasks, bank=30)

Success rates: oracle 81%, retrieved 79%.

| Condition | Positive | Negative | Ambiguous (noise-floor) |
|---|---|---|---|
| oracle | 9 | 2 | 90 |
| retrieved | 8 | 4 | 90 |

Negative transfer is real at this volume — the earlier 12-pair pilot's "zero negative flips" result was correctly treated as provisional and did not hold at scale.

By shift_category (positive/negative/n): familiar 6/0/54, novel_composition 8/2/55, novel_tool 3/0/47 (noise-caution), **distractor_tool 0/4/47** — the one category with zero positive and all negative flips concentrated in it (see §7).

---

## 7. Headline Finding: `distractor_tool` Negative Transfer (statistically + mechanistically supported, with an honest limitation)

**Statistical evidence:** distractor_tool's negative-flip rate (4/47, 8.5%) vs. the pooled other three categories (2/156, 1.3%). Fisher's exact test: **p = 0.0267**, odds ratio 7.07, 95% CI [0.98, 80.6]. Crosses conventional α=0.05, but the wide CI (lower bound essentially at the null) means the effect size is not precisely estimated — report both numbers together, never the p-value alone. **Does not survive a Bonferroni correction for testing across 4 categories** (α/4 = 0.0125) — state this plainly. The test is motivated by a genuine a priori mechanistic hypothesis (distractor_tool structurally requires declining to call anything, unlike the other three categories which require correct selection/composition), not post-hoc cherry-picking of the best-looking bar — both facts belong in the writeup.

**Mechanistic verification (all 4 negative-flip trajectories read directly):**
- **4/4 confirm the behavioral pattern:** baseline correctly declined (empty call, gold_call null) in every case; injecting an experience caused the agent to call a tool it should have declined.
- **3/4 fit a clean causal story:** the injected experience itself modeled a successful tool call, and the agent appears to imitate the pattern "past example called a tool → I should too."
- **1/4 does not fit that story:** the oracle case on `bfcl_irrelevance_58` used an experience whose content was about correctly *declining* a tool call — yet the agent still over-called. This rules out "imitating a called-tool example" as the *complete* explanation. Notably, this same task failed under both oracle and retrieved regardless of injected content, suggesting `bfcl_irrelevance_58` specifically may be an independently fragile task (prone to over-triggering `solar_panel.calculate_need`) rather than purely a content-driven effect. **Follow-up resolved (Day 3):** baseline (no experience at all) succeeded on this task — empty predicted_calls, correct decline. The fragility is confirmed **experience-triggered specifically**, not a flaky-regardless-of-condition task. This strengthens rather than weakens the over-triggering mechanism's relevance, even for the one case that doesn't fit the "imitate a called-tool example" causal story.

**Bottom line for the paper:** the over-triggering mechanism is well-supported (3/4 direct causal evidence, 4/4 behavioral outcome evidence) but not fully general. Report the 1/4 exception plainly alongside the main finding — it strengthens rather than weakens the paper's credibility to show the mechanism was actually checked rather than assumed.

---

## 8. Efficiency Transfer — Δ_eff (73 matched-success pairs/condition, bank=30)

`retry_count` was found contaminated by shared API-infrastructure noise (identical retry deltas across conditions on the same task) and excluded from the aggregate. Effort metric = completion_tokens + steps + tool_call_count only.

- **Oracle:** mean +9.00 / median −8.00 completion-token delta. Sign disagreement between mean and median, confirmed at n=73 (not an n=4 pilot artifact) — a single favorable outlier masks a typical-case cost increase.
- **Retrieved:** mean −30.11 / median −10.00. Consistently negative both ways — retrieved-condition successes cost more completion tokens than baseline, typically and on average.

**Interpretation:** retrieval-selected experiences appear systematically less efficient than optimally-matched (oracle) ones — the efficiency analogue of the success/failure oracle-vs-retrieved gap in §4. **Open connection, not yet confirmed:** if oracle's median cost-increase pattern holds at further scale, it may represent a subtler form of negative transfer (real cost, no benefit) occurring inside the 91%-ceiling majority where Δ(E,T) structurally cannot show anything at all.

---

## 9. Day 2/3 Work (superseded — see §10-13 for final numbers)

Everything originally listed here as "not yet done" is now complete:

- Representation crossing → §10 below (Day 2 Step 1, 48 runs / 16 base pairs).
- Predictor v1 and v2 (both targets) → §11.
- Predictor-specific held-out split → frozen, `results/processed/predictor_split.json` (v1) and `predictor_split_v2.json` (v2, additive extension).
- Break-It-Down utility-score baseline → built; the flagged degeneracy risk was confirmed real (§11): only 4-5 unique utility-score values across the dataset, ~58% sharing the modal value. Reported plainly throughout rather than treated as a strong baseline to "beat."
- `bfcl_irrelevance_58` baseline-only check → resolved, see §7 (updated inline).

---

## 10. Day 3, Step 1 — Representation Crossing (48 runs, 16 base pairs × 3 representations)

**Scope, stated plainly:** 16 (task, oracle-selected experience) base pairs spanning all four `shift_category`s and both hard/easy, each run through `raw`/`reflection`/`procedure`. This is a deliberate subset, not exhaustive coverage of the (task × experience × representation) space — full-matrix testing was not affordable this week.

**Result:** 4/16 pairs (25%) changed success/fail across representations — a real effect, not just noise, at this sample size.

| task | raw | reflection | procedure |
|---|---|---|---|
| `bfcl_irrelevance_182` (hard) | FAIL | FAIL | **PASS** |
| `bfcl_simple_python_165` (hard) | FAIL | FAIL | **PASS** |
| `bfcl_simple_python_203` (hard) | PASS | **FAIL** | PASS |
| `bfcl_parallel_multiple_106` (easy) | PASS | **FAIL** (the only outright negative-transfer case in this subset) | PASS |

**Preliminary pattern (n=4 changed pairs — treat as suggestive, not conclusive):** `procedure` flips 2 failures to success and never uniquely hurts; `reflection` flips 2 successes to failure (including the one negative-transfer case) and never uniquely helps; `raw` is never the outlier in either direction. Consistent with a hypothesis that the more structured, step-by-step encoding transfers more reliably than the free-text lesson encoding in this domain — worth stating as a candidate finding for future work, not a settled result.

---

## 11. Day 3, Steps 1-3 — Predictor v1 vs. v2, Both Targets, Final Numbers

### 11.1 Feature set (unchanged from Day 2 design)

Primary: `shift_category`, `representation`, and the 5 reliable properties (`length`, `num_steps`, `success_failure`, `task_structure`, `tool_dependence`). Secondary ablation adds the 4 clustered properties (`abstraction`, `specificity`, `composability`, `information_context`) — per §3, never included silently.

### 11.2 Break-It-Down baseline degeneracy (checked, not assumed)

`utility_score = specificity × abstraction` takes only 4-5 unique values across the dataset (varies slightly by split), with ~57-58% of rows sharing the identical modal value. Not fully constant, but severely discretized — a direct consequence of specificity/abstraction clustering (§3). Beating this baseline is a lower bar than in a domain with more natural variance; stated explicitly throughout rather than presenting a win over it as equally meaningful.

### 11.3 Δ_eff (completion_tokens) — primary target

Two framings tried, per the Day 3 plan ("two specific, evidence-motivated attempts, not keep tuning until something works"):

**(a) Magnitude regression (original framing).** Loses to both baselines, both v1 (yesterday's split, n=35 held-out) and v2 (today's split, enriched with the distractor_tool oversample, n=49 held-out):

| version | predictor MAE | mean-Δ baseline MAE | Break-It-Down baseline MAE | beats either? |
|---|---|---|---|---|
| v1 | 105.24 | 69.67 | 75.90 | No |
| v2 (final) | 162.90 | 105.96 | 108.89 | No |

Confirmed not an overfitting artifact (train MAE ≈ held-out MAE, checked Day 2 Step 5) — genuine lack of learnable magnitude signal, most likely because the underlying completion-token deltas are fat-tailed/outlier-dominated (motivating framing (b) below).

**(b) Sign classification (Day 3 reframe — direction of effort change, not magnitude).** Motivation: oracle's mean/median completion-token deltas disagreeing in sign (§8) is direct evidence of a fat-tailed distribution, a bad fit for MAE-based regression but possibly fine for direction, which large outliers can't distort the same way. **This works, modestly but really:**

| version | feature set | predictor macro-F1 | best baseline macro-F1 | beats it? |
|---|---|---|---|---|
| v1 | primary | 0.319 | 0.271 | **Yes** |
| v1 | ablation | 0.367 | 0.271 | **Yes, more decisively** |
| v2 (final) | primary | 0.382 | 0.237 | **Yes** |
| v2 (final) | ablation | 0.394 | 0.237 | **Yes, more decisively** |

Loses on raw accuracy in every version (majority-class benefits mechanically from class imbalance — no exact-zero deltas occurred, so it's a binary problem where the majority sign dominates); wins consistently on macro-F1, the metric that actually accounts for that imbalance. The v2 gap is *wider* than v1's, not narrower — more data strengthened this result rather than diluting it.

**v2 is authoritative for the paper.** v1's numbers are preserved above for transparency about how the result evolved with more data, not as a competing claim.

### 11.4 Δ(E,T) success-flip — secondary target (small-n, hedged)

Enriched with the distractor_tool oversample specifically to fix thin held-out negative-class support (§12). **Did not meaningfully change the picture — loses to both baselines in both versions:**

| version | held-out negative-class support | predictor acc / F1 | majority-class acc / F1 | beats either? |
|---|---|---|---|---|
| v1 | 1 (of 51 rows) | 0.529 / 0.240 | 0.804 / 0.297 | No, both metrics |
| v2 (final) | 2 (of 66 rows) | 0.424 / 0.302 | 0.833 / 0.303 | No, both metrics (F1 now essentially tied) |

**Why enrichment barely moved this:** the v2 split stratifies by `shift_category` only, not by outcome label — of 27 new distractor_tool rows landing in held-out (the honest 80/20 split), only 1 of the 11 new negative cases happened to land there. A legitimate, unbiased outcome of the split methodology, not a bug — but it means this target's held-out set is still too thin for a fair predictor test. Report as **inconclusive/underpowered**, not as "predictor fails," even though the raw numbers look like a loss.

### 11.5 The hero claim — stated plainly, both attempts complete

**Does not hold in its main forms.** Neither Δ_eff magnitude regression nor Δ(E,T) ternary classification beats both baselines, confirmed today with fresh, enriched data — not merely carried over from yesterday's negative result.

**One genuine, narrower positive result:** Δ_eff **sign** classification (does experience make an already-solvable task cheaper or more expensive, directionally) beats both baselines on macro-F1, consistently, and more strongly with more data. Report this as its own distinct, narrower finding — "can we predict the direction of efficiency transfer" — not as a rescue of the primary hero claim ("can we predict transfer gain and its magnitude"), which those are not the same question.

### 11.6 Sanity check (Day 2 Step 6, still holds)

The Δ(E,T) classifier's feature importances remain coherent with known findings even though held-out performance is poor: `shift_category_distractor_tool` ranks 2nd overall; predicted P(Δ=-1) is elevated specifically for `retrieved`-condition distractor_tool rows (~34%) vs. `oracle`-condition rows (~2-4%), matching the actual asymmetry in the negative-flip data (§13.2). One clear outlier (`bfcl_irrelevance_93`) doesn't fit and is reported as such. The model learned real structure; that structure isn't sufficient to generalize reliably at this sample size.

---

## 12. Day 3, Step 2 — distractor_tool Oversampled: Significance Test Strengthened

**Motivation:** the original 25-hard-task gap analysis and Fisher's test (§7) came from a 30-experience bank and a thin category sample; the finding needed either confirmation or softening at real volume before it could be called confirmed.

**Action:** ran oracle + retrieved on the remaining 36 of 60 distractor_tool test tasks never before tested under either condition (24 already covered) — exhausting the entire category's test pool, so there's no ambiguity about whether more sampling would have helped further.

**Result: 11 new negative flips, ZERO new positive flips**, out of 72 new pairs. Combined with all prior distractor_tool data (including representation-crossing's distractor_tool rows): **15 negative / 0 positive out of 131 pairs (11.5%)**.

**Fisher's exact test, final (supersedes §7's number):**

| | original (§7) | final (Day 3) |
|---|---|---|
| distractor_tool rate | 4/47 (8.5%) | **15/131 (11.5%)** |
| pooled other 3 categories | 2/156 (1.3%) | **3/192 (1.6%)** |
| odds ratio | 7.07 | **8.15** |
| p-value (two-sided Fisher) | 0.0267 | **0.00024** |
| 95% CI on odds ratio | [0.98, 80.6] | **[2.22, 44.56]** |
| survives Bonferroni (4 categories, α/4=0.0125)? | **No** | **Yes** |

**This is now a confirmed finding, not a borderline one.** The CI no longer touches the null; the p-value clears Bonferroni correction with wide margin. The rate went *up* with more data (8.5%→11.5%), not down — more evidence, same direction, stronger conclusion.

---

## 13. Day 3, Step 4 — Robustness Checks

### 13.1 Experience leakage (checked directly, not assumed clean)

- **Pool-level gap found:** 13 exact goal-text duplicates exist across the frozen experience-generation/test splits (task_id-disjoint, but BFCL's templated task generation can produce byte-identical goal phrasing under different task IDs). 8 of these involve source tasks actually used in the 110-experience bank — a real risk, not hypothetical.
- **Checked against actual usage: zero contamination.** Of all 308 (task, experience) combinations actually run this week, zero were among the 8 at-risk leaked pairs (exact match), and zero showed near-verbatim overlap (SequenceMatcher ratio ≥0.8) either. **This week's results are not affected by experience leakage** — confirmed, not assumed.
- **Recommendation for future work:** the split-generation process should deduplicate by goal text, not just task_id, to close this gap structurally rather than relying on a post-hoc check each time.

### 13.2 Retrieval-vs-transfer-failure — consolidated final statement

Four independent lines of evidence, converging:

1. **Direct gap analysis** (25 hard tasks, canonical, unchanged by later re-runs): only 4% of oracle's successes were retrieval-attributable gaps (retrieved couldn't reach what oracle reached); the other 96% either succeeded under both conditions or neither.
2. **Bank-size sensitivity** (30→110 experiences): no-transfer rate did not improve in either condition beyond noise (oracle 64%→68%, retrieved 68%→64%, net 34/50→33/50) — rules against a coverage explanation.
3. **Efficiency gap** (now n=128/102 matched-success pairs): retrieved is the more consistently costly condition on both mean and median completion-token delta; oracle's mean/median sign-disagreement narrowed with more data but persists.
4. **Predictor sanity check**: independently recovered the same oracle/retrieved asymmetry from raw features alone (§11.6).

**One-sentence version for the paper:** *retrieval quality accounts for at most a small minority of observed transfer failure (≈4% of the hard-task gap-analysis cases, no measurable improvement from 3.7x more retrieval candidates); the dominant limiting factor is that the available experience content genuinely does not transfer to most hard held-out tasks, not that automatic retrieval fails to find good experiences that are present.*

**Documented complication:** the oracle selection rule is non-monotonic (§5) — its ceiling is itself a slight underestimate of the true achievable-with-perfect-selection rate. The bank-size result (line 2 above) is the stronger test of the coverage hypothesis regardless, since it doesn't depend on oracle's rule quality the way the static gap analysis does.

### 13.3 Final visualizations

`results/figures/`: `transfer_by_category.png`, `delta_eff_calibration.png` (magnitude regression scatter + sign-classification confusion matrix), `property_correlation.png`, `oracle_vs_retrieved_gap.png` (hard-task breakdown + efficiency comparison). All built from the complete, final enriched dataset (323 pairs) via `scripts/build_final_visualizations.py`.

**Process note, kept for transparency (not a finding, a pipeline lesson):** while testing the new one-command reproducibility runner, `scripts/run_analysis.py`'s default input path was found to still point at the tiny 24-pair Wednesday pilot file rather than the 203-pair Phase 1 dataset — a stale default from before `conditions_full.jsonl` became the real corpus. Running it with no argument silently regenerated `transfer_pairs.jsonl` with the wrong data, which briefly corrupted that test run's figures. Caught immediately (numbers didn't match the day's already-reported text, which came from direct calls with explicit correct paths and was never affected), fixed the default, and regenerated correctly. Recorded here as an example of exactly the kind of check the reproducibility pass exists to catch.

---

## 14. Protocol Frozen

**As of Day 3, the experimental design is frozen. No further post-hoc changes to conditions, splits, or feature sets.** Specifically:

- Task-level split (`tasks/split.json`) — frozen since Day 1, untouched throughout.
- Predictor-pair split (`predictor_split.json` v1, `predictor_split_v2.json` v2) — v2 is final; v1's assignments were never modified, only extended.
- Experimental conditions (baseline/oracle/retrieved) and the oracle selection rule — unchanged since Day 2, including its documented non-monotonicity limitation (§5), which is reported as-is rather than patched post-hoc.
- Retrieval method (MiniLM cosine similarity over `raw`) — fixed before evaluation began (Day 2), never tuned after seeing results.
- Feature set (primary 5 + `shift_category` + `representation`, clustered 4 as ablation-only) — final per §11.1.
- The two Day 3 predictor reframes (Δ_eff sign classification, distractor_tool oversampling) were each tried once, per an explicit evidence-motivated hypothesis, and reported exactly as they came out — one succeeded (§11.3b), one didn't move the target it was aimed at (§11.4) while strongly moving a different, related one (§12). Neither was iterated on after seeing its result.

Any further investigation from here is new work, not a continuation of this protocol, and should be reported as such.
