# Experiment Log

## 2026-08-27 — Property-rating ceiling/floor clustering (Step 6)

**Finding:** of the 9 experience properties, the 4 LLM-rated ones
(`abstraction`, `specificity`, `composability`, `information_context`) show
almost no variance across the 30 generated experiences — most cluster at
one extreme regardless of source task. The 5 rule-based properties
(`length`, `num_steps`, `success_failure`, `task_structure`,
`tool_dependence`) are unaffected and behave as expected.

**Three fix attempts, in order, none produced real discrimination:**
1. Original reflection/procedure prompts explicitly instructed "general,
   reusable, not just restate what happened" → abstraction/specificity/
   composability all clustered near ceiling (mean 4.7-5.0), only
   `information_context` was well-distributed (mean 2.47).
2. Removed that instruction, left the prompt neutral → clustering didn't
   resolve, it flipped poles (abstraction/composability collapsed toward
   floor instead of ceiling) and `information_context` — previously the
   one well-behaved property — collapsed to ceiling instead (29/30 at 5).
   Confirmed via direct before/after text comparison that the underlying
   reflection/procedure content DOES genuinely vary (old text avoided
   naming concrete entities, new text names them explicitly) — so the
   generation step isn't the bottleneck.
3. Rewrote the rating prompt with concrete anchored examples (1/3/5) per
   property, explicitly instructing the four axes to be judged
   independently rather than as one bundled "how concrete does this feel"
   judgment → still clustered (abstraction 21/27 at floor, specificity
   24/27 at ceiling, composability 23/27 at floor, information_context
   23/27 at ceiling on the first full pass; broadly consistent after
   patching 3 initially-null records).

**Working conclusion:** likely a genuine characteristic of this specific
experience pool (short, structurally similar BFCL AST tasks -- simple
single/few-call tool-use problems) rather than a fixable prompt-engineering
problem, at least not within today's time budget. Not pursuing a fourth
iteration.

**How this should be used downstream:** treat `abstraction`, `specificity`,
`composability`, and `information_context` as exploratory/secondary
features for the property-correlation analysis and predictor -- expect weak
or unreliable signal from them given the near-constant distribution. Lean
on the 5 rule-based properties (especially `task_structure` and
`tool_dependence`, which vary meaningfully and cheaply) plus `shift_category`
as the primary features. If revisited later: consider whether the 4-1-5
integer scale itself is too coarse for GLM-4.7-Flash to use discriminatively,
or whether a genuinely more diverse experience-generation task sample
(harder/more varied BFCL categories) would produce more natural variance
in the underlying reflections/procedures in the first place, rather than
continuing to iterate on the rating prompt alone.
