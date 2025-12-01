much tighter masterplan.md


perfect — here’s your **tight high-level TODO list for Stage 2**, distilled straight from that north-star plan, cleanly grouped and sequenced so Codex can expand it later.

---

## 🚀 Stage 2 High-Level TODO

### 🎯 0. North-Star Goals

* Maximize **win-rate** within token/time budget.
* Keep **reliability** (no invalid / stalled moves).
* Ensure **paper novelty** (replicable improvements, not just “bigger LLM”).

---

### 🧠 1. Architecture Upgrades

* [ ] Implement **LLM Listener v2**

  * Faster summarization + caching.
  * Add contradiction/bandwagon detection flags.
* [ ] Implement **Presenter v2**

  * Justification sentences (short rationale per vote/talk).
  * Strict regex guard for invalid or leaking text.

---

### 🧩 2. Training Pipeline

#### (a) Imitation Learning Refresh

* [ ] Expand dataset → ~1 000 synthetic samples (balanced by role & phase).
* [ ] Include contradiction / vote-switch / bandwagon scenes.
* [ ] Train 5–6 epochs → `weights-il-v6.pt`.

#### (b) PPO Fine-Tune

* [ ] Warm-start from `il-v6`.
* [ ] Run self-play + bot pool, alternating roles.
* [ ] Rewards:

  * +1 win / –1 loss
  * +0.1 legal move
  * +0.05 vote aligned with suspicion
  * –0.05 for inconsistent talk
* [ ] ~0.5–1.5 M steps → `weights-rl-v1.pt`.

#### (c) Consistency Regularizer

* [ ] Add small auxiliary loss penalizing vote–belief mismatch.
* [ ] Log correlation metric for paper figure.

---

### 🧬 3. Novelty Modules (pick 1–2)

* [ ] **Counterfactual Listener Update** → simulate alternate claims, compute stability.
* [ ] **Role-Privacy Auditor** → detect hidden-info leaks; measure leak vs win rate.
* [ ] **Calibration-Guided Voting** → add calibration head; enforce agreement between suspicion rank & vote.

---

### ⚙️ 4. Inference & Reliability

* [ ] Quantize LLM (≤ 3 B params, ≤ 50 ms per call).
* [ ] Cache listener summaries per round.
* [ ] Presenter: enforce ≤ 32 tokens + valid “[i]” formats.
* [ ] Timeout guard (1.5 s → safe fallback).
* [ ] Add per-game JSON logging (phase, role, intent, target, latency).

---

### 🕓 5. Execution Timeline

**Today → Tomorrow**

* Implement Listener v2 & Presenter v2.
* Generate 1 k IL v6 dataset; train.
* Optional PPO run → `rl-v1`.

**Oct 24–26 (Competition Window)**

* Play ≥ 50 games (target 70–100).
* Use same model token/identity.
* Pause 10–15 s between matches.

**After Window**

* Write report with:

  * IL / PPO loss curves
  * Win rate by role
  * Ablations (LLM listener on/off, reg on/off)

---

### ⚖️ 6. Training Knobs (Defaults)

| Parameter | IL   | PPO    |
| --------- | ---- | ------ |
| epochs    | 6    | —      |
| batch     | 8    | 64–128 |
| lr        | 3e-4 | 3e-5   |
| γ         | —    | 0.99   |
| clip      | —    | 0.2    |
| vf coef   | —    | 0.5    |
| ent coef  | —    | 0.01   |
| λ         | —    | 0.95   |

---

### 🧯 7. Fallback Plan

* If LLM unstable → ship `il-v6` only + consistency regularizer + richer template bank.

---

Would you like me to now break this into a **phase-by-phase actionable list** (files + function touchpoints) so Codex can directly start patching each part?
