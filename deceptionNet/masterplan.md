love it — fresh TrueSkill for Stage 2 = fresh start. here’s a crisp, high-level plan to go from your IL baseline → a competitive (and paper-worthy) Stage-2 agent.

# 0) North star (what we’re optimizing for)

* **Win rate under time + token limits.**
* **Reliability** (no invalid moves, no stalls).
* **Paper novelty**: clear, reproducible improvements beyond “we used a bigger LLM.”

# 1) Architecture upgrade (kept modular, minimal risk)

**Keep your current stack** (Listener → Featurizer → BeliefNet → StateBuilder → Multi-Head Policy → Presenter).
Add two *drop-in* upgrades:

1. **LLM Listener v2 (lightweight, fast)**

* Use a local chat model (e.g., **Qwen 1.8B/3B Instruct** or **Phi-3-mini**) quantized (int8/int4).
* Task: turn raw chat into a **structured summary** (claims, accuses/defends, vote intent, contradictions) + short rationale.
* Output = *small numeric deltas* that adjust your rule-based listener’s suspicion/trust (bounded, e.g., ±0.15) → **stable + fast**.

2. **Presenter v2 (justification mode)**

* Keep **discrete intent** and **[idx]** actions.
* Add a **one-sentence justification generator** (LLM) gated by:

  * hard token cap (≤ 32 tokens),
  * **no role leaks** (regex guard),
  * **phase-aware templates** (accuse / defend / consolidate / ask-why).
* If LLM fails → fall back to your current template bank.
  **Net effect**: higher persuasion without risking invalid output.

# 2) Training plan (48–72h prep)

**2.1 IL refresh (fast)**

* Expand to **1k synthetic samples** balanced by role (Mafia/Detective/Villager/Doctor) and phases.
* Include **contradiction & bandwagon** scenarios; encode vote-switch explanations.
* Train 5–6 epochs → `weights-il-v6.pt`.

**2.2 PPO fine-tune (targeted, short)**

* Warm-start from `il-v6`.
* **Self-play + bot pool**; alternate roles per episode.
* **Rewards** (small & stable):

  * +1 win / −1 loss,
  * +0.1 for legal move each phase (rule-adherence),
  * +0.05 if vote aligns with top-k suspicion from BeliefNet (calibration),
  * −0.05 for talk that triggers “inconsistency” (listener flag).
* Run ~0.5–1.5M steps, batch 64–128, lr ~3e-5, γ=0.99, clip=0.2 → `weights-rl-v1.pt`.

**2.3 Consistency regularizer (novelty + stability)**

* Add auxiliary loss that penalizes **action–belief mismatch** (e.g., voted target should be among high suspicion).
  This is cheap to implement and great paper material.

# 3) Paper-worthy novelty (pick 1–2 you can finish)

1. **Counterfactual Listener Update**
   Before voting, simulate *one alternate claim* (e.g., “what if P3 is villager?”) via the LLM listener and see if suspicion flips. Add a **counterfactual-stability score** and bias policy toward robust choices.
   *Novel, measurable, small diff to code.*

2. **Role-Privacy Auditor**
   A guard that blocks presenter text if it implicitly leaks hidden info (regex + small classifier). Report **leak rate → win trade-off**.
   *Good for a safety/ethics angle + reproducible.*

3. **Calibration-guided Voting**
   Train a scalar **calibration head**; encourage agreement between suspicion ranking and final votes with a temperature parameter.
   *Simple, effective, easy ablation.*

# 4) Inference & reliability

* **LLM listener**: quantized 1.8B–3B, max 30–50 ms CPU/GPU per call; cache summaries per round.
* **Presenter**: hard length cap; regex assert for valid “[i]”.
* **Timeout guard**: if no action in 1.5s → safe fallback (existing templates).
* **Logging**: per-game JSON (phase, role, intent, target, validity, latency). You’ll reuse this in the report.

# 5) Execution timeline (concise)

**Today / Tomorrow**

* Plug **LLM Listener v2** (bounded deltas) + **Presenter v2** (justifications + guards).
* Generate **1k IL v6**; train → `il-v6`.
* Quick PPO run → `rl-v1` (if time permits).

**Stage-2 window (Oct 24–26, ET)**

* Use **one identity** (same name/token).
* Run in blocks of 10–15 games with 10–15s gaps to avoid socket stalls.
* Target **≥ 50 games**; aim for 70–100 if compute is free.

**After window (Report)**

* Figures: IL/ PPO loss curves, latency histogram, win rate by role, ablations (with/without LLM listener; with/without calibration regularizer).
* Sections: Method; Listener v2; Consistency regularizer; Experiments; Limitations.

# 6) Concrete knobs (so you don’t have to think later)

* **Models**: `Qwen2.5-1.8B-Instruct` (or Phi-3-mini) → int8; max_new_tokens=40; temperature=0.2.
* **IL**: epochs=6, batch=8, lr=3e-4.
* **PPO**: lr=3e-5, clip=0.2, vf_coef=0.5, ent_coef=0.01, gae_λ=0.95, updates≈400–800.
* **Guards**: talk ≤ 32 tokens; reject if contains “I am Mafia/Doctor/Detective”.

# 7) Fallback plan (zero-risk path)

If LLM isn’t stable: ship **IL-v6 only** with the **consistency regularizer** and a slightly richer template bank. Your current stack already proved reliable; this adds real lift without fragility.

---

If you want, I can turn this into a **tight TODO list with exact function/FILE touchpoints** (e.g., which `listener_llm.py` method to change, the mask asserts in `presenter.py`, the auxiliary loss in `runner_il.py/runner_ppo.py`).
