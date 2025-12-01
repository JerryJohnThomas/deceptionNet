> deceptionNet/config.py

  - Purpose: centralizes model dimensions, optimization settings, and runtime templates.
  - Contains ModelDims for shared sizes (players, roles, hidden dims).
  - OptimConfig, ILConfig, PPOConfig capture training hyperparameters.
  - TrainingConfig and InferenceConfig compose the above into ready-to-use bundles.
  - Presenter templates supply ASCII-safe sentence banks for the talk head.
  - Default instances (DEFAULT_TRAINING_CONFIG, DEFAULT_INFERENCE_CONFIG) exported.
  - Consumers: agent constructors and runners import to size tensors.
  - Inference config gives presenters consistent formatting (vote tokens, token caps).
  - Training config drives optimizer and loss weighting in runner_il and runner_ppo.
  - Role/phase conditioning pulls role counts and phase order from ModelDims.
  - Templates align with presenter logic so messages remain consistent in arena.
  - Acts as single source of truth for downstream modules during both IL and PPO stages.

  deceptionNet/agents/listener.py

  - Purpose: rule-based parser turning raw observations into structured signals.
  - Builds message lists from talk_history, optionally caching meta prefixes.
  - ListenerOutput dataclass holds embeddings and per-player rates (mentions, sentiment).
  - Uses crude heuristics to score accusations/support via cue dictionaries.
  - SimpleListener normalizes conversation text into fixed vectors for featurizer.
  - Alive/mentions counts feed suspicion features later in the pipeline.
  - Maintains base summary text for presenters to reference in talk output.
  - Called by IL loader and live agent to keep data flow consistent.
  - Works offline without external dependencies, ensuring reproducibility.
  - Provides conversation embedding by hashing bytes into fixed-size vector.
  - Integrates role/phase tags inserted as meta speaker entries.
  - Downstream modules (featurizer/state builder) consume the tensor outputs.

  deceptionNet/agents/listener_llm.py

  - Extends SimpleListener with optional local HuggingFace model summaries.
  - Lazily loads a configurable chat model (LLMListenerConfig) if available.
  - Generates JSON summaries (accuses/defends/claims/tone) per conversation.
  - Caches decoded summaries to avoid recomputation across identical inputs.
  - Converts JSON attributes into numeric tweaks on listener output (e.g., suspicion boosts).
  - Handles missing transformers/torch gracefully by falling back to base listener.
  - Prompt template stored as formatted string for consistent inference requests.
  - _safe_json_parse extracts JSON even when model emits surrounding text.
  - Device selection auto-detects CUDA when accessible.
  - Integrates seamlessly via MindGamesAgent flag --use-llm-listener.
  - Maintains same dataclass output so featurizer/state builder remain untouched.
  - Designed for offline usage, aligning with submission constraints.

  deceptionNet/agents/featurizer.py

  - Purpose: convert listener output plus observation metadata into tensors.
  - ObservationFeatures collects player/global features, masks, phase encodings.
  - Builds per-player stats (alive, votes, mentions, sentiment, accusations).
  - Encodes vote history into adjacency-like matrices for later graph ops.
  - Global features include round, turn, alive ratio, mafia estimate.
  - One-hot role history embedding reserved for known claims.
  - Phase encoding vector drives state builder attention branching.
  - Maintains talk_target validity masks for slot head decisions.
  - Called during IL batching and runtime inference before belief update.
  - Aligns with ModelDims to ensure consistent shapes across modules.
  - Supplies conversation embedding from listener directly.
  - Outputs flow into BeliefNet and StateBuilder for shared state creation.

  deceptionNet/agents/belief_net.py

  - Implements neural Bayes tracker that estimates roles/suspicion/trust.
  - BeliefState dataclass stores player embeddings, global hidden, logits.
  - Pipeline: encode player features, apply transformer, GRU for temporal memory.
  - Produces role logits (per player), suspicion/trust sigmoid outputs.
  - Initial state filled with zeros to bootstrap sequences.
  - Cross-attention handled in state builder, so belief focuses on per-step inference.
  - IL loss computes role CE, suspicion/trust BCE, calibration in runner.
  - Batch-first design matches featurizer outputs (B,N,D).
  - Alive mask used to weight mean pooling before GRU update.
  - Role logits feed IL loss while suspicion/trust inform state builder features.
  - Works both during supervised training and online inference.
  - Provides detach helpers to avoid gradient leakage during rollouts.

  deceptionNet/agents/state_builder.py

  - Combines featurizer output and belief embeddings into shared state S_t.
  - Includes graph message passing (optional layers) to propagate social signals.
  - Cross-attention injects conversation context into player embeddings.
  - GRU memory aggregates mean/sum pools for temporal continuity.
  - SharedState carries player embeddings, convo embedding, memory, masks.
  - Builds adjacency from vote matrices and mention scores.
  - Generates action masks (alive/vote/night/talk) based on role and self index.
  - Player fuser merges suspicion/trust scalars with belief embeddings.
  - Convo fuser blends global context plus conversation vector.
  - Underpins all policy heads by standardizing the shared observation encoding.
  - Ensures villagers vs mafia night actions get valid target masks.
  - Belief outputs stored for downstream loss computations in IL runner.

  deceptionNet/agents/heads.py

  - Declares policy/value heads for night, vote, talk intents/slots, and critic.
  - Each head is lightweight MLP→linear, using shared state features.
  - Night and vote heads respect masks via StateBuilder outputs.
  - Talk intent head maps convo+memory to discrete speech intents.
  - Slot head scores per-player targets when intent requires someone.
  - Value head compresses combined features for PPO baseline estimates.
  - Reused across IL and RL phases via MultiHeadPolicy.
  - MaskedHead logic moved to runner; heads focus on logits.
  - Design facilitates separate losses for each action stream.
  - Hidden size derived from config to keep architecture consistent.
  - Heads preserve dimensionality expectations for batching.
  - Value head returns scalar per batch (B) for advantage estimation.

  deceptionNet/agents/policy.py

  - MultiHeadPolicy wraps belief net, state builder, and heads.
  - Offers initial_state to start GRU memory and belief hidden states.
  - forward returns shared state plus logits for all action branches.
  - act samples masked actions, handling deterministic or stochastic modes.
  - Intent slot requirement handled with boolean mask intent_requires_target.
  - _sample_head applies masked softmax and fallback for invalid masks.
  - Exposes belief memory for storage in PPO buffer extra.
  - Provides consistent interface for IL (deterministic) and PPO evaluations.
  - Accepts optional prior belief/memory when stepping sequentially.
  - Returns logprob dictionary for PPO loss calculations.
  - Integrates seamlessly with presenters to convert actions to text/tokens.
  - Works under shared config dimensions to reuse across roles/phases.

  deceptionNet/agents/presenter.py

  - Responsible for formatting night/vote/talk actions into environment strings.
  - VotePresenter and NightPresenter enforce [index] output pattern.
  - TalkPresenter maps discrete intent/target to template sentences.
  - Template lists pulled from InferenceConfig presenter templates.
  - TalkAux dataclass passes optional belief summary or role claim context.
  - Templates trimmed/guarded to avoid exceeding token limits.
  - silence_token used when no message needed (intent 6).
  - TalkPresenter ensures output stays ASCII and single-paragraph.
  - Aligns with IL training output to keep behavior consistent.
  - NightPresenter fallback returns [0] when no valid target.
  - Maintains randomness through random.choice for variation.
  - Presenters invoked exclusively by online_agent to finalize actions.

  deceptionNet/agents/init.py

  - Aggregates public API for agent submodules.
  - Re-exports listener classes (SimpleListener, LLMListener) and outputs.
  - Exposes featurizer, belief net, state builder, policy heads, presenters.
  - Keeps module consumers from needing deep import paths.
  - Aligns with MindGamesAgent constructor imports.
  - Lists __all__ for clean wildcard usage.
  - Serves as adapter between package root and CLI.
  - Ensures new components (LLM listener) are discoverable.
  - Maintains ASCII-safe module metadata.
  - Updated when new agent components added.
  - Supports type hints across package imports.
  - No runtime logic beyond import abstraction.

  deceptionNet/runners/runner_il.py

  - Implements imitation learning training CLI and helper utilities.
  - ImitationLearner orchestrates optimizer, mixed precision, gradient steps.
  - ImitationBatch dataclass packages features and IL targets.
  - JSONL loader (load_jsonl_dataset) converts raw records into batches.
  - Collate logic stacks tensors, merging features for multi-sample mini batches.
  - Computes losses (role CE, suspicion/trust BCE, calibration, policy CE).
  - Tracks loss history, emitting PNG plots and CSV logs post-training.
  - Configurable via CLI flags (epochs, batch size, device, output path).
  - Integrates optional listener/featurizer pipeline for consistent tensors.
  - Supports default CPU training but respects config for CUDA.
  - Maintains random shuffle via DataLoader for dataset mixing.
  - Saves final policy state_dict to requested checkpoint path.

  deceptionNet/runners/runner_ppo.py

  - Provides PPO fine-tuning loop skeleton for future RL phase.
  - Consumes RolloutBuffer outputs and recomputes masked logprobs.
  - Handles advantage computation, clipping, entropy bonuses.
  - Uses mixed precision via GradScaler where available.
  - _compute_logprobs respects masks for night/vote/talk slots.
  - _to_device_* helpers move batched tensors to training device.
  - Tracks metrics (policy loss, value loss, entropy, KL, clip frac).
  - Requires prior rollout collection (not provided yet).
  - Designed to share MultiHeadPolicy to maintain parameter consistency.
  - Works with per-action masks to avoid invalid gradients.
  - Outputs average stats per update pass for tensorboard/logging.
  - Ready for integration once environment sampling pipeline is implemented.

  deceptionNet/runners/buffers.py

  - RolloutBuffer stores sequential observations/actions for PPO.
  - Supports stacking features, masks, values, rewards, dones.
  - Includes GAE advantage computation helper (advantages_and_returns).
  - Detaches tensors to CPU for later device transfers.
  - Maintains prev belief/memory snapshots per step.
  - Iterates minibatches via randomized index permutation.
  - RolloutTensors dataclass groups stacked feature tensors.
  - Provides safe serialization of learner state if needed.
  - Works closely with runner_ppo for training updates.
  - Enforces consistent shapes aligning with featurizer/policy outputs.
  - Stores logprobs per action head for ratio computation.
  - Designed for multi-env rollouts via num_envs parameter.

  deceptionNet/runners/init.py

  - Re-exports buffer, IL, and PPO runner utilities.
  - Provides consolidated import surface for training scripts.
  - Ensures default __all__ contains core training classes.
  - Keeps package structure tidy for future additions.
  - No runtime logic beyond module exposure.
  - Supports use in external scripts or notebooks.
  - Aligns with root deceptionNet.__init__ exports.
  - Encourages explicit imports vs deep module paths.
  - Maintains ASCII compliance.
  - Minimal but necessary for discoverability.
  - Updated when new runner utilities get added.
  - Complements training workflow documentation.

  deceptionNet/online_agent.py

  - CLI entrypoint instantiating MindGamesAgent for local or arena play.
  - Loads config, weights, listener choice, and presenters.
  - Provides --mode test for smoke runs with deterministic sequence.
  - --mode arena integrates with TextArena, handling runtime errors gracefully.
  - Accepts flags for model metadata, llm listener, weights path, device.
  - Ensures weights exist before loading to avoid runtime failures.
  - _run_test_mode prints phase/action history for quick debugging.
  - _run_arena_mode loops games, resetting state and catching env failures.
  - Uses DEFAULT_WEIGHTS_PATH pointing to latest IL checkpoint.
  - Works with TalkPresenter to format outputs per intent/slot.
  - Aligns listener/featurizer/policy data flow in live environment.
  - Forms the submission-ready agent once arena testing passes.