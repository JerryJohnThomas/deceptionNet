"""High-level policy that wires the torso, heads, and action sampling."""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.distributions import Categorical

from deceptionNet.config import ModelDims
from deceptionNet.datatypes import PolicyOutputs
from deceptionNet.utils import masked_log_softmax, masked_softmax

from .belief_net import BeliefNet, BeliefState
from .featurizer import ObservationFeatures
from .heads import NightPolicyHead, TalkIntentHead, TalkSlotHead, ValueHead, VotePolicyHead
from .state_builder import SharedState, StateBuilder


class MultiHeadPolicy(nn.Module):
    """Joint policy module driving night, talk, and vote behaviour."""

    # Input Features
    #     ↓
    # ┌─────────────┐
    # │   TORSO     │  ← Shared processing
    # ├─────────────┤
    # │ BeliefNet   │  "What do I believe about players?"
    # │     +       │
    # │StateBuilder │  "Combine everything into rich state"
    # └──────┬──────┘
    #     │ shared representation
    #     ├─────────────┬─────────────┬──────────────┬───────────┐
    #     ↓             ↓             ↓              ↓           ↓
    # ┌────────┐   ┌─────────┐   ┌────────┐   ┌──────────┐  ┌───────┐
    # │ Night  │   │  Vote   │   │ Intent │   │   Slot   │  │ Value │
    # │  Head  │   │  Head   │   │  Head  │   │   Head   │  │ Head  │
    # └────────┘   └─────────┘   └────────┘   └──────────┘  └───────┘
    #     ↓             ↓             ↓              ↓           ↓
    # Night Action  Vote Action  Talk Type    Talk Target   State Value

    def __init__(self, dims: ModelDims):
        super().__init__()
        self.dims = dims
        self.belief_net = BeliefNet(dims)
        self.state_builder = StateBuilder(dims)
        self.night_head = NightPolicyHead(dims)
        self.vote_head = VotePolicyHead(dims)
        self.intent_head = TalkIntentHead(dims)
        self.slot_head = TalkSlotHead(dims)
        self.value_head = ValueHead(dims)

        # First 6 talk intents (0-5) require specifying a target player
        # Examples:
        # Intent 0: "Accuse [player]" ← needs target ✅
        # Intent 1: "Defend [player]" ← needs target ✅
        # Intent 6: "Stay silent" ← no target ❌

        self.intent_requires_target = set(range(min(6, dims.num_talk_intents)))

    def initial_state(self, batch_size: int, device: Optional[torch.device] = None) -> Tuple[BeliefState, torch.Tensor]:
        # Initialize hidden states at the start of an episode
        device = device or torch.device("cpu")
        belief = self.belief_net.initial_state(batch_size, device)
        memory = torch.zeros(batch_size, self.dims.hidden_size, device=device)
        return belief, memory

    def forward(
        self,
        features: ObservationFeatures,
        belief_state: Optional[BeliefState] = None,
        memory_state: Optional[torch.Tensor] = None,
    ) -> Tuple[SharedState, PolicyOutputs]:
        # state: belief + features + memory 
        # policy : state --> logits

        # Belief module updates hidden representation of the world
        belief = self.belief_net(features, belief_state)

        #  State builder fuses features + belief + memory → single latent vector
        shared = self.state_builder(features, belief, prev_memory=memory_state)

        # Action Heads / Policies : state --> action probabilities
        night_logits = self.night_head(shared)
        vote_logits = self.vote_head(shared)
        intent_logits = self.intent_head(shared)
        slot_logits = self.slot_head(shared)
        values = self.value_head(shared)

        outputs = PolicyOutputs(
            night_logits=night_logits,
            vote_logits=vote_logits,
            talk_intent_logits=intent_logits,
            talk_slot_logits=slot_logits,
            values=values,
            extra={"memory_state": shared.memory_state},
        )
        return shared, outputs

    def act(
        self,
        outputs: PolicyOutputs,
        masks,
        deterministic: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        
        # Action Sampler from logits/Policy Output
        # converts logits into actual sampled actions, optionally using masks to rule out illegal moves
        # at​∼πθ​(a∣st​)

        # All below comments i am not sure

        # Input:
        # outputs - Raw logits from all heads (unnormalized scores)
        # masks - Which actions are valid (can't target dead players!)
        # deterministic - Sample randomly or pick best action?

        # Output:
        # actions - Actual action indices chosen
        # logprobs - Log probabilities of those actions (for RL training)

        #  Sample output 
        # actions = {
        #     "night": tensor([2]),      # Kill player 2
        #     "vote": tensor([3]),       # Vote player 3
        #     "talk_intent": tensor([0]), # Accuse
        #     "talk_target": tensor([3])  # Accuse player 3
        # }

        # logprobs = {
        #     "night": tensor([-1.2]),
        #     "vote": tensor([-0.8]),
        #     "talk_intent": tensor([-0.5]),
        #     "talk_target": tensor([-1.1])
        # }

        actions: Dict[str, torch.Tensor] = {}
        logprobs: Dict[str, torch.Tensor] = {}

        # Night actions (Conditional)
        # only happen if you're mafia/have night role!
        # night_mask = [0, 1, 1, 0]  # Can target players 1 and 2, hence sum > 0 means its possible to attack
        night_mask = masks.night_valid 
        if night_mask.sum() > 0:
            night_action, night_logprob = self._sample_head(outputs.night_logits, night_mask, deterministic)    # Sampling an action 
            actions["night"] = night_action
            logprobs["night"] = night_logprob
        else:
            # Return invalid action marker
            batch_size = outputs.vote_logits.shape[0]
            actions["night"] = torch.full((batch_size,), -1, dtype=torch.long, device=outputs.vote_logits.device)       # actions["night"] = -1  # -1 means "no action"
            logprobs["night"] = torch.zeros(batch_size, dtype=torch.float32, device=outputs.vote_logits.device)         # logprobs["night"] = 0.0  # Zero log prob (no action taken)

        # Vote Action
        # mask is for who all can be voted for, valid targets 
        vote_action, vote_logprob = self._sample_head(outputs.vote_logits, masks.vote_valid, deterministic)
        actions["vote"] = vote_action
        logprobs["vote"] = vote_logprob

        # Talk Intent, no masks, all intents always valid
        intent_dist = self._intent_distribution(outputs.talk_intent_logits, deterministic)
        intent_action = intent_dist.sample() if not deterministic else intent_dist.probs.argmax(dim=-1)
        actions["talk_intent"] = intent_action
        logprobs["talk_intent"] = intent_dist.log_prob(intent_action)

        # Talk Target 
        slot_mask = masks.talk_target_valid

        # Check which intents need a target
        requires_slot_bool = torch.tensor([bool(int(intent.item()) in self.intent_requires_target) for intent in intent_action], device=slot_mask.device)

        #  Sample target IF needed
        if requires_slot_bool.any():
            slot_action, slot_logprob = self._sample_head(outputs.talk_slot_logits, slot_mask, deterministic)
            slot_action = torch.where(requires_slot_bool, slot_action, torch.full_like(slot_action, -1))
            slot_logprob = torch.where(requires_slot_bool, slot_logprob, torch.zeros_like(slot_logprob))
            actions["talk_target"] = slot_action
            logprobs["talk_target"] = slot_logprob
        else:       # No one needs target (edge case)
            actions["talk_target"] = torch.full_like(intent_action, -1)
            logprobs["talk_target"] = torch.zeros_like(intent_action, dtype=torch.float32)

        return actions, logprobs

    def _sample_head(
        self,
        logits: torch.Tensor,
        mask: torch.Tensor,
        deterministic: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Add on snippet for logits and probability
        # Logits are the raw, unbounded output scores from a model before normalization, while probabilities are normalized values between 0 and 1) that represent the likelihood of an event
        # Logits are used during training for numerical stability and faster convergence, whereas probabilities are used for interpretation
        # Softmax Function (exponential normalisation): Converting Logits → Probabilities

        # what happens here is forward will use the corresponding head and give you the ouput in terms of logits
        # logitis --> mask invalid --> probabilities --> distribution--> 
        # sample an action (if schoastic) --> verify if valid action --> Compute log probability

        mask = mask.to(logits.dtype)    # converting bool and int also to same type: float 
        valid = mask.sum(dim=-1) > 0    # any valid action. valid action has 1 in that action_id slot if valid 
        safe_mask = torch.where(valid.unsqueeze(-1), mask, torch.ones_like(mask)) # eplace empty masks with all-1s temporarily
        probs = masked_softmax(logits, safe_mask)
        dist = Categorical(probs=probs)
        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            action = dist.sample()
        logprob = dist.log_prob(action)
        action = torch.where(valid, action, torch.full_like(action, -1))
        logprob = torch.where(valid, logprob, torch.zeros_like(logprob))
        return action, logprob

    def _intent_distribution(self, logits: torch.Tensor, deterministic: bool) -> Categorical:
        # Standard categorical over intents; we rely on guardrails externally for invalid combos.
        probs = torch.softmax(logits, dim=-1)
        return Categorical(probs=probs)


__all__ = ["MultiHeadPolicy"]
