"""Interactive human play session utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from deceptionNet.config import DEFAULT_INFERENCE_CONFIG, InferenceConfig
from deceptionNet.agents.presenter import VotePresenter, NightPresenter
from deceptionNet.agents.text_to_state_mapper import TextToStateMapper
from deceptionNet.utils import debug_log


HumanInputFunc = Callable[[str], str]


class SimpleSelfPlayEnv:
    """Minimal self-play environment used for rollout collection or human play."""

    def __init__(self, num_players: int = 6, episode_length: int = 60) -> None:
        self.num_players = num_players
        self.episode_length = max(1, episode_length)
        self._phase_cycle = ['day_talk', 'day_vote']
        self.reset()

    def reset(self) -> tuple[int, Dict[str, Any]]:
        self.turn = 0
        self.round = 1
        self.phase_index = 0
        self.current_player = random.randrange(self.num_players) if self.num_players else 0
        self.alive = [True] * self.num_players
        self.talk_history: List[Dict[str, Any]] = []
        self.vote_history: List[Dict[str, Any]] = []
        self._recent_text = ''
        return self.get_observation()

    def get_observation(self) -> tuple[int, Dict[str, Any]]:
        state = {
            'phase': self._phase_cycle[self.phase_index],
            'round': self.round,
            'turn': self.turn,
            'self_player': self.current_player,
            'role': 'villager',
            'alive': [1 if alive else 0 for alive in self.alive],
            'talk_history': list(self.talk_history[-12:]),
            'vote_history': list(self.vote_history[-12:]),
            'recent_text': self._recent_text,
        }
        return self.current_player, state

    def step(self, player_id: int, action_text: str) -> tuple[bool, Dict[str, Any]]:
        if self._phase_cycle[self.phase_index] == 'day_vote':
            target = _extract_vote_target(action_text)
            if target is not None:
                self.vote_history.append({'voter': player_id, 'target': target})
                self._recent_text = f"Player {player_id} voted [{target}]"
            else:
                self._recent_text = action_text
        else:
            self.talk_history.append({'speaker': player_id, 'text': action_text})
            self._recent_text = action_text

        self.turn += 1
        done = self.turn >= self.episode_length
        info: Dict[str, Any] = {'turn': self.turn}

        if not done:
            if self.turn % self.num_players == 0:
                self.phase_index = (self.phase_index + 1) % len(self._phase_cycle)
                if self.phase_index == 0:
                    self.round += 1

        self.current_player = (player_id + 1) % self.num_players
        return done, info

    def close_episode(self) -> tuple[List[float], List[Dict[str, Any]]]:
        rewards = [0.0] * self.num_players
        info = [{} for _ in range(self.num_players)]
        return rewards, info


class TextArenaOfflineWrapper:
    """Adapter around the SecretMafia TextArena environment for offline rollouts."""

    def __init__(self, num_players: int = 6) -> None:
        self.num_players = num_players
        self._env = None
        self._mappers = [TextToStateMapper(num_players=num_players, self_player=i) for i in range(num_players)]

    def _ensure_env(self) -> None:
        if self._env is None:
            try:
                import textarena as ta  # type: ignore
            except ImportError as exc:  # pragma: no cover
                raise ImportError("textarena package is required for --env-source textarena") from exc
            self._env = ta.make("SecretMafia-v0")

    def reset(self) -> tuple[int, Dict[str, Any]]:
        self._ensure_env()
        for mapper in self._mappers:
            mapper.reset()
        try:
            self._env.reset(num_players=self.num_players)
        except TypeError:
            self._env.reset()
        return self.get_observation()

    def get_observation(self) -> tuple[int, Dict[str, Any]]:
        self._ensure_env()
        pid, raw_obs = self._env.get_observation()
        print("[HUME]", raw_obs,"[:HUME]")
        mapper = self._mappers[pid % self.num_players]
        state = mapper.update(raw_obs)
        return pid, state

    def step(self, player_id: int, action_text: str) -> tuple[bool, Dict[str, Any]]:
        _ = player_id
        self._ensure_env()
        done, info = self._env.step(action=action_text)
        return done, info or {}

    def close_episode(self) -> tuple[List[float], List[Dict[str, Any]]]:
        if self._env is None:
            return [0.0] * self.num_players, [{} for _ in range(self.num_players)]
        rewards = None
        info = None
        try:
            rewards, info = self._env.close()
        except AttributeError:
            rewards, info = None, None
        except Exception:
            rewards, info = None, None
        finally:
            self._env = None
        if rewards is None:
            rewards = [0.0] * self.num_players
        if info is None:
            info = [{} for _ in range(self.num_players)]
        return list(rewards), list(info)


def run_human_play_session(
    env: Any,
    agents: Dict[int, Any],
    human_player: int,
    *,
    inference_config: InferenceConfig = DEFAULT_INFERENCE_CONFIG,
    debug: bool = False,
    debug_log_path: Optional[Path] = None,
    transcript_path: Optional[Path] = None,
    input_func: HumanInputFunc = input,
    max_steps: Optional[int] = None,
) -> 'HumanPlayResult':
    """Drive a human-vs-agent game loop until completion or interrupt."""

    if transcript_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        transcript_path = Path("logs") / f"human_play_{timestamp}.txt"
    transcript_path.parent.mkdir(parents=True, exist_ok=True)

    if debug and debug_log_path is None:
        debug_log_path = Path("logs") / f"debug_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    if debug_log_path is not None:
        debug_log_path.parent.mkdir(parents=True, exist_ok=True)

    vote_presenter = VotePresenter(inference_config)
    night_presenter = NightPresenter(inference_config)
    silence_token = inference_config.silence_token

    transcript_lines: List[str] = []
    total_steps = 0

    try:
        player_id, observation = env.reset()
    except TypeError:
        env.reset()
        player_id, observation = env.get_observation()

    for agent in agents.values():
        agent.reset_state()

    if debug and debug_log_path is not None:
        debug_log("=== Human play session started ===", path=debug_log_path)

    try:
        done = False
        while not done and (max_steps is None or total_steps < max_steps):
            phase = str(observation.get("phase", "?")).lower()
            _display_state(observation, human_player)

            if player_id == human_player:
                try:
                    message, action_desc = _prompt_human_action(
                        phase,
                        observation,
                        input_func,
                        vote_presenter,
                        night_presenter,
                        silence_token,
                    )
                except KeyboardInterrupt:
                    raise
                except Exception as exc:
                    print(f"[WARN] Invalid input: {exc}. Please try again.")
                    continue

                transcript_lines.append(f"[HUMAN] Player {human_player}: {message}")
                if debug and debug_log_path is not None:
                    debug_log(f"[HUMAN] Player {human_player} action: {action_desc}", path=debug_log_path)
                print(f"[You] -> {message}")
                done, info = env.step(player_id, message)
            else:
                agent = agents[player_id]
                details = agent.act_with_details(observation, deterministic=False, log=False)
                message = details["message"]
                transcript_lines.append(f"[AGENT] Player {player_id}: {message}")
                print(f"[Player {player_id}] -> {message}")
                if debug and debug_log_path is not None:
                    debug_log(
                        f"[AGENT] Player {player_id} action: {details['action_info']} message={message}",
                        path=debug_log_path,
                    )
                done, info = env.step(player_id, message)

            total_steps += 1

            if done:
                break

            try:
                player_id, observation = env.get_observation()
            except Exception:
                break

    except KeyboardInterrupt:
        print("\n[GAME] Session interrupted by user.")
        transcript_lines.append("[SYSTEM] Session interrupted by user.")
        done = True

    rewards = None
    info = None
    close_fn = getattr(env, "close_episode", None)
    if callable(close_fn):
        try:
            rewards, info = close_fn()
        except Exception:
            rewards, info = None, None

    if rewards is not None:
        try:
            summary = ", ".join(f"P{i}:{float(r):+.2f}" for i, r in enumerate(rewards))
            print(f"[GAME] Final rewards -> {summary}")
            transcript_lines.append(f"[SYSTEM] Final rewards -> {summary}")
        except Exception:
            pass

    transcript_path.write_text("\n".join(transcript_lines) + "\n", encoding="utf-8")
    if debug and debug_log_path is not None:
        debug_log("=== Human play session ended ===", path=debug_log_path)
    return HumanPlayResult(transcript_path=transcript_path, debug_log_path=debug_log_path, total_steps=total_steps)


@dataclass
class HumanPlayResult:
    transcript_path: Path
    debug_log_path: Optional[Path]
    total_steps: int


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _display_state(observation: Dict[str, Any], human_player: int) -> None:
    phase = observation.get("phase", "?")
    round_idx = observation.get("round")
    turn = observation.get("turn")
    alive = observation.get("alive")
    if isinstance(alive, list):
        alive_players = ", ".join(str(idx) for idx, flag in enumerate(alive) if flag)
    else:
        alive_players = "?"
    print(f"\n[GAME] Phase={phase} | Round={round_idx} | Turn={turn} | Alive: {alive_players}")

    talk_history = observation.get("talk_history") or []
    if talk_history:
        print("[GAME] Recent talk:")
        for entry in talk_history[-3:]:
            speaker = entry.get("speaker", "?")
            text = entry.get("text", "")
            print(f"  Player {speaker}: {text}")

    vote_history = observation.get("vote_history") or []
    if vote_history:
        last_votes = vote_history[-3:]
        printable = ", ".join(f"{item.get('voter')}?{item.get('target')}" for item in last_votes)
        print(f"[GAME] Recent votes: {printable}")


def _prompt_human_action(
    phase: str,
    observation: Dict[str, Any],
    input_func: HumanInputFunc,
    vote_presenter: VotePresenter,
    night_presenter: NightPresenter,
    silence_token: str,
) -> tuple[str, str]:
    phase = phase.lower()
    alive = observation.get("alive")
    valid_targets = [idx for idx, flag in enumerate(alive) if flag] if isinstance(alive, list) else []

    if phase == "day_vote":
        prompt = f"[You] Vote target among {valid_targets} (or 'skip'): "
        while True:
            response = input_func(prompt).strip().lower()
            if response in ("exit", "quit"):
                raise KeyboardInterrupt()
            if response in ("skip", "pass", ""):
                target = -1
                break
            try:
                target = int(response)
                break
            except ValueError:
                print("Please enter a player index or 'skip'.")
        message = vote_presenter.render(target)
        return message, f"vote->{target}"

    if phase == "night":
        prompt = f"[You] Night action target among {valid_targets} (or 'skip'): "
        while True:
            response = input_func(prompt).strip().lower()
            if response in ("exit", "quit"):
                raise KeyboardInterrupt()
            if response in ("skip", "pass", ""):
                target = -1
                break
            try:
                target = int(response)
                break
            except ValueError:
                print("Please enter a player index or 'skip'.")
        message = night_presenter.render(target)
        return message, f"night->{target}"

    response = input_func("[You] Say something (blank = silence, 'quit' exits): ").strip()
    if response.lower() in ("exit", "quit"):
        raise KeyboardInterrupt()
    if not response:
        response = silence_token
    return response, "talk"


def _extract_vote_target(action_text: str) -> Optional[int]:
    digits = ''.join(ch for ch in action_text if ch.isdigit())
    if not digits:
        return None
    try:
        return int(digits[-1])
    except ValueError:
        return None


__all__ = [
    "SimpleSelfPlayEnv",
    "TextArenaOfflineWrapper",
    "run_human_play_session",
    "HumanPlayResult",
]
