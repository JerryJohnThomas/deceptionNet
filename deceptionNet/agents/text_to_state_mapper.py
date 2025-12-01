
"""Utilities to convert raw TextArena strings into structured state dicts."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


_PHASE_PATTERNS = {
    "night": [r"night phase", r"night begins", r"night action", r"night"],
    "day_vote": [r"voting phase", r"vote phase", r"time to vote", r"day voting", r"cast your vote"],
    "day_talk": [r"discussion phase", r"day phase", r"day talk", r"discussion continues", r"day \d+ begins", r"day begins",r"day breaks", r"discuss for",],
}

_ELIM_PATTERN = re.compile(r"player\s*(\d+)\s+(?:was|has been)\s+(?:eliminated|killed)", re.IGNORECASE)
_ROLE_PATTERN = re.compile(r"your role[:\s]+([A-Za-z]+)", re.IGNORECASE)
_YOU_ARE_PATTERN = re.compile(r"you are(?: the)?\s+([A-Za-z]+)", re.IGNORECASE)
_TALK_PATTERN = re.compile(r"^\s*player\s*(\d+)\s*[:\-]\s*(.+)$", re.IGNORECASE)
_VOTE_PATTERN = re.compile(r"player\s*(\d+)\s+(?:votes?|voted)\s*(?:for\s*)?\[?(\d+)\]?", re.IGNORECASE)
_ROUND_DAY_PATTERN = re.compile(r"day\s*(\d+)", re.IGNORECASE)
_ROUND_NIGHT_PATTERN = re.compile(r"night\s*(\d+)", re.IGNORECASE)
_PLAYER_ID_PATTERN = re.compile(r"player\s*(\d+)", re.IGNORECASE)


@dataclass
class TextToStateMapper:
    """Maintains lightweight state extracted from raw text observations."""

    num_players: int = 6
    self_player: int = 0
    max_history: int = 200
    _initial_self_player: int = field(init=False, repr=False)
    _alive: List[bool] = field(init=False, repr=False)
    _role: str = field(init=False, repr=False)
    _phase: str = field(init=False, repr=False)
    _round: int = field(init=False, repr=False)
    _turn: int = field(init=False, repr=False)
    _last_text: str = field(init=False, repr=False)
    _talk_history: List[Dict[str, Any]] = field(init=False, repr=False)
    _vote_history: List[Dict[str, Any]] = field(init=False, repr=False)
    _raw_history: List[str] = field(init=False, repr=False)
    _override_state: Optional[Dict[str, Any]] = field(init=False, repr=False)
    _last_reset: bool = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._initial_self_player = self.self_player
        self.reset()

    # ------------------------------------------------------------------
    def reset(self) -> None:
        self.self_player = getattr(self, "_initial_self_player", self.self_player)
        self._alive = [True] * self.num_players
        self._role = "unknown"
        self._phase = "day_talk"
        self._round = 1
        self._turn = 0
        self._last_text = ""
        self._talk_history = []
        self._vote_history = []
        self._raw_history = []
        self._override_state = None
        self._last_reset = False

    # ------------------------------------------------------------------
    def update(self, observation: Any) -> Dict[str, Any]:
        """Ingest a raw environment observation and update cached state."""

        self._last_reset = False
        if isinstance(observation, dict):
            if observation.get("reset"):
                self.reset()
                self._last_reset = True
            self._override_state = observation.copy()
            if "self_player" in observation:
                try:
                    self.self_player = int(observation.get("self_player", self.self_player))
                except (TypeError, ValueError):
                    pass
            self._phase = observation.get("phase", self._phase)
            alive = observation.get("alive")
            if isinstance(alive, list) and len(alive) == self.num_players:
                self._alive = [bool(x) for x in alive]
            self._role = observation.get("role", self._role)
            self._round = int(observation.get("round", self._round))
            self._turn = int(observation.get("turn", self._turn))
            if "talk_history" in observation:
                self._talk_history = list(observation.get("talk_history") or [])[-self.max_history:]
            if "vote_history" in observation:
                self._vote_history = list(observation.get("vote_history") or [])[-self.max_history:]
            recent = observation.get("recent_text") or observation.get("current_message")
            if isinstance(recent, dict):
                self._last_text = str(recent.get("text", ""))
            elif isinstance(recent, str):
                self._last_text = recent
            return self.get_state()

        # string observation
        text = str(observation).strip()
        if not text:
            return self.get_state()
        self._override_state = None
        self._last_text = text
        self._raw_history.append(text)
        if len(self._raw_history) > self.max_history:
            self._raw_history.pop(0)

        lowered = text.lower()
        self._parse_phase(lowered)
        self._parse_round(lowered)
        self._parse_role(lowered)
        self._parse_elimination(lowered)
        self._parse_vote(lowered)
        self._parse_talk(text)

        return self.get_state()

    # ------------------------------------------------------------------
    def get_state(self) -> Dict[str, Any]:
        """Return the latest structured state dictionary."""

        state: Dict[str, Any]
        if self._override_state is not None:
            state = {k: v for k, v in self._override_state.items()}
        else:
            state = {}

        state.setdefault("phase", self._phase)
        state.setdefault("alive", [1 if x else 0 for x in self._alive])
        state.setdefault("role", self._role)
        state.setdefault("self_player", self.self_player)
        state.setdefault("round", self._round)
        state.setdefault("turn", self._turn)
        state.setdefault("talk_history", list(self._talk_history))
        state.setdefault("vote_history", list(self._vote_history))
        state.setdefault("recent_text", self._last_text)
        state.setdefault("raw_history", list(self._raw_history))
        state.setdefault("known_mafia", state.get("known_mafia", []))
        if self._last_reset:
            state["reset"] = True
        return state

    # ------------------------------------------------------------------
    def _parse_phase_not_working(self, lowered: str) -> None:
        print("[OSCAR] start:", lowered , "[::OSCAR]")
        for phase_name, patterns in _PHASE_PATTERNS.items():
            if any(re.search(pat, lowered) for pat in patterns):
                print("[OSCAR] match:", lowered, phase_name, "[::OSCAR]")
                if phase_name != self._phase:
                    self._phase = phase_name
                    self._turn = 0
                return

    # Scan the entire paragraph, Record the last matching phase keyword,
    # Switch the phase only when something new actually appears.
    def _parse_phase(self, lowered: str) -> None:
        latest_phase = self._phase
        latest_pos = -1
        for phase, pats in _PHASE_PATTERNS.items():
            for pat in pats:
                for m in re.finditer(pat, lowered, flags=re.IGNORECASE):
                    if m.start() > latest_pos:
                        latest_pos = m.start()
                        latest_phase = phase

        if latest_phase != self._phase:
            self._phase = latest_phase
            self._turn = 0

    def _parse_round(self, lowered: str) -> None:
        m_day = _ROUND_DAY_PATTERN.search(lowered)
        m_night = _ROUND_NIGHT_PATTERN.search(lowered)
        candidate = None
        if m_day:
            candidate = int(m_day.group(1))
        elif m_night:
            candidate = int(m_night.group(1))
        if candidate is not None and candidate >= 1:
            self._round = candidate

    def _parse_role(self, lowered: str) -> None:
        m = _ROLE_PATTERN.search(lowered)
        if m:
            self._role = m.group(1).lower()
            return
        m = _YOU_ARE_PATTERN.search(lowered)
        if m:
            self._role = m.group(1).lower()

    def _parse_elimination(self, lowered: str) -> None:
        m = _ELIM_PATTERN.search(lowered)
        if not m:
            return
        pid = int(m.group(1))
        if 0 <= pid < self.num_players:
            self._alive[pid] = False

    def _parse_vote(self, lowered: str) -> None:
        m = _VOTE_PATTERN.search(lowered)
        if not m:
            return
        voter = int(m.group(1))
        target = int(m.group(2))
        if 0 <= voter < self.num_players and 0 <= target < self.num_players:
            entry = {
                "voter": voter,
                "target": target,
                "phase": self._phase,
                "round": self._round,
            }
            self._vote_history.append(entry)
            if len(self._vote_history) > self.max_history:
                self._vote_history.pop(0)

    def _parse_talk(self, text: str) -> None:
        m = _TALK_PATTERN.match(text)
        if not m:
            return
        speaker = int(m.group(1)) if m.group(1).isdigit() else None
        content = m.group(2).strip()
        if speaker is None or not (0 <= speaker < self.num_players):
            return
        entry = {
            "speaker": speaker,
            "text": content,
            "phase": self._phase,
            "round": self._round,
        }
        # Avoid duplicates when consecutive identical messages arrive
        if not self._talk_history or self._talk_history[-1] != entry:
            self._talk_history.append(entry)
            if len(self._talk_history) > self.max_history:
                self._talk_history.pop(0)
        if self._phase == "day_talk":
            self._turn += 1
