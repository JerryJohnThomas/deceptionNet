"""LLM-based presenter for generating natural talk utterances."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from typing import Any, Optional

import logging

from deceptionNet.utils import truncate_to_lines, debug_log

import re

_LOGGER = logging.getLogger(__name__)
FALLBACK_MODEL = "microsoft/phi-1_5"


@dataclass
class PresenterPrompt:
    player_id: int
    intent: str
    target: str
    phase: str
    belief_summary: Optional[str]
    style: str
    max_lines: int


    def build(self) -> str:
        summary = self.belief_summary or "No extra beliefs supplied."
        
        # Use a chat-style format that most instruct models understand better
        template = (
            f"<|system|>\n"
            f"You are Player {self.player_id} in a Mafia-style social deduction game.\n"
            f"Context (internal): {summary}\n"
            f"Phase: {self.phase} | Intent: {self.intent} | Target: {self.target or 'none'}\n"
            f"<|user|>\n"
            f"What would Player {self.player_id} say right now? Respond with ONLY the dialogue - "
            f"no labels, no explanations, just {self.max_lines} natural sentence(s) in a {self.style} tone.\n"
            f"<|assistant|>\n"
        )
        
        return template

    def build_old(self) -> str:
        # summary = self.belief_summary or "No specific beliefs provided."

        # template = (
        #     f"You are Player {self.player_id} in a social deduction game (like Mafia).\n"
        #     f"Phase: {self.phase}. Intent: {self.intent}. Target: {self.target or 'none'}.\n"
        #     f"Belief summary: {summary}\n\n"
        #     f"Speak in-character as Player {self.player_id}. "
        #     f"Write the exact message you would say in this situation.\n"
        #     f"Do NOT prefix with 'Example:', 'Message:', 'Player X:', or any labels. "
        #     f"Just output the line of dialogue only.\n"
        #     f"Avoid meta commentary or explaining your role. "
        #     f"Keep it {self.style} and at most {self.max_lines} short sentences.\n\n"
        #     f"Example:\nPlayer 2: I think Player 5 has been too quiet today.\n\n"
        #     f"Now write your own message as Player {self.player_id}:"
        # )

        # template = (
        #     f"You are Player {self.player_id} in a Mafia-style social deduction game.\n"
        #     f"Phase: {self.phase}. Intent: {self.intent}. Target: {self.target or 'none'}.\n"
        #     f"Beliefs: {summary}\n\n"
        #     f"Speak naturally as Player {self.player_id}. Respond only with what you would say in chat.\n"
        #     f"Do not use prefixes like 'Example', 'Message', or quotes.\n"
        #     f"Keep it {self.style}, concise, and {self.max_lines} short sentences max."
        # )



        # template = (
        #     f"You are Player {self.player_id} in a social deduction game (like Mafia).\n"
        #     f"Current phase: {self.phase}.\n"
        #     f"Intent: {self.intent}.\n"
        #     f"Target: {self.target or 'none'}.\n"
        #     f"Belief summary: {summary}\n\n"
        #     f"Speak **in-character** as Player {self.player_id}.\n"
        #     f"Write a natural, realistic message in one or two short sentences — "
        #     f"stay true to your role, but **never explicitly reveal or hint at your hidden role or team**.\n"
        #     f"Do not use meta phrases like 'assistant:' or 'You are'.\n"
        #     f"Keep your tone {self.style} and concise (max {self.max_lines} lines)."
        # )

        #  template = (
        #     f"You are Player {self.player_id} in a social deduction game (like Mafia).\n"
        #     f"Phase: {self.phase}.\n"
        #     f"Intent: {self.intent}. Target: {self.target or 'none'}.\n"
        #     f"Context summary (for your reasoning, not to repeat):\n"
        #     f"{summary}\n\n"
        #     f"Now, speak **as Player {self.player_id}** — write what you would say in this situation.\n"
        #     f"⚠️ Do **not** repeat or mention the context summary.\n"
        #     f"⚠️ Do **not** prefix with 'Example:', 'Belief summary:', or 'Player X:'.\n"
        #     f"Respond with {self.max_lines} concise, natural sentences in a {self.style} tone."
        # )


        # hopefully works prompotiny technique
        summary = self.belief_summary or "No extra beliefs supplied."
        template = (
            f"### CONTEXT(INTERNAL - do not reveal)\n"
            f"Belief summary : {summary} \n\n"
            f"### GAME STATE\n"
            f"You are Player {self.player_id} in a Mafia-like social deduction game.\n"
            f"Phase: {self.phase}\n"
            f"Intent: {self.intent}\n"
            f"Target: {self.target or 'none'}\n\n"
            f"### INSTRUCTIONS\n"
            f"Write only the message you would actually speak in this situation.\n"
            f"- Stay in character, concise ({self.max_lines} short sentences).\n"
            f"- Never leak or reference your hidden role or the context above.\n"
            f"- Do NOT include labels like 'Example:', 'Message:', 'Belief summary:', or 'Player X:'.\n"
            f"- Output ONLY the dialogue line(s).\n\n"
            f"Now respond with ONLY what Player {self.player_id} would actually say.\n"
            f"Do not restate or reference these instructions.\n"
            f"Output just the dialogue line — nothing else."
            f"### RESPONSE\n"
        )

        return template

    # def build(self) -> str:
    #     summary = self.belief_summary or "No extra beliefs supplied."
    #     template = (
    #         "You are Player {player_id} in a social deduction game.\n"
    #         "Intent: {intent}. Target: {target}. Phase: {phase}.\n"
    #         "Belief summary: {summary}.\n"
    #         "Adopt a {style} tone. Respond with at most {max_lines} short sentences."
    #     )
    #     return template.format(
    #         player_id=self.player_id,
    #         intent=self.intent,
    #         target=self.target,
    #         phase=self.phase,
    #         summary=summary,
    #         style=self.style,
    #         max_lines=self.max_lines,
    #     )


class LLMPresenter:
    """Small wrapper over a HF text-generation pipeline with graceful fallback."""

    def __init__(
        self,
        model_name: str = "microsoft/Phi-3-mini-128k-instruct",
        cache_dir: Optional[str] = None,
        style: str = "neutral",
        max_lines: int = 1,
        device: str = "auto",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 64,
    ) -> None:
        self.primary_model = model_name
        self.cache_dir = cache_dir
        self.style = style
        self.max_lines = max(1, int(max_lines))
        self.temperature = temperature
        self.top_p = top_p
        self.max_new_tokens = max_new_tokens
        self.requested_device = (device or 'auto').lower()
        self.active_device: Optional[str] = None
        self.model_name: Optional[str] = None

        self._generator = None
        self._load_pipeline()

    def _load_pipeline(self) -> None:
        try:
            from transformers import pipeline  # type: ignore
        except Exception as exc:  # transformers missing
            self._generator = None
            _LOGGER.warning(
                "LLM presenter fallback to template strings because pipeline failed: %s", exc
            )
            debug_log(f"LLM Presenter unavailable: {exc}")
            return

        def load_model(model_name: str, device_choice: str) -> Any:
            kwargs: dict[str, Any] = {"model": model_name}
            if self.cache_dir:
                kwargs["cache_dir"] = self.cache_dir
            kwargs["device"] = 0 if device_choice == "cuda" else -1
            gen = pipeline("text-generation", **kwargs)
            if device_choice == "cpu":
                try:
                    gen.model.to("cpu")  # type: ignore
                except Exception:
                    pass
            return gen

        # device preference ordering
        pref = getattr(self, "requested_device", "auto")
        device_order: list[str] = []
        if pref == "template":
            device_order = ["template"]
        else:
            if pref in ("auto", "gpu") and torch.cuda.is_available():
                device_order.append("cuda")
            if pref in ("auto", "gpu", "cpu"):
                device_order.append("cpu")
            if not device_order:
                device_order.append("cpu")
            device_order.append("template")

        models = [getattr(self, "primary_model", self.model_name)]
        if models[0] != FALLBACK_MODEL:
            models.append(FALLBACK_MODEL)

        last_exc: Optional[Exception] = None
        for candidate in models:
            for device_choice in device_order:
                if device_choice == "template":
                    break
                try:
                    generator = load_model(candidate, device_choice)
                    self._generator = generator
                    self.model_name = candidate
                    self.active_device = device_choice
                    print(f"LLM Presenter active: {candidate} (device={device_choice})")
                    debug_log(f"LLM Presenter active: {candidate} (device={device_choice})")
                    return
                except RuntimeError as exc:
                    last_exc = exc
                    msg = str(exc).lower()
                    if "cuda" in msg or "out of memory" in msg:
                        print(f"LLM Presenter CUDA error: {exc}. Falling back to CPU.")
                        debug_log(f"LLM Presenter CUDA error: {exc}. Falling back to CPU.")
                        if torch.cuda.is_available():
                            try:
                                torch.cuda.empty_cache()
                            except Exception:
                                pass
                        continue
                    break
                except OSError as exc:
                    last_exc = exc
                    msg = str(exc).lower()
                    if "paging file" in msg and candidate != FALLBACK_MODEL:
                        print(f"LLM Presenter CPU load failed ({exc}); trying fallback model.")
                        debug_log(f"LLM Presenter CPU load failed ({exc}); trying fallback model.")
                        break
                    break
                except Exception as exc:
                    last_exc = exc
                    break

        # fallback if nothing worked
        self._generator = None
        self.active_device = "template"
        self.model_name = None
        _LOGGER.warning(
            "LLM presenter fallback to template strings because pipeline failed: %s",
            last_exc,
        )
        if last_exc is not None:
            debug_log(f"LLM Presenter unavailable: {last_exc}")


    # ------------------------------------------------------------------
    def render(
        self,
        intent: str,
        target: str,
        phase: str,
        belief_summary: Optional[str],
        player_id: int,
    ) -> str:
        # Important Presenter LLM critical function
        prompt = PresenterPrompt(
            player_id=player_id,
            intent=intent,
            target=target,
            phase=phase,
            belief_summary=belief_summary,
            style=self.style,
            max_lines=self.max_lines,
        ).build()
        print("[ALPHA START]")

        if self._generator is None:
            return self._fallback(intent, target, phase, belief_summary)

        try:
            print("[ALPHA INIT]")
            outputs = self._generator(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=True,
                pad_token_id=self._generator.tokenizer.eos_token_id if hasattr(self._generator, "tokenizer") else None,
            )
            if not outputs:
                raise RuntimeError("empty generation")
            raw_text = outputs[0]["generated_text"]
            completion = raw_text[len(prompt):].strip() if raw_text.startswith(prompt) else raw_text.strip()

            if completion.strip() in ["[0]", "0", ""]:
                completion = "Let's discuss and share our thoughts."
            print("[ALPHA BEFORE CLEAN]", completion ,"[:OVER]")
            completion = self._clean_output(completion, prompt)
            print("[ALPHA AFTER CLEAN]", completion ,"[:OVER]")
        except Exception as exc:  # pragma: no cover - runtime failure fallback
            _LOGGER.warning("LLM presenter generation failed (%s); using fallback", exc)
            return self._fallback(intent, target, phase, belief_summary)

        # # Remove generic meta prefixes (Example:, Message:, Reply:)
        # completion = re.sub(r"(?i)\b(?:example|message|reply)\b\s*[:\-]\s*", "", completion).strip()

        # # Remove LLM prompt echo tags (but keep useful '### Question' or '### Follow up' text)
        # completion = re.sub(
        #     r"#+\s*(game state|solution|context|instruction|response type|additional.*|explanation|answer.*)",
        #     "",
        #     completion,
        #     flags=re.I
        # )

        # # Remove self/role references
        # completion = re.sub(r"you are player\s*\d+.*?(?=\.|$)", "", completion, flags=re.I)
        # completion = re.sub(r"write only the message.*", "", completion, flags=re.I)


        # # Final cleanup sweep for edge cases
        # completion = re.sub(r"\[.*?(ROLE|PHASE|GAME|INTERNAL).*?\]", "", completion, flags=re.I)
        # completion = re.sub(r"you are player\s*\d+", "", completion, flags=re.I)
        # completion = re.sub(r"\s{2,}", " ", completion).strip()




        formatted = truncate_to_lines(completion, self.max_lines)
        return formatted

    def _clean_output(self, text: str, prompt: str) -> str:
        """Aggressively clean LLM output to extract only dialogue."""
        
        # Remove the prompt if echoed
        if text.startswith(prompt):
            text = text[len(prompt):].strip()
        
        # Remove everything after "Welcome to Secret Mafia!" (case-insensitive)
        text = re.sub(r'(?i)welcome to secret mafia!.*', '', text)
        
        # Remove ONLY meta-tags, preserve player number references like [5]
        # Match brackets containing specific meta keywords
        text = re.sub(r'\[(?:ROLE|PHASE|GAME|INTERNAL|CONTEXT|INSTRUCTION)[^\]]*\]', '', text, flags=re.I)
        # Also catch variations like [GAME=something]
        text = re.sub(r'\[[^\]]*(?:ROLE|PHASE|GAME|INTERNAL|CONTEXT|INSTRUCTION)[^\]]*\]', '', text, flags=re.I)
        
        # Split by common delimiters and take the first real content
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        
        # Filter out lines that look like instructions/meta-text
        dialogue_lines = []
        skip_patterns = [
            r'^(###|<\|)',  # Markdown headers or chat tokens
            r'(?i)^(you are|player \d+:|context|instruction|response|example|message)',
            r'(?i)^(internal|do not reveal|game state)',
            r'(?i)welcome to',  # Any line starting with "welcome to"
        ]
        
        for line in lines:
            # Skip if matches any skip pattern
            if any(re.match(pattern, line) for pattern in skip_patterns):
                continue
            
            # Remove common prefixes (but keep player numbers in brackets)
            line = re.sub(r'^(?:Player \d+:\s*|["\'])', '', line)
            line = re.sub(r'["\']$', '', line)
            
            # Remove meta-tags again (in case they survived line splitting)
            line = re.sub(r'\[(?:ROLE|PHASE|GAME|INTERNAL|CONTEXT)[^\]]*\]', '', line, flags=re.I)
            
            # Clean up extra whitespace
            line = re.sub(r'\s+', ' ', line).strip()
            
            if line and len(line) > 5:  # Ignore very short fragments
                dialogue_lines.append(line)
        
        # Join and return
        result = ' '.join(dialogue_lines)
        return result if result else "..."

    # ------------------------------------------------------------------
    def _fallback(
        self,
        intent: str,
        target: str,
        phase: str,
        belief_summary: Optional[str],
    ) -> str:
        base = {
            "accuse": f"I suspect {target} right now.",
            "defend_self": "I stand by my innocence.",
            "defend_other": f"Let's give {target} a chance.",
            "claim": f"I'm sharing what I know about {target}.",
            "agree": f"I back what {target} just said.",
            "question": f"{target}, can you clarify your story?",
            "filler": "We should stay focused on the facts.",
            "silent": "I'll stay quiet for now.",
        }
        sentence = base.get(intent, "I'm considering our options.")
        if belief_summary:
            sentence = f"{sentence} {belief_summary.split('.', 1)[0].strip()}"
        return truncate_to_lines(sentence, self.max_lines)


__all__ = ["LLMPresenter"]



