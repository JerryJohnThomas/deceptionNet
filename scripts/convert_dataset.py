#!/usr/bin/env python
"""CLI helper for converting raw werewolf logs into normalized JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.append(str(PROJECT_ROOT / "src"))

from deceptionnet.data.pipeline import convert_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert raw Werewolf logs to normalized JSONL")
    parser.add_argument(
        "--raw-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "dataset1" / "werewolf_data" / "werewolf_data",
        help="Directory containing raw *.json werewolf logs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "games.jsonl",
        help="Destination JSONL file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit on number of games to convert",
    )
    parser.add_argument("--translate", action="store_true", help="Translate conversation text to English")
    parser.add_argument(
        "--translator-type",
        choices=["nllb", "marian"],
        default="nllb",
        help="Translation backend to use when --translate is enabled",
    )
    parser.add_argument(
        "--translator-model",
        type=str,
        default="facebook/nllb-200-distilled-1.3B",
        help="HuggingFace model to use for translation",
    )
    parser.add_argument(
        "--translator-device",
        type=str,
        default="cuda",
        help="Device string passed to the translator (e.g. cpu, cuda)",
    )
    parser.add_argument(
        "--src-lang",
        type=str,
        default=None,
        help="Source language code (required for some models like NLLB)",
    )
    parser.add_argument(
        "--tgt-lang",
        type=str,
        default="eng_Latn",
        help="Target language code (for NLLB-style models)",
    )
    parser.add_argument(
        "--no-translate-summary",
        action="store_true",
        help="Skip translating the summary field (only translate raw text)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    translator = None
    if args.translate:
        if args.translator_type == "marian":
            from deceptionnet.utils.translation import MarianTranslator

            translator = MarianTranslator(
                model_name=args.translator_model,
                device=args.translator_device,
            )
        else:
            from deceptionnet.utils.translation import Seq2SeqTranslator

            inferred_src = args.src_lang
            if inferred_src is None:
                if "zh" in args.translator_model or "zho" in args.translator_model:
                    inferred_src = "zho_Hans"
                elif "ja" in args.translator_model or "jpn" in args.translator_model:
                    inferred_src = "jpn_Jpan"
                else:
                    inferred_src = "zho_Hans"

            translator = Seq2SeqTranslator(
                model_name=args.translator_model,
                device=args.translator_device,
                src_lang=inferred_src,
                tgt_lang=args.tgt_lang,
            )

    convert_dataset(
        args.raw_root,
        args.output,
        limit=args.limit,
        translator=translator,
        translate_summary=not args.no_translate_summary,
    )
    print(f"Conversion complete. Output written to {args.output}")


if __name__ == "__main__":
    main()
