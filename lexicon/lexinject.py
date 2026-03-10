#!/usr/bin/env python3
"""
This module performs batch lexicon injection between:
    ASR transcription → Lexicon retrieval → Translation dataset

Core features:
- JSONL batch processing
- CER + top-k controlled retrieval
- Multiple output formats
- Sentence-level statistics logging
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any

from lexmatch import LexiconMatcher, load_dictionary


# ----------------------------------------------------------------------
# Output Format Builders
# ----------------------------------------------------------------------

def build_regular_format(results: List[Dict]) -> str:
    """Human-readable compact lexicon format."""
    parts = []
    for r in results:
        word = r["word"]
        entries = []

        if r["matched"]:
            entries.extend([e["gloss"] for e in r["matched"]])
        else:
            entries.extend([alt["entry"]["gloss"] for alt in r["alts"]])

        if entries:
            parts.append(f"{word}: {'; '.join(entries)}")

    return "\n".join(parts)


def build_flat_json_format(results: List[Dict]) -> Dict[str, List[str]]:
    """Flat JSON key-value format."""
    output = {}
    for r in results:
        word = r["word"]
        entries = []

        if r["matched"]:
            entries.extend([e["gloss"] for e in r["matched"]])
        else:
            entries.extend([alt["entry"]["gloss"] for alt in r["alts"]])

        if entries:
            output[word] = entries

    return output


def build_full_format(results: List[Dict]) -> List[Dict]:
    """Full raw structured output."""
    return results


# ----------------------------------------------------------------------
# Core Processing
# ----------------------------------------------------------------------

def process_file(
    input_path: Path,
    output_path: Path,
    matcher: LexiconMatcher,
    output_mode: str = "regular",
):
    stats_accumulator = {
        "num_sentences": 0,
        "avg_before_topk": 0.0,
        "avg_after_topk": 0.0,
        "matched_ratio": 0.0,
    }

    with input_path.open("r", encoding="utf-8") as fin, \
         output_path.open("w", encoding="utf-8") as fout:

        for line in fin:
            data = json.loads(line)
            text = data.get("text", "").strip()

            if not text:
                continue

            results, stats = matcher.query_sentence(text)

            # Update statistics
            stats_accumulator["num_sentences"] += 1
            stats_accumulator["avg_before_topk"] += stats["avg_before_topk"]
            stats_accumulator["avg_after_topk"] += stats["avg_after_topk"]
            stats_accumulator["matched_ratio"] += stats["matched_ratio"]

            # Build output format
            if output_mode == "regular":
                lexicon_field = build_regular_format(results)
            elif output_mode == "flat_json":
                lexicon_field = build_flat_json_format(results)
            elif output_mode == "full":
                lexicon_field = build_full_format(results)
            else:
                raise ValueError(f"Unsupported output mode: {output_mode}")

            data["lexicon"] = lexicon_field
            fout.write(json.dumps(data, ensure_ascii=False) + "\n")

    # Normalize statistics
    n = stats_accumulator["num_sentences"]
    if n > 0:
        stats_accumulator["avg_before_topk"] /= n
        stats_accumulator["avg_after_topk"] /= n
        stats_accumulator["matched_ratio"] /= n

    print("\nBatch Summary:")
    print(f"  Sentences processed: {n}")
    print(f"  Avg before top-K: {stats_accumulator['avg_before_topk']:.2f}")
    print(f"  Avg after top-K:  {stats_accumulator['avg_after_topk']:.2f}")
    print(f"  Matched ratio:    {stats_accumulator['matched_ratio']:.2%}")


# ----------------------------------------------------------------------
# CLI Interface
# ----------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Batch lexicon injection")

    parser.add_argument("--input", type=str, required=True,
                        help="Input JSONL file with transcription")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file with lexicon injection")
    parser.add_argument("--dict", type=str, required=True,
                        help="Path to lexicon CSV file")

    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--cer_threshold", type=float, default=0.2)
    parser.add_argument("--affix", action="store_true")
    parser.add_argument("--symmetric_cer", action="store_true")

    parser.add_argument("--output_mode",
                        choices=["regular", "flat_json", "full"],
                        default="regular")

    return parser.parse_args()


def main():
    args = parse_args()

    entries = load_dictionary(args.dict)

    matcher = LexiconMatcher(
        entries=entries,
        top_k=args.top_k,
        cer_threshold=args.cer_threshold,
        affix=args.affix,
        symmetric_cer=args.symmetric_cer,
    )

    process_file(
        input_path=Path(args.input),
        output_path=Path(args.output),
        matcher=matcher,
        output_mode=args.output_mode,
    )


if __name__ == "__main__":
    main()
