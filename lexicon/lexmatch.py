#!/usr/bin/env python3
"""
Core functionality:
- Exact match
- Prefix/Suffix match
- CER-based fuzzy match
- Top-k filtering
- Sentence-level statistics
"""

import csv
from typing import List, Dict, Tuple, Optional
from Levenshtein import distance


class LexiconMatcher:
    """Core lexicon matching engine."""

    def __init__(
        self,
        entries: List[Dict],
        top_k: int = 2,
        cer_threshold: float = 0.2,
        affix: bool = True,
        symmetric_cer: bool = False,
        debug_cer: bool = False,
    ):
        self.entries = entries
        self.top_k = top_k
        self.cer_threshold = cer_threshold
        self.affix = affix
        self.symmetric_cer = symmetric_cer
        self.debug_cer = debug_cer

    # ------------------------------------------------------------------
    # Utility Functions
    # ------------------------------------------------------------------

    @staticmethod
    def _norm(text: str) -> str:
        return text.lower().strip()

    def _cer(self, a: str, b: str) -> float:
        if not a and not b:
            return 0.0
        if not b:
            return float("inf")

        d = distance(a, b)
        if self.symmetric_cer:
            return d / max((len(a) + len(b)) / 2, 1)
        return d / len(b)

    # ------------------------------------------------------------------
    # Matching Methods
    # ------------------------------------------------------------------

    def _exact_match(self, word: str) -> List[Dict]:
        lw = self._norm(word)
        return [
            entry
            for entry in self.entries
            if lw == self._norm(entry.get("lexical_unit", ""))
            or lw == self._norm(entry.get("variant", ""))
        ]

    def _prefix_suffix(self, word: str) -> Tuple[List[Dict], List[Dict]]:
        prefixes, suffixes = [], []
        lw = self._norm(word)

        for entry in self.entries:
            lex = self._norm(entry.get("lexical_unit", ""))
            if not lex:
                continue

            if lex.endswith("-") and lw.startswith(lex[:-1]):
                prefixes.append(entry)
            elif lex.startswith("-") and lw.endswith(lex[1:]):
                suffixes.append(entry)

        return prefixes, suffixes

    def _alternatives(
        self,
        word: str,
        exclude_ids: set,
        return_all: bool = False,
    ):
        lw = self._norm(word)
        scored = []
        force_include = []

        for entry in self.entries:
            if id(entry) in exclude_ids:
                continue

            for cand in (entry.get("lexical_unit", ""), entry.get("variant", "")):
                if not cand:
                    continue

                cand_norm = self._norm(cand)
                cer_val = self._cer(lw, cand_norm)

                if self.debug_cer:
                    print(f"[CER] '{lw}' vs '{cand}' = {cer_val:.4f}")

                if cer_val <= self.cer_threshold:
                    scored.append((entry, cer_val))
                    break

                if cand_norm in lw or lw in cand_norm:
                    force_include.append((entry, cer_val))
                    break

        scored.sort(key=lambda x: x[1])
        force_include.sort(key=lambda x: x[1])

        if return_all:
            return scored + force_include

        return scored[: self.top_k] + force_include

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def query(
        self,
        word: str,
        use_affix: Optional[bool] = None,
    ) -> Optional[Dict]:

        use_affix = self.affix if use_affix is None else use_affix

        exact_matches = self._exact_match(word)
        prefixes, suffixes = [], []
        full_alts, topk_alts = [], []

        if not exact_matches:
            if use_affix:
                prefixes, suffixes = self._prefix_suffix(word)

            used_ids = {id(e) for e in (*prefixes, *suffixes)} if use_affix else set()
            full_alts = self._alternatives(word, used_ids, return_all=True)
            topk_alts = full_alts[: self.top_k]

        if not (exact_matches or full_alts or prefixes or suffixes):
            return None

        result = {
            "word": word,
            "matched": exact_matches,
            "alts": [{"entry": e, "cer": d} for e, d in topk_alts],
            "meta": {
                "num_full_alts": len(full_alts) if not exact_matches else 0,
                "num_affix": len(prefixes) + len(suffixes)
                if (not exact_matches and use_affix)
                else 0,
                "num_final": len(exact_matches)
                if exact_matches
                else len(topk_alts)
                + (len(prefixes) + len(suffixes) if use_affix else 0),
                "matched": bool(exact_matches),
            },
        }

        if use_affix:
            result["prefix"] = prefixes
            result["suffix"] = suffixes
        else:
            result["prefix"] = None
            result["suffix"] = None

        return result

    def query_sentence(self, sentence: str) -> Tuple[List[Dict], Dict]:
        results = []
        num_words = 0
        total_before_topk = 0
        total_after_topk = 0
        num_matched_words = 0

        for word in sentence.strip().split():
            res = self.query(word)
            num_words += 1

            if not res:
                continue

            results.append(res)
            meta = res["meta"]

            total_before_topk += (
                1 if meta["matched"] else meta["num_full_alts"] + meta["num_affix"]
            )
            total_after_topk += (
                1 if meta["matched"] else meta["num_final"]
            )
            num_matched_words += 1 if meta["matched"] else 0

        stats = {
            "num_words": num_words,
            "avg_before_topk": total_before_topk / num_words if num_words else 0.0,
            "avg_after_topk": total_after_topk / num_words if num_words else 0.0,
            "matched_ratio": num_matched_words / num_words if num_words else 0.0,
        }

        return results, stats


# ----------------------------------------------------------------------
# Dictionary Utilities
# ----------------------------------------------------------------------

def load_dictionary(csv_path: str) -> List[Dict]:
    entries = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            entries.append(
                {
                    "lexical_unit": row.get("lexical_unit", "").strip(),
                    "variant": row.get("variant", "").strip(),
                    "pos": row.get("pos", "").strip(),
                    "gloss": row.get("gloss", "").strip(),
                }
            )
    return entries


# ----------------------------------------------------------------------
# Debug Helper
# ----------------------------------------------------------------------

def pretty_print(match_res: List[Dict]):
    for res in match_res:
        print(f"\nInput: {res['word']}")

        if res["matched"]:
            for m in res["matched"]:
                print(f"Matched: {m['lexical_unit']} ({m['pos']}) - {m['gloss']}")
        else:
            print("Matched: —")

        if res["alts"]:
            print("Alternatives:")
            for item in res["alts"]:
                e = item["entry"]
                print(
                    f"  {e['lexical_unit']} ({e['pos']}) - {e['gloss']} "
                    f"[CER: {item['cer']:.4f}]"
                )

        if res["prefix"]:
            print("Prefix Matches:")
            for p in res["prefix"]:
                print(f"  {p['lexical_unit']} ({p['pos']}) - {p['gloss']}")

        if res["suffix"]:
            print("Suffix Matches:")
            for s in res["suffix"]:
                print(f"  {s['lexical_unit']} ({s['pos']}) - {s['gloss']}")


if __name__ == "__main__":
    lexicon_csv = "./cleaned_lexicon.csv"
    entries = load_dictionary(lexicon_csv)

    matcher = LexiconMatcher(
        entries,
        top_k=1,
        cer_threshold=0.3,
        affix=False,
        symmetric_cer=False,
        debug_cer=False,
    )

    sentence = (
        "yilgbayi yininya nanawunyinyilgbayi yininya nanawunyin "
        "gabarri yidugal yi gungan gin yi waray gandiya bla naughty "
        "waray gandi yilgbayi wan ngayana yinggi yinggangala "
        "yanimayi dularriya gungan gin yidugal nana na waray gandiya "
        "yibiyan yi waray yanggunburr gan yanima yimbalyarri wan yimbanay"
    )

    results, stats = matcher.query_sentence(sentence)
    pretty_print(results)

    print("\nSentence Summary:")
    print(f"  Avg before top-K: {stats['avg_before_topk']:.2f}")
    print(f"  Avg after top-K:  {stats['avg_after_topk']:.2f}")
    print(f"  Matched ratio:    {stats['matched_ratio']:.2%}")
