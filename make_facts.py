#!/usr/bin/env python
"""Generate stronger fact extraction for memory_eval.

Usage:
  python make_facts.py --transcript transcript.txt --out facts_strong.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from typing import List, Tuple

import memory_eval as me


FACT_PATTERNS = [
    re.compile(r"\bI am\b[^\.!\?]*", re.IGNORECASE),
    re.compile(r"\bI'm\b[^\.!\?]*", re.IGNORECASE),
    re.compile(r"\bI have\b[^\.!\?]*", re.IGNORECASE),
    re.compile(r"\bI like\b[^\.!\?]*", re.IGNORECASE),
    re.compile(r"\bI love\b[^\.!\?]*", re.IGNORECASE),
    re.compile(r"\bI prefer\b[^\.!\?]*", re.IGNORECASE),
    re.compile(r"\bI enjoy\b[^\.!\?]*", re.IGNORECASE),
    re.compile(r"\bMy\b[^\.!\?]*", re.IGNORECASE),
    re.compile(r"\bWe are\b[^\.!\?]*", re.IGNORECASE),
    re.compile(r"\bWe have\b[^\.!\?]*", re.IGNORECASE),
]


def _clean_sentence(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" -:;,.")
    return s


def _is_question(s: str) -> bool:
    return "?" in s


def _score_sentence(s: str) -> float:
    score = 0.0
    low = s.lower()
    for kw in me.FACT_KEYWORDS:
        if kw in low:
            score += 2.0
    for kw in me.PLAN_KEYWORDS:
        if kw in low:
            score += 1.0
    # Prefer self-referential facts
    if re.search(r"\b(i|my|we)\b", low):
        score += 2.0
    # Penalize too short or too long
    if len(s) < 20:
        score -= 1.0
    if len(s) > 140:
        score -= 1.0
    return score


def _extract_fact_phrase(s: str) -> str:
    for pat in FACT_PATTERNS:
        m = pat.search(s)
        if m:
            return _clean_sentence(m.group(0))
    return _clean_sentence(s)


def _dedupe(items: List[str]) -> List[str]:
    out: List[str] = []
    for s in items:
        if not s:
            continue
        if any(me.jaccard(s, t) > 0.7 for t in out):
            continue
        out.append(s)
    return out


def pick_facts_from_convo(text: str, per_convo: int, early_turns: int) -> List[str]:
    # Use only early turns to simulate long-memory recall
    lines = [ln for ln in text.splitlines() if ln.strip()]
    early_lines = lines[:early_turns]
    early_text = "\n".join(early_lines)

    sents = me.split_sentences(early_text)
    candidates: List[Tuple[float, str]] = []
    for s in sents:
        s = _clean_sentence(s)
        if not s or _is_question(s):
            continue
        score = _score_sentence(s)
        if score <= 0:
            continue
        candidates.append((score, s))

    candidates.sort(key=lambda x: x[0], reverse=True)
    facts = [_extract_fact_phrase(s) for _, s in candidates[: per_convo * 3]]
    facts = _dedupe(facts)[:per_convo]
    return facts


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate stronger facts.jsonl.")
    parser.add_argument("--transcript", required=True, help="Path to transcript.txt")
    parser.add_argument("--out", required=True, help="Output facts.jsonl")
    parser.add_argument("--per-convo", type=int, default=2, help="Facts per conversation")
    parser.add_argument("--early-turns", type=int, default=6, help="Use first N lines per conversation")
    args = parser.parse_args()

    convos = me.parse_transcript_conversations(args.transcript)
    lines = []
    for idx, convo in enumerate(convos):
        facts = pick_facts_from_convo(convo, args.per_convo, args.early_turns)
        for f in facts:
            obj = {
                "fact": f,
                "answers": [f],
                "type": "fact",
                "question": "What was said early in this conversation?",
                "conv_idx": idx,
            }
            lines.append(json.dumps(obj, ensure_ascii=False))

    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("conversations:", len(convos))
    print("facts:", len(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
