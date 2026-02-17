#!/usr/bin/env python
"""Memory strategy evaluator (recursive baselines and dual-rate memory).

Usage examples:
  python memory_eval.py --transcript transcript.txt --facts facts.jsonl
  python memory_eval.py --transcript transcript.txt --facts facts.jsonl --chunk-tokens 700 --main-tokens 250

Facts JSONL format (one per line):
  {"fact": "user prefers short answers", "answers": ["short answers"], "type": "fact", "question": "What does the user prefer?"}

If "answers" is missing, "fact" is used for matching.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from typing import List, Dict, Iterable, Tuple, Optional


# ----------------------------- Tokenization -----------------------------

def _try_tiktoken():
    try:
        import tiktoken  # type: ignore
        return tiktoken
    except Exception:
        return None


def estimate_tokens(text: str, encoder=None) -> int:
    if not text:
        return 0
    if encoder is not None:
        try:
            return len(encoder.encode(text))
        except Exception:
            pass
    # Fallback: rough tokenization by words + punctuation
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    return len(tokens)


# ----------------------------- Text utils -----------------------------

STOPWORDS = set(
    """the a an and or but if then else of to in for on at by with is are was were be been
    i you he she it we they this that these those as from not no yes do does did can could
    will would should may might about into over under out up down more most less least very
    """.split()
)

FACT_KEYWORDS = [
    "always", "never", "must", "cannot", "constraint", "prefer", "preference", "goal",
    "require", "required", "rule", "rules",
]

PLAN_KEYWORDS = [
    "todo", "next", "plan", "task", "action", "doing", "in progress",
    "step", "steps", "progress",
]

IMPORTANCE_KEYWORDS = set(FACT_KEYWORDS + PLAN_KEYWORDS + [
    "important", "critical", "must", "need", "require",
])

NEGATIONS = set(["not", "never", "no", "cannot", "can't", "won't", "don't", "didn't"])


def split_sentences(text: str) -> List[str]:
    # Split by common sentence delimiters, keep meaningful sentences
    parts = re.split(r"(?<=[\.\!\?])\s+", text.strip(), flags=re.UNICODE)
    sentences = [p.strip() for p in parts if p.strip()]
    return sentences


def score_sentence(sentence: str, freq: Dict[str, int], mode: str) -> float:
    words = re.findall(r"\w+", sentence.lower(), flags=re.UNICODE)
    score = sum(freq.get(w, 0) for w in words if w not in STOPWORDS)

    if mode == "facts":
        if any(k in sentence for k in FACT_KEYWORDS):
            score += 5
    elif mode == "plans":
        if any(k in sentence for k in PLAN_KEYWORDS):
            score += 5
    elif mode == "fusion":
        if any(k in sentence for k in FACT_KEYWORDS):
            score += 3
        if any(k in sentence for k in PLAN_KEYWORDS):
            score += 3
    return score


def extract_summary(text: str, max_tokens: int, mode: str, encoder=None) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return ""

    # Term frequencies
    words = re.findall(r"\w+", text.lower(), flags=re.UNICODE)
    freq: Dict[str, int] = {}
    for w in words:
        if w in STOPWORDS:
            continue
        freq[w] = freq.get(w, 0) + 1

    scored: List[Tuple[int, float, str]] = []
    for idx, sent in enumerate(sentences):
        scored.append((idx, score_sentence(sent, freq, mode), sent))

    # Select high-score sentences, then preserve original order
    scored.sort(key=lambda x: x[1], reverse=True)
    chosen = []
    total = 0
    for idx, _, sent in scored:
        sent_tokens = estimate_tokens(sent, encoder)
        if total + sent_tokens > max_tokens:
            continue
        chosen.append((idx, sent))
        total += sent_tokens
        if total >= max_tokens:
            break

    if not chosen:
        # Fallback: truncate first sentence
        first = sentences[0]
        return truncate_to_tokens(first, max_tokens, encoder)

    chosen.sort(key=lambda x: x[0])
    return " ".join(s for _, s in chosen)


def truncate_to_tokens(text: str, max_tokens: int, encoder=None) -> str:
    if estimate_tokens(text, encoder) <= max_tokens:
        return text
    if encoder is not None:
        try:
            tokens = encoder.encode(text)
            return encoder.decode(tokens[:max_tokens])
        except Exception:
            pass
    parts = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    return " ".join(parts[:max_tokens])


def _tokens(text: str) -> List[str]:
    return [w for w in re.findall(r"\w+", text.lower(), flags=re.UNICODE) if w not in STOPWORDS]


def importance_score(text: str) -> float:
    tokens = _tokens(text)
    if not tokens:
        return 0.0
    score = 0.0
    for t in tokens:
        if t in IMPORTANCE_KEYWORDS:
            score += 2.0
    # Slightly favor longer, informative chunks
    score += min(len(tokens), 80) / 40.0
    return score


def jaccard(a: str, b: str) -> float:
    ta = set(_tokens(a))
    tb = set(_tokens(b))
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return inter / union if union else 0.0


def detect_conflicts(text_a: str, text_b: str, min_overlap: int = 2) -> List[str]:
    conflicts: List[str] = []
    sents_a = split_sentences(text_a)
    sents_b = split_sentences(text_b)
    for sa in sents_a:
        ta = set(_tokens(sa))
        if not ta:
            continue
        na = any(n in sa.lower() for n in NEGATIONS)
        for sb in sents_b:
            tb = set(_tokens(sb))
            if not tb:
                continue
            if len(ta & tb) < min_overlap:
                continue
            nb = any(n in sb.lower() for n in NEGATIONS)
            if na != nb:
                snippet = sa if len(sa) <= len(sb) else sb
                conflicts.append(snippet)
                if len(conflicts) >= 3:
                    return conflicts
    return conflicts


 




# ----------------------------- Chunking -----------------------------

@dataclass
class Block:
    idx: int
    text: str


def chunk_by_tokens(text: str, chunk_tokens: int, encoder=None) -> List[Block]:
    # Simple greedy chunking by sentence, falling back to line splits
    sentences = split_sentences(text)
    if not sentences:
        sentences = [line.strip() for line in text.splitlines() if line.strip()]

    blocks: List[Block] = []
    cur = []
    cur_tokens = 0
    idx = 1

    for sent in sentences:
        t = estimate_tokens(sent, encoder)
        if cur_tokens + t > chunk_tokens and cur:
            blocks.append(Block(idx=idx, text=" ".join(cur)))
            idx += 1
            cur = [sent]
            cur_tokens = t
        else:
            cur.append(sent)
            cur_tokens += t

    if cur:
        blocks.append(Block(idx=idx, text=" ".join(cur)))

    return blocks


def parse_transcript_conversations(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    convos: List[List[str]] = []
    current: List[str] = []
    for line in lines:
        if line.startswith("Conversation "):
            if current:
                convos.append(current)
                current = []
            continue
        if line.strip() == "" and current:
            convos.append(current)
            current = []
            continue
        if line.strip():
            current.append(line)

    if current:
        convos.append(current)

    return ["\n".join(c) for c in convos]


# ----------------------------- Strategies -----------------------------

@dataclass
class StrategyResult:
    name: str
    main_memory: str
    recent_blocks: List[str]
    calls: int = 0
    in_tokens: int = 0
    out_tokens: int = 0
    time_ms: float = 0.0


@dataclass
class StrategySnapshots:
    name: str
    contexts: List[str]  # context after each block


def run_baseline(blocks: List[Block], recent_keep: int) -> StrategyResult:
    recent = []
    t0 = time.perf_counter()
    for b in blocks:
        recent.append(b.text)
        if len(recent) > recent_keep:
            recent.pop(0)
    dt = (time.perf_counter() - t0) * 1000.0
    return StrategyResult(
        name="baseline_sliding",
        main_memory="",
        recent_blocks=recent,
        calls=0,
        in_tokens=0,
        out_tokens=0,
        time_ms=dt,
    )


def run_recursive(blocks: List[Block], main_tokens: int, recent_keep: int, encoder=None) -> StrategyResult:
    main = ""
    recent = []
    calls = 0
    in_tokens = 0
    out_tokens = 0
    t0 = time.perf_counter()
    for b in blocks:
        inp = (main + "\n" + b.text)
        in_tokens += estimate_tokens(inp, encoder)
        main = extract_summary(inp, max_tokens=main_tokens, mode="fusion", encoder=encoder)
        out_tokens += estimate_tokens(main, encoder)
        calls += 1
        recent.append(b.text)
        if len(recent) > recent_keep:
            recent.pop(0)
    dt = (time.perf_counter() - t0) * 1000.0
    return StrategyResult(
        name="recursive_summary",
        main_memory=main,
        recent_blocks=recent,
        calls=calls,
        in_tokens=in_tokens,
        out_tokens=out_tokens,
        time_ms=dt,
    )


def run_recursive_dual_rate(
    blocks: List[Block],
    main_tokens: int,
    recent_keep: int,
    slow_tokens: int,
    slow_update_every: int,
    importance_threshold: float,
    fast_update_every: int,
    fast_importance: float,
    encoder=None,
) -> StrategyResult:
    # fast memory updates every block; slow memory updates only when important or on schedule
    fast = ""
    slow = ""
    recent: List[str] = []
    calls = 0
    in_tokens = 0
    out_tokens = 0
    t0 = time.perf_counter()

    for b in blocks:
        do_fast = False
        if fast_update_every > 0 and (b.idx % fast_update_every == 0):
            do_fast = True
        if importance_score(b.text) >= fast_importance:
            do_fast = True

        if do_fast:
            inp_fast = fast + "\n" + b.text
            in_tokens += estimate_tokens(inp_fast, encoder)
            fast = extract_summary(inp_fast, max_tokens=main_tokens, mode="fusion", encoder=encoder)
            out_tokens += estimate_tokens(fast, encoder)
            calls += 1

        do_slow = False
        if slow_update_every > 0 and (b.idx % slow_update_every == 0):
            do_slow = True
        if importance_score(b.text) >= importance_threshold:
            do_slow = True

        if do_slow:
            inp_slow = slow + "\n" + b.text + "\n" + fast
            in_tokens += estimate_tokens(inp_slow, encoder)
            slow = extract_summary(inp_slow, max_tokens=slow_tokens, mode="fusion", encoder=encoder)
            out_tokens += estimate_tokens(slow, encoder)
            calls += 1

        # consistency check: if fast loses too much overlap with slow, reinforce slow into fast
        if slow and jaccard(fast, slow) < 0.15:
            inp_fix = slow + "\n" + fast
            in_tokens += estimate_tokens(inp_fix, encoder)
            fast = extract_summary(inp_fix, max_tokens=main_tokens, mode="fusion", encoder=encoder)
            out_tokens += estimate_tokens(fast, encoder)
            calls += 1

        recent.append(b.text)
        if len(recent) > recent_keep:
            recent.pop(0)

    main = (slow + "\n" + fast).strip()
    dt = (time.perf_counter() - t0) * 1000.0
    return StrategyResult(
        name="recursive_dual_rate",
        main_memory=main,
        recent_blocks=recent,
        calls=calls,
        in_tokens=in_tokens,
        out_tokens=out_tokens,
        time_ms=dt,
    )




def run_baseline_snapshots(blocks: List[Block], recent_keep: int) -> StrategySnapshots:
    recent: List[str] = []
    contexts: List[str] = []
    for b in blocks:
        recent.append(b.text)
        if len(recent) > recent_keep:
            recent.pop(0)
        context = ("\n".join(recent)).strip()
        contexts.append(context)
    return StrategySnapshots(name="baseline_sliding", contexts=contexts)


def run_recursive_snapshots(
    blocks: List[Block], main_tokens: int, recent_keep: int, encoder=None
) -> StrategySnapshots:
    main = ""
    recent: List[str] = []
    contexts: List[str] = []
    for b in blocks:
        main = extract_summary(main + "\n" + b.text, max_tokens=main_tokens, mode="fusion", encoder=encoder)
        recent.append(b.text)
        if len(recent) > recent_keep:
            recent.pop(0)
        context = (main + "\n" + "\n".join(recent)).strip()
        contexts.append(context)
    return StrategySnapshots(name="recursive_summary", contexts=contexts)


def run_recursive_dual_rate_snapshots(
    blocks: List[Block],
    main_tokens: int,
    recent_keep: int,
    slow_tokens: int,
    slow_update_every: int,
    importance_threshold: float,
    fast_update_every: int,
    fast_importance: float,
    encoder=None,
) -> StrategySnapshots:
    fast = ""
    slow = ""
    recent: List[str] = []
    contexts: List[str] = []

    for b in blocks:
        do_fast = False
        if fast_update_every > 0 and (b.idx % fast_update_every == 0):
            do_fast = True
        if importance_score(b.text) >= fast_importance:
            do_fast = True

        if do_fast:
            fast = extract_summary(fast + "\n" + b.text, max_tokens=main_tokens, mode="fusion", encoder=encoder)

        do_slow = False
        if slow_update_every > 0 and (b.idx % slow_update_every == 0):
            do_slow = True
        if importance_score(b.text) >= importance_threshold:
            do_slow = True

        if do_slow:
            slow = extract_summary(slow + "\n" + b.text + "\n" + fast, max_tokens=slow_tokens, mode="fusion", encoder=encoder)

        if slow and jaccard(fast, slow) < 0.15:
            fast = extract_summary(slow + "\n" + fast, max_tokens=main_tokens, mode="fusion", encoder=encoder)

        recent.append(b.text)
        if len(recent) > recent_keep:
            recent.pop(0)

        context = (slow + "\n" + fast + "\n" + "\n".join(recent)).strip()
        contexts.append(context)

    return StrategySnapshots(name="recursive_dual_rate", contexts=contexts)


# ----------------------------- Evaluation -----------------------------

@dataclass
class FactItem:
    fact: str
    answers: List[str]
    q: str
    typ: str
    conv_idx: int | None = None


def load_facts(path: str) -> List[FactItem]:
    items: List[FactItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            fact = obj.get("fact", "").strip()
            if not fact:
                continue
            answers = obj.get("answers")
            if isinstance(answers, str):
                answers = [answers]
            if not answers:
                answers = [fact]
            q = obj.get("question", "")
            typ = obj.get("type", "")
            conv_idx = obj.get("conv_idx")
            if isinstance(conv_idx, int):
                conv_val = conv_idx
            else:
                conv_val = None
            items.append(
                FactItem(
                    fact=fact,
                    answers=[a.strip() for a in answers],
                    q=q,
                    typ=typ,
                    conv_idx=conv_val,
                )
            )
    return items


def normalize_text(s: str) -> str:
    s = s.lower()
    s = re.sub(r"\s+", " ", s, flags=re.UNICODE).strip()
    return s


def contains_any(context: str, answers: List[str]) -> bool:
    ctx = normalize_text(context)
    for a in answers:
        if normalize_text(a) in ctx:
            return True
    return False


def token_coverage_match(context: str, fact: str, threshold: float) -> bool:
    ctx_tokens = set(
        w for w in re.findall(r"\w+", context.lower(), flags=re.UNICODE) if w not in STOPWORDS
    )
    fact_tokens = [
        w for w in re.findall(r"\w+", fact.lower(), flags=re.UNICODE) if w not in STOPWORDS
    ]
    if not fact_tokens:
        return False
    covered = sum(1 for w in fact_tokens if w in ctx_tokens)
    return (covered / len(fact_tokens)) >= threshold


def evaluate(strategy: StrategyResult, facts: List[FactItem], threshold: float) -> Dict[str, float]:
    context = (strategy.main_memory + "\n" + "\n".join(strategy.recent_blocks)).strip()
    if not context:
        return {"recall": 0.0, "total": len(facts)}
    hit = 0
    for f in facts:
        if contains_any(context, f.answers) or token_coverage_match(context, f.fact, threshold):
            hit += 1
    recall = hit / max(len(facts), 1)
    return {"recall": recall, "total": len(facts)}


def locate_fact_block(blocks: List[Block], fact: str, threshold: float) -> Optional[int]:
    for b in blocks:
        if contains_any(b.text, [fact]) or token_coverage_match(b.text, fact, threshold):
            return b.idx
    return None


# ----------------------------- CLI -----------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate long-memory strategies with recursive summaries.")
    parser.add_argument("--transcript", required=True, help="Path to transcript text file")
    parser.add_argument("--facts", required=True, help="Path to facts.jsonl")
    parser.add_argument("--chunk-tokens", type=int, default=700, help="Tokens per chunk")
    parser.add_argument("--main-tokens", type=int, default=250, help="Max tokens for main memory")
    parser.add_argument("--recent-keep", type=int, default=1, help="How many recent raw blocks to keep")
    parser.add_argument("--match-threshold", type=float, default=0.6, help="Token coverage threshold for fuzzy matching")
    parser.add_argument("--min-blocks", type=int, default=2, help="Skip conversations with fewer blocks")
    parser.add_argument("--max-delay", type=int, default=4, help="Max delay (in blocks) for retention curve")
    parser.add_argument("--slow-tokens", type=int, default=200, help="Max tokens for slow memory")
    parser.add_argument("--slow-update-every", type=int, default=4, help="Update slow memory every N blocks")
    parser.add_argument("--slow-importance", type=float, default=3.0, help="Importance threshold to update slow memory")
    parser.add_argument("--fast-update-every", type=int, default=0, help="Update fast memory every N blocks (0=disable schedule)")
    parser.add_argument("--fast-importance", type=float, default=2.5, help="Importance threshold to update fast memory")
    parser.add_argument("--dump", action="store_true", help="Dump strategy contexts to stdout")

    args = parser.parse_args()

    tiktoken = _try_tiktoken()
    encoder = None
    if tiktoken is not None:
        try:
            encoder = tiktoken.get_encoding("cl100k_base")
        except Exception:
            encoder = None

    conversations = parse_transcript_conversations(args.transcript)
    if not conversations:
        print("No conversations parsed. Check transcript format.")
        return 1

    facts = load_facts(args.facts)
    if not facts:
        print("No facts loaded. Check facts.jsonl.")
        return 1

    # Group facts per conversation
    facts_by_conv: List[List[FactItem]] = [[] for _ in range(len(conversations))]
    if any(f.conv_idx is not None for f in facts):
        for f in facts:
            if f.conv_idx is None:
                continue
            if 0 <= f.conv_idx < len(conversations):
                facts_by_conv[f.conv_idx].append(f)
    else:
        # Fallback: distribute evenly by order
        per = max(len(facts) // max(len(conversations), 1), 1)
        for i, f in enumerate(facts):
            conv_i = min(i // per, len(conversations) - 1)
            facts_by_conv[conv_i].append(f)

    print("Conversations:", len(conversations))
    print("Facts:", len(facts))
    print("")

    strategy_names = [
        "baseline_sliding",
        "recursive_summary",
        "recursive_dual_rate",
    ]
    totals = {name: 0.0 for name in strategy_names}
    counts = {name: 0 for name in strategy_names}
    cost_calls = {name: 0 for name in strategy_names}
    cost_in = {name: 0 for name in strategy_names}
    cost_out = {name: 0 for name in strategy_names}
    sum_time = {name: 0.0 for name in strategy_names}

    for conv_idx, convo_text in enumerate(conversations):
        blocks = chunk_by_tokens(convo_text, args.chunk_tokens, encoder)
        if not blocks or len(blocks) < args.min_blocks:
            continue
        local_facts = facts_by_conv[conv_idx]
        if not local_facts:
            continue

        results = [
            run_baseline(blocks, args.recent_keep),
            run_recursive(blocks, args.main_tokens, args.recent_keep, encoder),
            run_recursive_dual_rate(
                blocks,
                args.main_tokens,
                args.recent_keep,
                args.slow_tokens,
                args.slow_update_every,
                args.slow_importance,
                args.fast_update_every,
                args.fast_importance,
                encoder,
            ),
        ]

        for r in results:
            metrics = evaluate(r, local_facts, args.match_threshold)
            totals[r.name] += metrics["recall"]
            counts[r.name] += 1
            cost_calls[r.name] += r.calls
            cost_in[r.name] += r.in_tokens
            cost_out[r.name] += r.out_tokens
            sum_time[r.name] += r.time_ms

            if args.dump:
                context = (r.main_memory + "\n" + "\n".join(r.recent_blocks)).strip()
                print(f"--- convo {conv_idx+1} | {r.name} context start ---")
                print(context)
                print(f"--- convo {conv_idx+1} | {r.name} context end ---")

    # Delay retention curves
    delay_totals = {name: [0.0 for _ in range(args.max_delay + 1)] for name in strategy_names}
    delay_counts = {name: [0 for _ in range(args.max_delay + 1)] for name in strategy_names}

    for conv_idx, convo_text in enumerate(conversations):
        blocks = chunk_by_tokens(convo_text, args.chunk_tokens, encoder)
        if not blocks or len(blocks) < args.min_blocks:
            continue
        local_facts = facts_by_conv[conv_idx]
        if not local_facts:
            continue

        snapshots = [
            run_baseline_snapshots(blocks, args.recent_keep),
            run_recursive_snapshots(blocks, args.main_tokens, args.recent_keep, encoder),
            run_recursive_dual_rate_snapshots(
                blocks,
                args.main_tokens,
                args.recent_keep,
                args.slow_tokens,
                args.slow_update_every,
                args.slow_importance,
                args.fast_update_every,
                args.fast_importance,
                encoder,
            ),
        ]

        for f in local_facts:
            fact_block = locate_fact_block(blocks, f.fact, args.match_threshold)
            if fact_block is None:
                continue
            for delay in range(1, args.max_delay + 1):
                target_block = fact_block + delay
                if target_block > len(blocks):
                    continue
                snap_index = target_block - 1
                for snap in snapshots:
                    context = snap.contexts[snap_index]
                    hit = contains_any(context, f.answers) or token_coverage_match(
                        context, f.fact, args.match_threshold
                    )
                    delay_totals[snap.name][delay] += 1.0 if hit else 0.0
                    delay_counts[snap.name][delay] += 1

    for name in strategy_names:
        if counts[name] == 0:
            avg = 0.0
        else:
            avg = totals[name] / counts[name]
        print(f"{name}: avg_recall={avg:.3f} convos={counts[name]}")
        if counts[name] > 0:
            avg_calls = cost_calls[name] / counts[name]
            avg_in = cost_in[name] / counts[name]
            avg_out = cost_out[name] / counts[name]
            avg_time = sum_time.get(name, 0.0) / counts[name]
            print(f"  cost_avg: calls={avg_calls:.1f} in_tokens={avg_in:.0f} out_tokens={avg_out:.0f} time_ms={avg_time:.1f}")

    print("")
    print("Delay retention (avg recall by delay in blocks):")
    for name in strategy_names:
        parts = []
        for delay in range(1, args.max_delay + 1):
            if delay_counts[name][delay] == 0:
                val = 0.0
            else:
                val = delay_totals[name][delay] / delay_counts[name][delay]
            parts.append(f"d{delay}={val:.3f}")
        print(f"{name}: " + " ".join(parts))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
