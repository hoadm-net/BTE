"""Extraction bake-off over the diagnostic probe (docs/models.md 07/2026
revision): run each candidate model's extraction on every probe item's
text sessions, replay through the ingestion pipeline with the domain's
rules, and score against the probe's replay-validated gold labels.

Metrics:
- fact P/R/F1: extracted vs oracle facts on statement/filler sessions,
  keyed by normalized (subject, relation, object);
- axis accuracy: predicted update/correction on the contradiction turn;
- retraction recall: retraction emitted when gold expects one;
- pre/post state accuracy: probed relation's active value before and
  after the contradiction (the end-to-end metric);
- post accuracy by hop depth (H1's shape, extraction-conditional).

Usage: uv run python pilot/bakeoff_extraction.py [--limit N]
Writes .plan/results/bakeoff_extraction_v0.{json,md}
"""

import argparse
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

from bte.extraction import extract_facts
from bte.ingest import Ingestor
from bte.llm import CachedLLM
from bte.probe import DOMAINS, domain_rules, generate

DOMAIN_BY_NAME = {d.name: d for d in DOMAINS}

CANDIDATES = [
    {"label": "gpt-4o-mini", "model": "gpt-4o-mini",
     "base_url": None, "api_key_env": "OPENAI_API_KEY"},
    {"label": "gpt-5.4-nano", "model": "gpt-5.4-nano",
     "base_url": None, "api_key_env": "OPENAI_API_KEY",
     "extra": {"reasoning_effort": "none"}},
    {"label": "deepseek-v3.2", "model": "deepseek/deepseek-v3.2",
     "base_url": "https://openrouter.ai/api/v1",
     "api_key_env": "OPENROUTER_API_KEY"},
]


def fact_key(f):
    return (f["subject"], f["relation"], f["object"].strip().lower())


def probed_relation(item):
    d = DOMAIN_BY_NAME[item.domain]
    return d.relations[0] if item.hop_depth == 1 \
        else d.conclusions[item.hop_depth - 2]


def active_values(ing, item):
    return sorted(e.object for e in
                  ing.graph.find(subject="user",
                                 relation=probed_relation(item)))


def vocab_hint(item):
    d = DOMAIN_BY_NAME[item.domain]
    rels = ", ".join(d.relations + d.conclusions)
    return f"relation names to use when applicable: {rels}"


def run_item(llm, item):
    d = DOMAIN_BY_NAME[item.domain]
    ing = Ingestor(rules=domain_rules(d))
    rec = {"probe_id": item.probe_id, "tp": 0, "fp": 0, "fn": 0,
           "axis_pred": "none", "retraction_expected": False,
           "retraction_emitted": False, "error": None}
    n = len(item.sessions)
    try:
        for i in range(n):
            text = " ".join(item.sessions[i])
            context = [vocab_hint(item)] + [
                f"({e.subject}, {e.relation}, {e.object})"
                for e in ing.graph.find()
            ]
            payload = extract_facts(llm, text, f"2026-06-{i + 1:02d}",
                                    context)
            facts = payload.get("facts", [])
            retr = payload.get("retractions", [])
            last = i == n - 1
            if not last:
                got = {fact_key(f) for f in facts}
                want = {fact_key(f) for f in item.oracle_facts[i]}
                rec["tp"] += len(got & want)
                rec["fp"] += len(got - want)
                rec["fn"] += len(want - got)
            else:
                rec["retraction_expected"] = bool(
                    item.oracle_retractions[i])
                rec["retraction_emitted"] = bool(retr)
                if any(f.get("is_correction") for f in facts) or any(
                        r.get("was_wrong") for r in retr):
                    rec["axis_pred"] = "correction"
                elif facts or retr:
                    rec["axis_pred"] = "update"
                rec["pre_ok"] = active_values(ing, item) == [item.gold_pre]
            report = ing.ingest_facts(facts, f"t{i}")
            ing.apply_retractions(retr, report)
        post = active_values(ing, item)
        rec["post_ok"] = (post == [] if item.gold_post == "unknown"
                          else post == [item.gold_post])
    except Exception as exc:  # per-item isolation: one bad item != dead run
        rec["error"] = f"{type(exc).__name__}: {exc}"
        rec.setdefault("pre_ok", False)
        rec["post_ok"] = False
    return rec


def aggregate(label, items, records):
    by_id = {i.probe_id: i for i in items}
    tp = sum(r["tp"] for r in records)
    fp = sum(r["fp"] for r in records)
    fn = sum(r["fn"] for r in records)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec_ = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec_ / (prec + rec_) if prec + rec_ else 0.0
    axis_ok = sum(1 for r in records
                  if r["axis_pred"] == by_id[r["probe_id"]].gold_axis)
    retr = [r for r in records if r["retraction_expected"]]
    retr_ok = sum(1 for r in retr if r["retraction_emitted"])
    pre_ok = sum(1 for r in records if r.get("pre_ok"))
    post_ok = sum(1 for r in records if r.get("post_ok"))
    depth = defaultdict(lambda: [0, 0])
    for r in records:
        dep = by_id[r["probe_id"]].hop_depth
        depth[dep][1] += 1
        depth[dep][0] += bool(r.get("post_ok"))
    errors = [r for r in records if r["error"]]
    return {
        "label": label, "n": len(records),
        "fact_precision": round(prec, 3), "fact_recall": round(rec_, 3),
        "fact_f1": round(f1, 3),
        "axis_accuracy": round(axis_ok / len(records), 3),
        "retraction_recall":
            round(retr_ok / len(retr), 3) if retr else None,
        "pre_state_accuracy": round(pre_ok / len(records), 3),
        "post_state_accuracy": round(post_ok / len(records), 3),
        "post_by_depth": {
            k: f"{v[0]}/{v[1]}" for k, v in sorted(depth.items())},
        "errors": len(errors),
        "error_samples": [e["error"] for e in errors[:3]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    items = generate()
    if args.limit:
        items = items[:args.limit]
    summaries = []
    for cand in CANDIDATES:
        llm = CachedLLM(model=cand["model"], base_url=cand["base_url"],
                        api_key_env=cand["api_key_env"],
                        extra=cand.get("extra"))
        with ThreadPoolExecutor(max_workers=12) as pool:
            records = list(pool.map(lambda it: run_item(llm, it), items))
        s = aggregate(cand["label"], items, records)
        s["llm_calls"] = llm.calls
        s["cache_hits"] = llm.cache_hits
        summaries.append(s)
        print(json.dumps(s, indent=1))

    with open(".plan/results/bakeoff_extraction_v0.json", "w") as f:
        json.dump(summaries, f, indent=1)

    lines = ["# Extraction bake-off v0 (probe_v0, canned surface)\n"]
    cols = ["label", "fact_f1", "fact_precision", "fact_recall",
            "axis_accuracy", "retraction_recall", "pre_state_accuracy",
            "post_state_accuracy", "errors"]
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "---|" * len(cols))
    for s in summaries:
        lines.append("| " + " | ".join(str(s[c]) for c in cols) + " |")
    lines.append("\npost-state accuracy by hop depth:")
    for s in summaries:
        lines.append(f"- {s['label']}: {s['post_by_depth']}")
    with open(".plan/results/bakeoff_extraction_v0.md", "w") as f:
        f.write("\n".join(lines))
    print("wrote .plan/results/bakeoff_extraction_v0.{json,md}")


if __name__ == "__main__":
    main()
