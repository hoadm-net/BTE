"""Full-stack smoke test (Phase 2 exit criterion): probe conversations
through real extraction (DeepSeek-V3.2), the BJG with rules and BBP,
four-channel retrieval with real embeddings, and the fixed reader
(Qwen3.5-9B). Asks the probe question before and after the contradiction
session and compares with gold.

Usage: uv run python pilot/e2e_probe_demo.py
Writes .plan/results/pilot/e2e_probe_demo.md
"""

from bte.conflict import ConflictDetector, make_llm_adjudicator
from bte.extraction import extract_facts
from bte.ingest import Ingestor
from bte.llm import CachedLLM
from bte.memory import BJGMemory
from bte.probe import DOMAINS, domain_rules, generate
from bte.retrieval import CachedEmbedder, Retriever

PICKS = ["emp-d2-ass-corr-adj-hi-r0", "res-d3-ass-upda-dis-hi-r1",
         "tra-d2-der-upda-adj-hi-r2"]

DOMAIN_BY_NAME = {d.name: d for d in DOMAINS}


def main():
    extractor = CachedLLM(model="deepseek/deepseek-v3.2",
                          base_url="https://openrouter.ai/api/v1",
                          api_key_env="OPENROUTER_API_KEY")
    reader_llm = CachedLLM(model="qwen/qwen3.5-9b",
                           base_url="https://openrouter.ai/api/v1",
                           api_key_env="OPENROUTER_API_KEY")
    items = {i.probe_id: i for i in generate()}
    lines = ["# Full-stack probe smoke (extraction + retrieval + reader)\n"]
    for pid in PICKS:
        item = items[pid]
        d = DOMAIN_BY_NAME[item.domain]
        ing = Ingestor(
            extract=lambda text, ts, ctx: extract_facts(
                extractor, text, ts, ctx),
            detector=ConflictDetector(
                adjudicate=make_llm_adjudicator(extractor)),
            rules=domain_rules(d),
        )
        mem = BJGMemory(ing, Retriever(ing.graph, CachedEmbedder()),
                        reader=reader_llm.complete_text)
        lines.append(f"\n## {pid}  (gold pre: {item.gold_pre} | "
                     f"gold post: {item.gold_post})\n")
        n = len(item.sessions)
        for i, session in enumerate(item.sessions[:n - 1]):
            mem.ingest_session(session, f"2026-06-{i + 1:02d}")
        pre = mem.answer(item.question, reference_time=f"2026-06-{n:02d}")
        lines.append(f"- question: {item.question}")
        lines.append(f"- answer BEFORE contradiction: {pre}")
        mem.ingest_session(item.sessions[n - 1], f"2026-06-{n:02d}")
        post = mem.answer(item.question,
                          reference_time=f"2026-06-{n + 1:02d}")
        lines.append(f"- contradiction: {item.sessions[n - 1][0]}")
        lines.append(f"- answer AFTER contradiction: {post}")

    path = ".plan/results/pilot/e2e_probe_demo.md"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {path}")
    print(f"extractor calls {extractor.calls} hits {extractor.cache_hits}; "
          f"reader calls {reader_llm.calls} hits {reader_llm.cache_hits}")


if __name__ == "__main__":
    main()
