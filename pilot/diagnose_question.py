"""Rebuild one LongMemEval question's graph from the LLM cache (free)
and report where the pipeline lost the answer: extraction (gold value
never entered the graph), retrieval (in graph, not retrieved), or
reader (retrieved, wrong answer).

Usage: uv run python pilot/diagnose_question.py <question_id> <keyword> [...]
"""

import sys

from bte.bench import load_longmemeval, render_session
from bte.conflict import ConflictDetector, make_llm_adjudicator
from bte.extraction import extract_facts
from bte.ingest import Ingestor
from bte.llm import CachedLLM
from bte.memory import BJGMemory
from bte.retrieval import CachedEmbedder, Retriever

DATA = "data/longmemeval/longmemeval_s_cleaned.json"


def main():
    qid, keywords = sys.argv[1], [k.lower() for k in sys.argv[2:]]
    q = next(x for x in load_longmemeval(DATA)
             if str(x["question_id"]) == qid)
    extractor = CachedLLM(model="deepseek/deepseek-v3.2",
                          base_url="https://openrouter.ai/api/v1",
                          api_key_env="OPENROUTER_API_KEY")
    reader = CachedLLM(model="qwen/qwen3.5-9b",
                       base_url="https://openrouter.ai/api/v1",
                       api_key_env="OPENROUTER_API_KEY")
    ing = Ingestor(
        extract=lambda text, ts, ctx: extract_facts(extractor, text, ts, ctx),
        detector=ConflictDetector(adjudicate=make_llm_adjudicator(extractor)),
    )
    mem = BJGMemory(ing, Retriever(ing.graph, CachedEmbedder()),
                    reader=reader.complete_text)
    dates = q.get("haystack_dates") or ["unknown"] * len(q["haystack_sessions"])
    for session, date in zip(q["haystack_sessions"], dates):
        mem.ingest_session([render_session(session)], date or "unknown")

    print(f"question: {q['question']}")
    print(f"gold: {q['answer']}")
    print(f"graph: {len(ing.graph.edges)} edges, "
          f"{sum(1 for e in ing.graph.edges if ing.graph.is_active(e))} active")

    print("\nedges matching keywords (active and superseded):")
    hits = 0
    for eid, e in ing.graph.edges.items():
        text = f"{e.subject} {e.relation} {e.object}".lower()
        if any(k in text for k in keywords):
            s = ing.graph.status(eid)
            flag = "ACTIVE" if ing.graph.is_active(eid) else "superseded"
            print(f"  ({e.subject} | {e.relation} | {e.object}) "
                  f"[{flag}] t={e.t_transaction}")
            hits += 1
    if not hits:
        print("  NONE - extraction never captured the gold fact")

    print("\nretrieved for the question:")
    for e in mem.retriever.retrieve(q["question"],
                                    q.get("question_date"), k=8):
        print(f"  ({e.subject} | {e.relation} | {e.object})")
    print("\nanswer:", mem.answer(q["question"], q.get("question_date")))


if __name__ == "__main__":
    main()
