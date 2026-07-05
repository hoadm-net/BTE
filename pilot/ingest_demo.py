"""End-to-end pipeline demo on the three smoke-test scenarios that Mem0
failed (1b.3): real LLM extraction + adjudication into the BJG, BBP on
each resolved conflict, then a state dump showing which facts remain
active. Counterpart to pilot/smoke_test_mem0.py.

Usage: uv run python pilot/ingest_demo.py
Writes .plan/results/pilot/ingest_demo_output.md
"""

from bte.conflict import ConflictDetector, make_llm_adjudicator
from bte.extraction import extract_facts
from bte.ingest import Ingestor
from bte.llm import CachedLLM

SCENARIOS = [
    ("correction-transaction-axis", [
        "I work at Acme Corp as a data analyst.",
        "Since I work at Acme, my health insurance is through Acme's provider, Aetna.",
        "Actually, I need to correct something: I never worked at Acme. I work at Apex Analytics - I mixed up the names earlier.",
    ]),
    ("update-valid-axis", [
        "I live in Seattle.",
        "My dentist is Dr. Lee, whose clinic is in Seattle - convenient since I live there.",
        "Big news: I moved to Austin last month, the relocation is complete.",
    ]),
    ("two-hop-chain", [
        "I'm training for the Boston Marathon next April.",
        "Because of the marathon training, I do my long runs every Saturday morning.",
        "Since my Saturday mornings are taken by long runs, never schedule meetings for me on Saturday mornings.",
        "Update: I've withdrawn from the marathon - my knee injury won't heal in time.",
    ]),
]


def main():
    llm = CachedLLM(model="gpt-4o-mini")
    lines = ["# BJG ingestion demo transcript\n"]
    for name, turns in SCENARIOS:
        ing = Ingestor(
            extract=lambda text, ts, ctx: extract_facts(llm, text, ts, ctx),
            detector=ConflictDetector(adjudicate=make_llm_adjudicator(llm)),
        )
        lines.append(f"\n## {name}\n")
        for i, turn in enumerate(turns):
            report = ing.ingest_text(turn, f"2026-07-0{i + 1}")
            lines.append(f"- turn: {turn}")
            for eid in report.asserted:
                e = ing.graph.edges[eid]
                lines.append(f"    asserted ({e.subject}, {e.relation}, {e.object})")
            for eid in report.derived:
                e = ing.graph.edges[eid]
                just = " | ".join(",".join(sorted(a)) for a in e.justification)
                lines.append(f"    derived  ({e.subject}, {e.relation}, {e.object}) <= {just}")
            for d in report.decisions:
                if d.conflict:
                    lines.append(
                        f"    conflict: {d.new_edge_id} vs {d.old_edge_id} "
                        f"axis={d.axis} via={d.via} inherited={d.inherited}")
            for p in report.propagations:
                lines.append(f"    bbp: changed={sorted(p.changed)} waves={p.waves}")

        lines.append("  final state:")
        for eid, e in ing.graph.edges.items():
            s = ing.graph.status(eid)
            flag = "ACTIVE" if ing.graph.is_active(eid) else "superseded"
            lines.append(
                f"    {eid} ({e.subject}, {e.relation}, {e.object}) "
                f"[{e.source_type}] sigma=({s.valid.name},{s.trans.name}) {flag}")

    path = ".plan/results/pilot/ingest_demo_output.md"
    with open(path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {path}")
    print(f"llm calls: {llm.calls}, cache hits: {llm.cache_hits}")


if __name__ == "__main__":
    main()
