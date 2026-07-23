"""Annotator tool for the diagnostic probe's human-validation step.

See data/probe/README.md for the review rubric this tool implements.

Usage:
  cd annotator && uv run --with-requirements requirements.txt \
      streamlit run app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
PROBE_PATH = ROOT / "data" / "probe" / "probe_v0.json"
REVIEWS_PATH = Path(__file__).resolve().parent / "reviews" / "verdicts.json"

CHECKS = [
    "Readable: turns plausible, nothing broken/garbled",
    "Chain sound: each turn connects to the exact entity the previous "
    "one introduced (hop depth >= 2 only)",
    "Contradiction unambiguous: the final turn targets one specific "
    "earlier fact, with no equally-plausible second reading",
    "gold_pre correct given only the turns before the last one",
    "gold_post correct given all the turns (including \"unknown\" "
    "when nothing supplies a replacement value)",
    "Question fit: unambiguous which fact it's asking for",
]


@st.cache_data
def load_probe() -> list[dict]:
    return json.loads(PROBE_PATH.read_text())


def load_verdicts() -> dict:
    if REVIEWS_PATH.exists():
        return json.loads(REVIEWS_PATH.read_text())
    return {}


def save_verdict(probe_id: str, verdict: dict) -> None:
    REVIEWS_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = load_verdicts()
    data[probe_id] = verdict
    REVIEWS_PATH.write_text(json.dumps(data, indent=1))


def render_sessions(item: dict) -> None:
    n = len(item["sessions"])
    for i, session in enumerate(item["sessions"]):
        is_last = i == n - 1
        label = f"Session {i + 1}" + (" — contradiction turn" if is_last else "")
        with st.container(border=True):
            st.caption(label)
            for msg in session:
                st.markdown(f"🗣️ {msg}")


def main():
    st.set_page_config(page_title="Probe Annotator", layout="wide")
    items = load_probe()
    verdicts = load_verdicts()

    st.sidebar.title("Probe review")
    st.sidebar.caption(f"All {len(items)} items")

    n_done = sum(1 for i in items if i["probe_id"] in verdicts)
    n_pass = sum(1 for i in items
                if verdicts.get(i["probe_id"], {}).get("verdict") == "PASS")
    n_fail = sum(1 for i in items
                if verdicts.get(i["probe_id"], {}).get("verdict") == "FAIL")
    st.sidebar.metric("Reviewed", f"{n_done} / {len(items)}")
    if n_done:
        st.sidebar.metric("Agreement (PASS rate)", f"{n_pass / n_done:.0%}")
    st.sidebar.caption(f"PASS {n_pass} · FAIL {n_fail}")

    if n_done:
        fail_lines = "\n".join(
            f"- `{i['probe_id']}`: "
            + "; ".join(verdicts[i["probe_id"]].get("failed_checks", []))
            + (f" — {verdicts[i['probe_id']]['reason']}"
               if verdicts[i["probe_id"]].get("reason") else "")
            for i in items
            if verdicts.get(i["probe_id"], {}).get("verdict") == "FAIL"
        )
        report = (
            f"# Probe human validation\n\n"
            f"Reviewed: {n_done} / {len(items)}\n"
            f"PASS: {n_pass}  FAIL: {n_fail}  "
            f"Agreement: {n_pass / n_done:.1%}\n\n"
            f"## FAIL items\n\n{fail_lines or '(none)'}\n"
        )
        st.sidebar.download_button(
            "Download report (.md)", report,
            file_name="probe_validation_report.md")

    st.sidebar.divider()
    filter_choice = st.sidebar.radio(
        "Show", ["All", "Pending only", "FAIL only"], index=0)
    domain_filter = st.sidebar.multiselect(
        "Domain", sorted({i["domain"] for i in items}))
    hop_filter = st.sidebar.multiselect(
        "Hop depth", sorted({i["hop_depth"] for i in items}))

    visible = items
    if filter_choice == "Pending only":
        visible = [i for i in visible if i["probe_id"] not in verdicts]
    elif filter_choice == "FAIL only":
        visible = [i for i in visible
                  if verdicts.get(i["probe_id"], {}).get("verdict") == "FAIL"]
    if domain_filter:
        visible = [i for i in visible if i["domain"] in domain_filter]
    if hop_filter:
        visible = [i for i in visible if i["hop_depth"] in hop_filter]

    st.sidebar.divider()
    labels = [
        f"{'✅' if v.get(i['probe_id'], {}).get('verdict') == 'PASS' else '❌' if v.get(i['probe_id'], {}).get('verdict') == 'FAIL' else '⬜'} "
        f"{i['probe_id']}"
        for i, v in ((i, verdicts) for i in visible)
    ]
    if not visible:
        st.sidebar.info("No items match the current filter.")
        st.info("No items match the current filter.")
        return
    idx = st.sidebar.radio("Item", range(len(visible)),
                           format_func=lambda k: labels[k], index=0)
    item = visible[idx]

    st.title(item["probe_id"])
    meta_cols = st.columns(6)
    for col, (label, value) in zip(meta_cols, [
        ("Domain", item["domain"]), ("Hop depth", item["hop_depth"]),
        ("Contradicted", item["contradicted"]), ("Axis", item["axis"]),
        ("Density", item["density"]), ("Confidence", item["confidence"]),
    ]):
        col.metric(label, value)

    left, right = st.columns([3, 2])
    with left:
        st.subheader("Conversation")
        render_sessions(item)

    with right:
        st.subheader("Question")
        st.info(item["question"])

        st.subheader("Gold answers")
        st.markdown(f"**Before the last turn (`gold_pre`):** {item['gold_pre']}")
        st.markdown(f"**After the last turn (`gold_post`):** {item['gold_post']}")
        if item.get("gold_invalidated"):
            st.caption("Relations expected stale after: "
                      + ", ".join(item["gold_invalidated"]))

        st.divider()
        st.subheader("Your review")
        existing = verdicts.get(item["probe_id"], {})

        with st.form(key=f"form-{item['probe_id']}"):
            verdict = st.radio(
                "Verdict", ["PASS", "FAIL"],
                index=0 if existing.get("verdict") != "FAIL" else 1,
                horizontal=True)
            failed_checks = st.multiselect(
                "If FAIL, which check(s)?", CHECKS,
                default=existing.get("failed_checks", []))
            reason = st.text_area(
                "Reason / notes", value=existing.get("reason", ""),
                placeholder="One line is enough for a clean PASS; for "
                           "FAIL, say what you'd change.")
            submitted = st.form_submit_button("Save")
            if submitted:
                save_verdict(item["probe_id"], {
                    "verdict": verdict,
                    "failed_checks": failed_checks if verdict == "FAIL" else [],
                    "reason": reason,
                })
                st.success("Saved.")
                st.rerun()

    with st.expander("Review rubric (data/probe/README.md, section 4)"):
        for i, c in enumerate(CHECKS, 1):
            st.markdown(f"{i}. {c}")


if __name__ == "__main__":
    main()
