# Probe annotator tool

A small Streamlit app for the diagnostic probe's human-validation
step. It is separate from the main `bte` package on purpose — a review
UI has nothing to do with the research code, so it keeps its own
dependency (Streamlit) out of `pyproject.toml`.

For what to actually check in each item, read
[`../data/probe/README.md`](../data/probe/README.md) first — this tool
just makes that checklist faster to apply than reading raw JSON.

## Run it

```
cd annotator
uv run --with-requirements requirements.txt streamlit run app.py
```

Opens in your browser. It loads all 168 items from
`data/probe/probe_v0.json` - the set is small enough to review in
full, so there's no sampling step.

## Using it

- The sidebar lists every item, with a status icon (⬜ pending,
  ✅ PASS, ❌ FAIL) and your running PASS/FAIL/agreement counts.
- Pick an item to see the full conversation on the left (session by
  session, last one marked as the contradiction turn) and the
  question + gold answers on the right.
- Record a PASS/FAIL verdict, optionally note which of the six rubric
  checks failed and why, and hit Save. Verdicts are written to
  `reviews/verdicts.json` as you go, so you can close the app and
  resume later without losing anything.
- Filter to "Pending only" to keep working through what's left, or
  "FAIL only" to review what you've flagged.
- Once you've reviewed enough items, use "Download report" in the
  sidebar for a ready-to-share summary (counts + the FAIL list with
  reasons) in the format `data/probe/README.md` asks for.

`reviews/` is gitignored — it's your personal working file, not
something to commit. Hand the downloaded report back instead.
