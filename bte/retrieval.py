"""Temporal-validity-aware retrieval (proposed-model.md section 4,
following Hindsight's channel pattern): semantic (embedding cosine),
keyword (Okapi BM25), graph (spreading activation over the
justification structure and shared entities), temporal (validity at the
query's reference time, recency-ranked). Fused with reciprocal rank
fusion. Every channel sees only active edges, and retrieve() filters on
validity at the reference time, so answers automatically reflect
whatever BBP has propagated — no separate consistency check at query
time. Cross-encoder reranking is a pluggable later addition; RRF alone
keeps the module fair-and-fixed across compared systems.

A fifth, optional state-basis channel exists because similarity
channels are premise-locked: a query phrased in a stale fact's own
vocabulary retrieves that fact and its lexical neighbors, never the
newer cross-domain facts that invalidate its premise (observed on STALE
T2 — a "limit my commitments" query retrieved the old health facts
while the ACTIVE overnight-work-schedule facts ranked in no channel;
the STALE paper diagnoses the same failure class as "visibility does
not imply authority" and answers it with premise-centered retrieval,
arXiv 2605.06527 section 5). Here it is graph-native and deterministic:
take the semantic channel's top hits as premise anchors, walk the
shared domain-dependency graph BACKWARD (domains.affected_by) to the
domains whose changes bear on each anchor, and surface the subject's
most recent active facts there. No LLM calls; the same
DomainDependencies instance the conflict detector learns into.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from pathlib import Path
from typing import Optional, Protocol

from openai import OpenAI

from ._util import atomic_write_text
from .domains import DomainDependencies
from .graph import BJG, Edge

# Edges that became inactive with no successor edge to carry a
# "(previously: X)" annotation (see Retriever's module docstring and
# _orphaned_invalidations): direct supersession always creates a winner
# edge whose .supersedes points back, so render_fact's existing walk
# covers it. Two paths leave no such winner - a derived edge invalidated
# by BBP propagation through its justification (no new edge asserts a
# replacement conclusion), and a bare retraction (no replacement value
# at all) - and the loser then vanishes from retrieval silently. Probed
# on the diagnostic probe (res-d4-ass-upda-adj-hi-r1): the graph
# correctly retired the derived conclusion, but retrieval still
# returned the surviving raw component facts, and the reader re-chained
# them into the exact conclusion BBP had just retired.
INVALIDATED_PREFIX = "INVALIDATED"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def edge_text(e: Edge) -> str:
    return f"{e.subject.replace('_', ' ')} {e.relation.replace('_', ' ')} {e.object}"


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...


class HashEmbedder:
    """Deterministic bag-of-hashed-tokens embedder for LLM-free tests."""

    dim = 128

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for tok in tokenize(t):
                h = int(hashlib.sha256(tok.encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0
            norm = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / norm for x in v])
        return out


class CachedEmbedder:
    def __init__(self, model: str = "text-embedding-3-small",
                 base_url: Optional[str] = None,
                 api_key_env: str = "OPENAI_API_KEY",
                 cache_dir: str = ".cache/emb") -> None:
        self.model = model
        self.cache_dir = Path(cache_dir)
        self._client = OpenAI(base_url=base_url,
                              api_key=os.environ.get(api_key_env))

    def _path(self, text: str) -> Path:
        digest = hashlib.sha256(f"{self.model}\x00{text}".encode()).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[Optional[list[float]]] = [None] * len(texts)
        missing = []
        for i, t in enumerate(texts):
            p = self._path(t)
            if p.exists():
                try:
                    out[i] = json.loads(p.read_text())
                    continue
                except json.JSONDecodeError:
                    pass  # corrupted cache entry: regenerate below
            missing.append(i)
        if missing:
            resp = self._client.embeddings.create(
                model=self.model, input=[texts[i] for i in missing])
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            for slot, item in zip(missing, resp.data):
                out[slot] = item.embedding
                atomic_write_text(self._path(texts[slot]),
                                  json.dumps(item.embedding))
        return out  # type: ignore[return-value]


def cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    da = math.sqrt(sum(x * x for x in a)) or 1.0
    db = math.sqrt(sum(x * x for x in b)) or 1.0
    return num / (da * db)


class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs: list[list[str]] = []

    def fit(self, docs: list[list[str]]) -> None:
        self.docs = docs
        self.n = len(docs)
        self.avgdl = (sum(len(d) for d in docs) / self.n) if self.n else 0.0
        self.df: dict[str, int] = {}
        for d in docs:
            for tok in set(d):
                self.df[tok] = self.df.get(tok, 0) + 1

    def scores(self, query_tokens: list[str]) -> list[float]:
        out = []
        for d in self.docs:
            s = 0.0
            for q in query_tokens:
                tf = d.count(q)
                if not tf:
                    continue
                df = self.df.get(q, 0)
                idf = math.log((self.n - df + 0.5) / (df + 0.5) + 1.0)
                s += idf * tf * (self.k1 + 1) / (
                    tf + self.k1 * (1 - self.b + self.b * len(d) / self.avgdl))
            out.append(s)
        return out


class Retriever:
    def __init__(self, graph: BJG, embedder: Optional[Embedder] = None,
                 per_channel: int = 10, rrf_k: int = 60,
                 domain_deps: Optional[DomainDependencies] = None,
                 basis_seeds: int = 3,
                 basis_per_domain: int = 3) -> None:
        self.graph = graph
        self.embedder = embedder or HashEmbedder()
        self.per_channel = per_channel
        self.rrf_k = rrf_k
        # None disables the state-basis channel (default: the other four
        # channels behave exactly as before)
        self.domain_deps = domain_deps
        self.basis_seeds = basis_seeds
        self.basis_per_domain = basis_per_domain
        self._ids: list[str] = []
        self._tokens: list[list[str]] = []
        self._vecs: list[list[float]] = []
        self._bm25 = BM25()

    # -- index ----------------------------------------------------------

    def index(self) -> None:
        edges = [e for e in self.graph.edges.values()
                 if self.graph.is_active(e.id)]
        edges += self._orphaned_invalidations()
        self._ids = [e.id for e in edges]
        texts = [edge_text(e) for e in edges]
        self._tokens = [tokenize(t) for t in texts]
        self._bm25.fit(self._tokens)
        self._vecs = self.embedder.embed(texts) if edges else []

    def _orphaned_invalidations(self) -> list[Edge]:
        """Inactive edges with no successor - see the module-level
        comment on INVALIDATED_PREFIX. edge_text() still embeds/tokenizes
        them normally (their content is exactly what a question about
        the retired conclusion would match), and render_fact renders
        them as an explicit retraction instead of a live fact."""
        superseded_ids = {e.supersedes for e in self.graph.edges.values()
                          if e.supersedes}
        return [e for e in self.graph.edges.values()
                if not self.graph.is_active(e.id) and e.id not in superseded_ids]

    # -- channels (each returns edge ids, best first) --------------------

    def _top(self, scores: list[float]) -> list[str]:
        order = sorted(range(len(scores)), key=lambda i: -scores[i])
        return [self._ids[i] for i in order[:self.per_channel]
                if scores[i] > 0]

    def semantic(self, query: str) -> list[str]:
        if not self._ids:
            return []
        q = self.embedder.embed([query])[0]
        return self._top([cosine(q, v) for v in self._vecs])

    def keyword(self, query: str) -> list[str]:
        if not self._ids:
            return []
        return self._top(self._bm25.scores(tokenize(query)))

    def graph_channel(self, query: str) -> list[str]:
        qtok = set(tokenize(query))
        activation: dict[str, float] = {}
        for eid, toks in zip(self._ids, self._tokens):
            if qtok & set(toks):
                activation[eid] = 1.0
        frontier = dict(activation)
        for _ in range(2):  # two hops of spreading, halving each hop
            nxt: dict[str, float] = {}
            for eid, a in frontier.items():
                edge = self.graph.edges[eid]
                neighbors = set(self.graph.dependents(eid))
                for alt in edge.justification:
                    neighbors |= alt
                for n in neighbors:
                    if n in self.graph.edges and self.graph.is_active(n):
                        spread = a * 0.5
                        if spread > activation.get(n, 0.0):
                            nxt[n] = spread
            activation.update(nxt)
            frontier = nxt
        ranked = sorted(activation, key=lambda e: -activation[e])
        return ranked[:self.per_channel]

    def state_basis(self, query: str) -> list[str]:
        """See the module docstring: current active facts from the
        domains that bear on the query's premise-matching facts.

        Ordering is structural, not global recency - a first version
        ranked the pooled basis by t_transaction and the newest facts
        of ANY affected domain (shopping, phone trouble) crowded out
        the anchor's actual state change several sessions older. Now:
        anchors in semantic rank order; each anchor's source domains
        ordered by how often the detector CONFIRMED that domain
        invalidating the anchor's (learned counts - write-time evidence
        steering query-time recovery), prior-only sources after; a
        per-domain quota with one most-recent fact per relation, so one
        noisy domain never exhausts the channel."""
        if self.domain_deps is None or not self._ids:
            return []
        # seeds may reappear in the basis (a fact can be both premise
        # anchor and current state); RRF fusion handles the overlap
        seeds = self.semantic(query)[:self.basis_seeds]
        out: list[str] = []
        emitted: set[str] = set()
        for sid in seeds:
            seed = self.graph.edges[sid]
            sources = self.domain_deps.affected_by(seed.domain)
            if not sources:
                continue
            learned = self.domain_deps.learned
            for src in sorted(
                    sources,
                    key=lambda s: (-learned.get((s, seed.domain), 0), s)):
                newest_per_rel: dict[str, Edge] = {}
                for cid in self._ids:
                    c = self.graph.edges[cid]
                    if c.subject != seed.subject or c.domain != src:
                        continue
                    cur = newest_per_rel.get(c.relation)
                    if cur is None or (c.t_transaction or "") > \
                            (cur.t_transaction or ""):
                        newest_per_rel[c.relation] = c
                picked = sorted(newest_per_rel.values(),
                                key=lambda e: e.t_transaction or "",
                                reverse=True)[:self.basis_per_domain]
                for e in picked:
                    if e.id not in emitted:
                        emitted.add(e.id)
                        out.append(e.id)
        return out[:self.per_channel]

    def temporal(self, reference_time: Optional[str]) -> list[str]:
        candidates = [eid for eid in self._ids
                      if self._valid_at(eid, reference_time)]
        candidates.sort(
            key=lambda eid: self.graph.edges[eid].t_transaction or "",
            reverse=True)
        return candidates[:self.per_channel]

    # -- fusion -----------------------------------------------------------

    def _valid_at(self, eid: str, ref: Optional[str]) -> bool:
        if ref is None:
            return True
        e = self.graph.edges[eid]
        if e.t_valid_start is not None and ref < e.t_valid_start:
            return False
        if e.t_valid_end is not None and ref >= e.t_valid_end:
            return False
        return True

    def retrieve(self, query: str, reference_time: Optional[str] = None,
                 k: int = 8) -> list[Edge]:
        rankings = [
            self.semantic(query),
            self.keyword(query),
            self.graph_channel(query),
            self.temporal(reference_time),
            self.state_basis(query),
        ]
        fused: dict[str, float] = {}
        for ranking in rankings:
            for rank, eid in enumerate(ranking):
                fused[eid] = fused.get(eid, 0.0) + 1.0 / (self.rrf_k + rank)
        ranked = sorted(fused, key=lambda e: -fused[e])
        out = [self.graph.edges[eid] for eid in ranked
               if self._valid_at(eid, reference_time)]
        return out[:k]
