"""OpenAI-compatible chat client with a content-addressed disk cache.

Requests default to temperature 0 (reasoning-tier models take
temperature=None to omit the parameter) and are cached under a hash of
the full request payload, so re-runs are free and byte-reproducible
(harness requirement, .plan 2.5). Works against OpenAI directly or any
OpenAI-compatible endpoint (OpenRouter) via base_url.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI


class CachedLLM:
    def __init__(
        self,
        model: str,
        base_url: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        cache_dir: str = ".cache/llm",
        extra: Optional[dict] = None,
        temperature: Optional[float] = 0.0,
    ) -> None:
        self.model = model
        self.extra = extra or {}
        # reasoning-tier models reject non-default temperature; pass None
        # to omit the parameter (determinism then rests on the cache)
        self.temperature = temperature
        self.cache_dir = Path(cache_dir)
        self._client = OpenAI(
            base_url=base_url, api_key=os.environ.get(api_key_env))
        self.calls = 0
        self.cache_hits = 0
        self._lock = threading.Lock()
        self._schema_unsupported = False

    def complete_json(self, system: str, user: str, schema_name: str,
                      schema: dict) -> dict:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "schema": schema,
            "temperature": self.temperature,
        }
        if self.extra:
            payload["extra"] = self.extra
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / f"{digest}.json"
        if path.exists():
            with self._lock:
                self.cache_hits += 1
            return json.loads(path.read_text())

        with self._lock:
            self.calls += 1
        data = self._request(payload, schema_name, schema)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=1))
        return data

    def _create(self, attempts: int = 3, **kwargs):
        """Provider hiccups (truncated bodies, transient 5xx) surface as
        parse or API errors; retry with backoff before giving up."""
        if self.temperature is not None:
            kwargs.setdefault("temperature", self.temperature)
        for i in range(attempts):
            try:
                return self._client.chat.completions.create(**kwargs)
            except Exception:
                if i == attempts - 1:
                    raise
                time.sleep(2.0 * (i + 1))

    def complete_text(self, system: str, user: str) -> str:
        payload = {
            "model": self.model, "kind": "text",
            "temperature": self.temperature,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
        if self.extra:
            payload["extra"] = self.extra
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / f"{digest}.json"
        if path.exists():
            with self._lock:
                self.cache_hits += 1
            return json.loads(path.read_text())["text"]
        with self._lock:
            self.calls += 1
        resp = self._create(
            model=self.model, messages=payload["messages"], **self.extra)
        text = resp.choices[0].message.content or ""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"text": text}))
        return text

    @staticmethod
    def _extract_json(text: str) -> dict:
        """Parse model output that should be one JSON object but may come
        fenced, or as several concatenated objects (observed from
        DeepSeek via OpenRouter on long inputs): merge top-level dicts in
        order of appearance."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
            cleaned = cleaned.rsplit("```", 1)[0]
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            pass
        decoder = json.JSONDecoder()
        merged: dict = {}
        idx, found = 0, False
        while idx < len(cleaned):
            brace = cleaned.find("{", idx)
            if brace < 0:
                break
            try:
                obj, end = decoder.raw_decode(cleaned, brace)
            except json.JSONDecodeError:
                idx = brace + 1
                continue
            if isinstance(obj, dict):
                merged.update(obj)
                found = True
            idx = end
        if not found:
            raise ValueError("no JSON object in model output")
        return merged

    def _request(self, payload: dict, schema_name: str, schema: dict) -> dict:
        if not self._schema_unsupported:
            try:
                resp = self._create(
                    attempts=1,
                    model=self.model,
                    messages=payload["messages"],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": schema_name,
                                        "schema": schema, "strict": True},
                    },
                    **self.extra,
                )
                return self._extract_json(resp.choices[0].message.content)
            except (json.JSONDecodeError, ValueError):
                pass  # malformed output: retry below in json_object mode
            except Exception as exc:
                # fall through only for schema-support errors; anything
                # else (auth, params, transport) must surface as-is
                if not any(k in str(exc) for k in
                           ("json_schema", "response_format", "schema")):
                    raise
                self._schema_unsupported = True
        messages = [dict(m) for m in payload["messages"]]
        messages[0]["content"] += (
            "\nRespond with a single JSON object matching this schema "
            "exactly:\n" + json.dumps(schema))
        last: Exception = ValueError("unreachable")
        for attempt in range(3):
            resp = self._create(
                model=self.model,
                messages=messages,
                response_format={"type": "json_object"},
                **self.extra,
            )
            try:
                return self._extract_json(resp.choices[0].message.content)
            except (json.JSONDecodeError, ValueError) as exc:
                last = exc
                time.sleep(1.0 * (attempt + 1))
        raise last
