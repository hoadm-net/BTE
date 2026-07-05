"""OpenAI-compatible chat client with a content-addressed disk cache.

Every request is temperature-0 and cached under a hash of the full
request payload, so re-runs are free and byte-reproducible (harness
requirement, .plan 2.5). Works against OpenAI directly or any
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
    ) -> None:
        self.model = model
        self.extra = extra or {}
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
            model=self.model, messages=payload["messages"],
            temperature=0, **self.extra)
        text = resp.choices[0].message.content or ""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"text": text}))
        return text

    def _request(self, payload: dict, schema_name: str, schema: dict) -> dict:
        if not self._schema_unsupported:
            try:
                resp = self._create(
                    attempts=1,
                    model=self.model,
                    messages=payload["messages"],
                    temperature=0,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": schema_name,
                                        "schema": schema, "strict": True},
                    },
                    **self.extra,
                )
                return json.loads(resp.choices[0].message.content)
            except Exception:
                # providers without strict schema support fall through to
                # json_object mode with the schema stated in the prompt
                self._schema_unsupported = True
        messages = [dict(m) for m in payload["messages"]]
        messages[0]["content"] += (
            "\nRespond with a single JSON object matching this schema "
            "exactly:\n" + json.dumps(schema))
        resp = self._create(
            model=self.model,
            messages=messages,
            temperature=0,
            response_format={"type": "json_object"},
            **self.extra,
        )
        return json.loads(resp.choices[0].message.content)
