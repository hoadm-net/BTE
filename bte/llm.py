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
    ) -> None:
        self.model = model
        self.cache_dir = Path(cache_dir)
        self._client = OpenAI(
            base_url=base_url, api_key=os.environ.get(api_key_env))
        self.calls = 0
        self.cache_hits = 0

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
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True).encode()).hexdigest()
        path = self.cache_dir / f"{digest}.json"
        if path.exists():
            self.cache_hits += 1
            return json.loads(path.read_text())

        self.calls += 1
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=payload["messages"],
            temperature=0,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name, "schema": schema, "strict": True},
            },
        )
        data = json.loads(resp.choices[0].message.content)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=1))
        return data
