"""Minimal OpenRouter chat client (no SDK dependency, just requests).

Reads the API key from the OPENROUTER_API_KEY environment variable. One key
provides access to models from OpenAI, Anthropic, Google, Meta and others,
which keeps the benchmark provider-agnostic.
"""

from __future__ import annotations

import json
import os
import time

import requests

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterError(RuntimeError):
    pass


def chat(model: str, system: str, user: str, *, temperature: float = 0.0,
         max_tokens: int = 1200, retries: int = 3, timeout: int = 120) -> str:
    """Send a chat completion and return the assistant text content."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise OpenRouterError(
            "OPENROUTER_API_KEY is not set. Export it, or use --provider mock."
        )
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/Steel-predictor-project/steel-llm-eval",
        "X-Title": "steel-llm-eval",
    }
    payload = {
        "model": model,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    }

    last_err = None
    for attempt in range(retries):
        try:
            resp = requests.post(OPENROUTER_URL, headers=headers,
                                 data=json.dumps(payload), timeout=timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise OpenRouterError(f"HTTP {resp.status_code}: {resp.text[:200]}")
            resp.raise_for_status()
            data = resp.json()
            return data["choices"][0]["message"]["content"]
        except Exception as e:  # noqa: BLE001 - retry on any transient failure
            last_err = e
            time.sleep(2 * (attempt + 1))
    raise OpenRouterError(f"OpenRouter call failed after {retries} tries: {last_err}")
