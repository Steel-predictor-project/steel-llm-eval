"""Run the steel-property benchmark for a single model.

Examples:
    # A real model via OpenRouter (needs OPENROUTER_API_KEY):
    python harness/run_eval.py --model openai/gpt-4o-mini

    # Offline sanity check, no API key needed:
    python harness/run_eval.py --provider mock
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prompts  # noqa: E402
from openrouter_client import chat  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "data" / "benchmark.csv"
RESULTS = ROOT / "results"

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)


def parse_scores(text: str):
    """Extract edge_retention and toughness from a model reply."""
    if not text:
        return None, None
    m = _JSON_RE.search(text)
    blob = m.group(0) if m else text
    try:
        obj = json.loads(blob)
    except json.JSONDecodeError:
        return None, None

    def clamp(v):
        try:
            return max(1.0, min(10.0, float(v)))
        except (TypeError, ValueError):
            return None

    return clamp(obj.get("edge_retention")), clamp(obj.get("toughness"))


def mock_predict(row):
    """Deterministic composition heuristic used when no API key is present.

    Not a serious model — a transparent floor so the harness is runnable
    offline and its output format can be inspected.
    """
    c = float(row.get("C", 0) or 0)
    v = float(row.get("V", 0) or 0)
    cr = float(row.get("Cr", 0) or 0)
    edge = 2.0 + 1.5 * c + 0.5 * v + 0.05 * cr
    tough = 7.5 - 1.2 * c - 0.3 * v
    clip = lambda x: max(1.0, min(10.0, x))  # noqa: E731
    return round(clip(edge), 1), round(clip(tough), 1)


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", help="OpenRouter model id, e.g. openai/gpt-4o-mini")
    ap.add_argument("--provider", choices=["openrouter", "mock"], default="openrouter")
    ap.add_argument("--limit", type=int, default=0, help="limit steels (debug)")
    ap.add_argument("--temperature", type=float, default=0.0)
    args = ap.parse_args()

    df = pd.read_csv(BENCHMARK)
    if args.limit:
        df = df.head(args.limit)

    is_mock = args.provider == "mock" or not args.model
    label = "mock-heuristic" if is_mock else args.model
    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"raw_{slug(label)}.csv"

    rows = []
    for i, row in df.iterrows():
        if is_mock:
            edge, tough = mock_predict(row)
            raw = "mock"
        else:
            raw = chat(args.model, prompts.SYSTEM_PROMPT,
                       prompts.build_user_prompt(row), temperature=args.temperature)
            edge, tough = parse_scores(raw)
        rows.append({
            "steel_name": row["steel_name"],
            "model": label,
            "edge_retention": edge,
            "toughness": tough,
            "raw_response": (raw or "").replace("\n", " ")[:500],
        })
        print(f"  [{i + 1}/{len(df)}] {row['steel_name']:12s} "
              f"edge={edge} tough={tough}")

    pd.DataFrame(rows).to_csv(out, index=False)
    ok = sum(1 for r in rows if r["edge_retention"] is not None)
    print(f"\nWrote {out}  ({ok}/{len(rows)} parsed successfully)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
