"""Prompt construction for the steel-property LLM benchmark.

The model is given only a steel's chemical composition (plus whether it is
powder-metallurgy and its test hardness when known) and must predict two
properties that we have *objective laboratory measurements* for:

  - edge_retention : how long the edge keeps cutting (ground truth: CATRA TCC, mm)
  - toughness      : resistance to chipping/breaking (ground truth: Charpy, ft-lbs)

Both are requested on a 1-10 scale so the task is identical for every model and
directly comparable to the reference ML model. We score with scale-free rank
correlation and pairwise accuracy, so the absolute calibration of the 1-10
scale does not advantage or penalize any model.
"""

from __future__ import annotations

COMPOSITION_ELEMENTS = ["C", "Cr", "V", "Mo", "W", "Co", "N", "Mn", "Si", "Nb", "Ni"]

SYSTEM_PROMPT = (
    "You are a metallurgist specializing in knife and tool steels. Given a "
    "steel's chemical composition you estimate its performance properties. "
    "Respond ONLY with a compact JSON object and nothing else."
)

INSTRUCTIONS = """Estimate the following two properties for this steel on a 1-10 scale
(1 = worst among common knife steels, 10 = best), based only on the chemistry below.

- edge_retention: how long the blade keeps cutting before dulling (wear resistance).
- toughness: resistance to chipping and breaking under impact.

Return ONLY this JSON (no prose, no code fence):
{"edge_retention": <number 1-10>, "toughness": <number 1-10>}"""


def build_composition_block(row) -> str:
    parts = []
    for el in COMPOSITION_ELEMENTS:
        val = row.get(el, 0)
        try:
            v = float(val)
        except (TypeError, ValueError):
            v = 0.0
        if v and v > 0:
            parts.append(f"{el}={v:g}%")
    comp = ", ".join(parts) if parts else "(no composition given)"

    pm = "yes" if float(row.get("powder_metallurgy", 0) or 0) >= 1 else "no"
    lines = [
        f"Steel: {row['steel_name']}",
        f"Composition (weight %): {comp}",
        f"Powder metallurgy: {pm}",
    ]
    hrc = row.get("catra_test_hrc")
    try:
        if hrc is not None and float(hrc) > 0:
            lines.append(f"Test hardness: {float(hrc):g} HRC")
    except (TypeError, ValueError):
        pass
    return "\n".join(lines)


def build_user_prompt(row) -> str:
    return build_composition_block(row) + "\n\n" + INSTRUCTIONS
