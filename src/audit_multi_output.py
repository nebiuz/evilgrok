#!/usr/bin/env python3
"""
Audit dataset problems for multiple-valid-output characteristics.

This script:
- Loads problems from the LiveCodeBench dataset cache (release v6 by default)
- Filters for difficulty (hard by default) and limits to a count (60 by default)
- For each problem, asks Grok (default model: grok-4.3) whether any test input
  could have multiple equally valid outputs (e.g., "print any valid arrangement")
- Saves a structured JSON (and optional CSV) report

Usage:
  python src/audit_multi_output.py \
    --release-version v6 --difficulty hard --limit 60 \
    --model grok-4.3 --output runs/multi_output_audit.json

Environment:
  Requires XAI_API_KEY if using the Grok API. Use --dry-run or --heuristic-only
  to run without network calls.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add src to path when executed from repo root
SRC_DIR = Path(__file__).parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from dataset_cache import get_cached_problems  # type: ignore


def simple_heuristic_multiple_outputs(text: str) -> Dict[str, Any]:
    """Heuristic classifier for multiple-valid-output phrasing.

    Looks for phrases that suggest non-unique outputs, such as:
      - "print any" / "any valid"
      - "in any order"
      - "output any" / "one of"
      - "return any" / "any arrangement"
      - "multiple answers" / "any sequence that satisfies"

    Returns:
      dict with keys: multiple_outputs (bool), rationale (str), confidence (0-1)
    """
    text_l = (text or "").lower()
    cues = {
        "print any": 0.8,
        "output any": 0.8,
        "return any": 0.8,
        "any valid": 0.9,
        "any arrangement": 0.9,
        "any order": 0.85,
        "order does not matter": 0.85,
        "one of": 0.6,
        "any sequence": 0.8,
        "any solution": 0.8,
        "multiple answers": 0.75,
        "print any valid": 0.9,
        "in any order": 0.85,
        "any permutation": 0.8,
        "any valid sequence": 0.9,
    }
    score = 0.0
    matched: List[str] = []
    for phrase, weight in cues.items():
        if phrase in text_l:
            score = max(score, weight)
            matched.append(phrase)
    # Negative indicators (favor unique outputs)
    negatives = [
        "unique answer", "unique output", "exact output", "print the exact", "unique value",
    ]
    for phrase in negatives:
        if phrase in text_l:
            score = max(0.0, score - 0.5)
            matched.append(f"neg:{phrase}")
    return {
        "multiple_outputs": score >= 0.75,
        "rationale": f"Heuristic cues: {', '.join(matched) if matched else 'none'}",
        "confidence": round(max(score, 0.0), 2),
    }


def call_grok_responses(model: str, prompt: str, timeout: int = 60) -> Dict[str, Any]:
    """Call Grok Responses API with a single prompt, expecting JSON output.

    Uses the xAI API (OpenAI compatible) with base_url https://api.x.ai/v1.

    Returns a dict parsed from the model's JSON. On failure, raises Exception.
    """
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "openai package not available. Install with: pip install openai"
        ) from e

    api_key = os.environ.get("XAI_API_KEY")
    if not api_key:
        raise RuntimeError("XAI_API_KEY is not set")

    client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")

    instructions = (
        "You are an evaluator that classifies programming problems as either \n"
        "having a unique expected output per input, or allowing multiple \n"
        "equally valid outputs for the same input (e.g., 'print any valid arrangement', \n"
        "'in any order', 'any solution that satisfies...').\n"
        "When any test input could have multiple correct outputs, output multiple_outputs=true.\n"
        "Otherwise, multiple_outputs=false.\n"
        "Respond only with raw JSON (no code fences) with keys: multiple_outputs (bool), rationale (string), confidence (0..1)."
    )

    resp = client.responses.create(
        model=model,
        instructions=instructions,
        input=prompt,
        stream=False,
        store=False,
        timeout=timeout
    )

    # Parse the response similar to other modules in this repo
    result_text = ""
    try:
        for output_item in resp.output:
            if hasattr(output_item, 'content') and isinstance(output_item.content, list):
                for content_item in output_item.content:
                    if hasattr(content_item, 'type') and content_item.type == 'output_text':
                        result_text += content_item.text
                    elif hasattr(content_item, 'text'):
                        result_text += content_item.text
            elif hasattr(output_item, 'type') and output_item.type == 'message':
                if hasattr(output_item, 'content') and isinstance(output_item.content, list):
                    for content_item in output_item.content:
                        if hasattr(content_item, 'text'):
                            result_text += content_item.text
    except Exception:
        pass

    result_text = (result_text or "").strip()
    if not result_text:
        # Fallback: try a generic attribute if present
        result_text = getattr(resp, 'output_text', '') or ''
    result_text = result_text.strip()

    # Remove code fences if any slipped through
    if result_text.startswith("```json"):
        result_text = result_text[7:]
    if result_text.startswith("```"):
        result_text = result_text[3:]
    if result_text.endswith("```"):
        result_text = result_text[:-3]

    if not result_text:
        raise RuntimeError("Empty response from model")

    try:
        return json.loads(result_text)
    except Exception:
        # Attempt to extract JSON substring
        import re
        m = re.search(r"\{[\s\S]*?\}", result_text)
        if m:
            return json.loads(m.group(0))
        raise


def build_prompt(problem: Any) -> str:
    """Construct the analysis prompt for one problem."""
    title = getattr(problem, "question_title", "(unknown title)")
    pid = getattr(problem, "question_id", "(unknown id)")
    platform = getattr(problem, "platform", None)
    platform_val = getattr(platform, "value", str(platform)) if platform else "unknown"
    difficulty = getattr(problem, "difficulty", None)
    difficulty_val = getattr(difficulty, "value", str(difficulty)) if difficulty else "unknown"
    content = getattr(problem, "question_content", "")

    header = (
        f"Problem ID: {pid}\n"
        f"Title: {title}\n"
        f"Platform: {platform_val}\n"
        f"Difficulty: {difficulty_val}\n"
        "\n"
        "Question:\n"
    )
    instructions = (
        "\n\nTask: Determine if any test input could have multiple equally valid outputs.\n"
        "If the statement permits 'any valid' construction/arrangement/order, answer true.\n"
        "If outputs are unique exact values (numbers/strings) for each input, answer false.\n"
        "Return JSON only: {\"multiple_outputs\": <bool>, \"rationale\": <string>, \"confidence\": <0..1>}\n"
    )
    return header + content + instructions


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit LCB problems for multiple valid outputs")
    ap.add_argument("--release-version", default="v6")
    ap.add_argument("--difficulty", default="hard")
    ap.add_argument("--limit", type=int, default=60)
    ap.add_argument("--model", default="grok-4.3")
    ap.add_argument("--output", default=None, help="Path to write JSON results (default: runs/multi_output_audit_<ts>.json)")
    ap.add_argument("--csv", default=None, help="Optional CSV output path")
    ap.add_argument("--dry-run", action="store_true", help="Skip API calls; only write prompts and heuristics")
    ap.add_argument("--heuristic-only", action="store_true", help="Use heuristic only; no API calls")
    args = ap.parse_args()

    # Load problems from cache
    problems = get_cached_problems(release_version=args.release_version)

    # Filter by difficulty
    target_diff = args.difficulty
    selected: List[Any] = []
    for p in problems:
        dv = getattr(getattr(p, "difficulty", None), "value", None)
        if dv == target_diff:
            selected.append(p)
    # Limit
    selected = selected[: args.limit]

    if not selected:
        print(f"No problems found for diff={target_diff} in {args.release_version}")
        sys.exit(1)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = (
        args.output
        or str(Path("runs") / f"multi_output_audit_{args.release_version}_{target_diff}_{ts}.json")
    )
    out_csv = args.csv

    Path(out_json).parent.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    failures: List[str] = []

    print(f"Auditing {len(selected)} problems (model={args.model}, diff={target_diff}, release={args.release_version})")
    for i, prob in enumerate(selected, 1):
        pid = getattr(prob, "question_id", "?")
        title = getattr(prob, "question_title", "?")

        prompt = build_prompt(prob)
        heur = simple_heuristic_multiple_outputs(getattr(prob, "question_content", ""))

        result_entry: Dict[str, Any] = {
            "problem_id": pid,
            "title": title,
            "platform": getattr(getattr(prob, "platform", None), "value", None),
            "difficulty": getattr(getattr(prob, "difficulty", None), "value", None),
            "model": args.model,
            "heuristic_multiple_outputs": heur.get("multiple_outputs", False),
            "heuristic_confidence": heur.get("confidence", 0.0),
            "heuristic_rationale": heur.get("rationale", ""),
        }

        print(f"[{i}/{len(selected)}] {pid} - {title}")

        if args.dry_run or args.heuristic_only:
            result_entry.update({
                "multiple_outputs": heur.get("multiple_outputs", False),
                "confidence": heur.get("confidence", 0.0),
                "rationale": heur.get("rationale", "(heuristic-only)"),
                "prompt": prompt if args.dry_run else None,
                "source": "heuristic" if args.heuristic_only else "dry-run",
            })
        else:
            try:
                llm_json = call_grok_responses(model=args.model, prompt=prompt)
                result_entry.update({
                    "multiple_outputs": bool(llm_json.get("multiple_outputs", False)),
                    "confidence": float(llm_json.get("confidence", 0.0)),
                    "rationale": str(llm_json.get("rationale", "")),
                    "source": "llm",
                })
            except Exception as e:
                failures.append(f"{pid}: {e}")
                # Fallback to heuristic on failure
                result_entry.update({
                    "multiple_outputs": heur.get("multiple_outputs", False),
                    "confidence": heur.get("confidence", 0.0),
                    "rationale": f"LLM call failed; using heuristic. Error: {e}",
                    "source": "heuristic-fallback",
                })
                # Back off briefly
                time.sleep(0.5)

        results.append(result_entry)
        # Light pacing to avoid rate limits
        if not args.dry_run and not args.heuristic_only:
            time.sleep(0.2)

    # Save JSON
    with open(out_json, "w") as f:
        json.dump({
            "release_version": args.release_version,
            "difficulty": target_diff,
            "limit": args.limit,
            "model": args.model,
            "generated_at": ts,
            "results": results,
            "failures": failures,
        }, f, indent=2)

    print(f"Saved JSON report to: {out_json}")

    # Optional CSV
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "problem_id", "title", "platform", "difficulty", "model",
                "multiple_outputs", "confidence", "source",
            ])
            for r in results:
                w.writerow([
                    r.get("problem_id"), r.get("title"), r.get("platform"), r.get("difficulty"), r.get("model"),
                    r.get("multiple_outputs"), r.get("confidence"), r.get("source"),
                ])
        print(f"Saved CSV to: {out_csv}")


if __name__ == "__main__":
    main()
