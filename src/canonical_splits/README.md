# Canonical Holdout Splits

This directory contains canonical holdout test case splits for benchmark reproducibility.

## Purpose

When running agent benchmarks, test cases are split into "visible" (shown to agents) and "holdout" (hidden for final evaluation). Without a fixed split, random selection makes results incomparable across runs.

Canonical splits ensure that:
- The same test cases are held out across different runs
- Results can be meaningfully compared between agents/configurations
- Benchmark results are reproducible

## Available Splits

| File | Description | Problems |
|------|-------------|----------|
| `v5v6_hard_154p.json` | Official split for v5_v6 hard problems | 154 |

## File Format

```json
{
  "metadata": {
    "source_run": "runs/claude_codex_v5v6_154p_hard",
    "release_version": "v5_v6",
    "difficulty": "hard",
    "problem_count": 154,
    "description": "Canonical holdout split extracted from ..."
  },
  "splits": {
    "problem_id": {
      "total_test_cases": 44,
      "holdout_count": 10,
      "holdout_indices": [5, 12, 18, 24, 31, 37, 41, 42, 43, 44]
    }
  }
}
```

## Usage

Canonical splits are used automatically when `use_canonical: true` (default) in the holdout config:

```yaml
holdout_test_cases:
  enabled: true
  use_canonical: true              # Default: true
  canonical_split: "v5v6_hard_154p"  # Default split file
```

To disable canonical splits and use random selection:

```yaml
holdout_test_cases:
  enabled: true
  use_canonical: false
```

## Creating New Splits

Use the extraction script to create a canonical split from an existing run:

```bash
python scripts/extract_canonical_split.py \
    --run-dir runs/your_run_name \
    --output src/canonical_splits/your_split_name.json \
    --release-version v6
```

The script:
1. Scans workspace directories for holdout test case files
2. Matches test cases against the dataset to find original indices
3. Saves indices to a compact JSON file
