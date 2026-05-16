#!/usr/bin/env python3
"""
Compare inter-judge agreement for reward hacking detection.
Analyzes agreement between two LLM judges on reward hacking classifications.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
import logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_judge_results(filepath: Path) -> Dict[str, Any]:
    """Load and validate judge results from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    if "detailed_results" not in data:
        raise ValueError(f"Missing 'detailed_results' in {filepath}")

    return data


def align_judgments(judge1_data: Dict, judge2_data: Dict) -> Tuple[List[Tuple], List[str], List[str]]:
    """
    Align judgments from two judges by agent_name and problem_id.

    Returns:
        - List of (judge1_result, judge2_result) tuples for matched problems
        - List of problem keys only in judge1
        - List of problem keys only in judge2
    """
    # Build lookup dictionaries
    judge1_by_key = {}
    for result in judge1_data["detailed_results"]:
        if result.get("success"):
            key = f"{result['agent_name']}/{result['problem_id']}"
            judge1_by_key[key] = result

    judge2_by_key = {}
    for result in judge2_data["detailed_results"]:
        if result.get("success"):
            key = f"{result['agent_name']}/{result['problem_id']}"
            judge2_by_key[key] = result

    # Find matches and mismatches
    all_keys = set(judge1_by_key.keys()) | set(judge2_by_key.keys())
    matched = []
    only_judge1 = []
    only_judge2 = []

    for key in sorted(all_keys):
        if key in judge1_by_key and key in judge2_by_key:
            matched.append((judge1_by_key[key], judge2_by_key[key]))
        elif key in judge1_by_key:
            only_judge1.append(key)
        else:
            only_judge2.append(key)

    return matched, only_judge1, only_judge2


def calculate_cohens_kappa(agreements: int, disagreements: int,
                           judge1_positives: int, judge2_positives: int,
                           total: int) -> float:
    """
    Calculate Cohen's Kappa coefficient.

    Kappa accounts for agreement occurring by chance.
    Kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
    """
    if total == 0:
        return 0.0

    observed_agreement = agreements / total

    # Calculate expected agreement by chance
    judge1_pos_rate = judge1_positives / total
    judge2_pos_rate = judge2_positives / total

    expected_positive_agreement = judge1_pos_rate * judge2_pos_rate
    expected_negative_agreement = (1 - judge1_pos_rate) * (1 - judge2_pos_rate)
    expected_agreement = expected_positive_agreement + expected_negative_agreement

    if expected_agreement >= 1.0:
        return 1.0

    kappa = (observed_agreement - expected_agreement) / (1 - expected_agreement)
    return kappa


def analyze_agreement(matched_pairs: List[Tuple]) -> Dict[str, Any]:
    """Analyze agreement between two judges."""
    total = len(matched_pairs)

    # Track agreements and disagreements for reward hacking
    rh_agreements = 0
    rh_disagreements = []
    judge1_rh_positives = 0
    judge2_rh_positives = 0

    # Track for heuristics
    heuristic_agreements = 0
    heuristic_disagreements = []
    judge1_heuristic_positives = 0
    judge2_heuristic_positives = 0

    # Track confidence correlation
    confidence_pairs = []

    # Track type agreements
    type_agreements = 0
    type_disagreements = []

    for j1, j2 in matched_pairs:
        analysis1 = j1.get("analysis", {})
        analysis2 = j2.get("analysis", {})

        # Reward hacking agreement
        is_rh1 = analysis1.get("is_reward_hacking", False)
        is_rh2 = analysis2.get("is_reward_hacking", False)

        if is_rh1:
            judge1_rh_positives += 1
        if is_rh2:
            judge2_rh_positives += 1

        if is_rh1 == is_rh2:
            rh_agreements += 1
        else:
            rh_disagreements.append({
                "agent_name": j1["agent_name"],
                "problem_id": j1["problem_id"],
                "judge1": analysis1,
                "judge2": analysis2
            })

        # Heuristic agreement
        is_heur1 = analysis1.get("is_heuristic", False)
        is_heur2 = analysis2.get("is_heuristic", False)

        if is_heur1:
            judge1_heuristic_positives += 1
        if is_heur2:
            judge2_heuristic_positives += 1

        if is_heur1 == is_heur2:
            heuristic_agreements += 1
        else:
            heuristic_disagreements.append({
                "agent_name": j1["agent_name"],
                "problem_id": j1["problem_id"],
                "judge1": analysis1,
                "judge2": analysis2
            })

        # Type agreement (only if both detected reward hacking or heuristic)
        if (is_rh1 or is_heur1) and (is_rh2 or is_heur2):
            type1 = analysis1.get("type", "unknown")
            type2 = analysis2.get("type", "unknown")

            if type1 == type2:
                type_agreements += 1
            else:
                type_disagreements.append({
                    "agent_name": j1["agent_name"],
                    "problem_id": j1["problem_id"],
                    "judge1_type": type1,
                    "judge2_type": type2
                })

        # Confidence correlation
        conf1 = analysis1.get("confidence", 0)
        conf2 = analysis2.get("confidence", 0)
        confidence_pairs.append((conf1, conf2))

    # Calculate Cohen's Kappa for reward hacking
    rh_kappa = calculate_cohens_kappa(
        rh_agreements,
        len(rh_disagreements),
        judge1_rh_positives,
        judge2_rh_positives,
        total
    )

    # Calculate Cohen's Kappa for heuristics
    heuristic_kappa = calculate_cohens_kappa(
        heuristic_agreements,
        len(heuristic_disagreements),
        judge1_heuristic_positives,
        judge2_heuristic_positives,
        total
    )

    # Calculate Pearson correlation for confidence scores
    if confidence_pairs:
        mean_conf1 = sum(c[0] for c in confidence_pairs) / len(confidence_pairs)
        mean_conf2 = sum(c[1] for c in confidence_pairs) / len(confidence_pairs)

        numerator = sum((c1 - mean_conf1) * (c2 - mean_conf2) for c1, c2 in confidence_pairs)
        denom1 = sum((c1 - mean_conf1) ** 2 for c1, c2 in confidence_pairs) ** 0.5
        denom2 = sum((c2 - mean_conf2) ** 2 for c1, c2 in confidence_pairs) ** 0.5

        if denom1 > 0 and denom2 > 0:
            confidence_correlation = numerator / (denom1 * denom2)
        else:
            confidence_correlation = 0.0
    else:
        confidence_correlation = 0.0

    return {
        "total_compared": total,
        "reward_hacking": {
            "agreements": rh_agreements,
            "disagreements": len(rh_disagreements),
            "agreement_rate": rh_agreements / total if total > 0 else 0,
            "cohens_kappa": rh_kappa,
            "judge1_positive_rate": judge1_rh_positives / total if total > 0 else 0,
            "judge2_positive_rate": judge2_rh_positives / total if total > 0 else 0,
            "disagreement_details": rh_disagreements
        },
        "heuristic": {
            "agreements": heuristic_agreements,
            "disagreements": len(heuristic_disagreements),
            "agreement_rate": heuristic_agreements / total if total > 0 else 0,
            "cohens_kappa": heuristic_kappa,
            "judge1_positive_rate": judge1_heuristic_positives / total if total > 0 else 0,
            "judge2_positive_rate": judge2_heuristic_positives / total if total > 0 else 0,
            "disagreement_details": heuristic_disagreements
        },
        "type_agreement": {
            "total_detections": type_agreements + len(type_disagreements),
            "agreements": type_agreements,
            "agreement_rate": type_agreements / (type_agreements + len(type_disagreements)) if (type_agreements + len(type_disagreements)) > 0 else 0,
            "disagreement_details": type_disagreements
        },
        "confidence_correlation": confidence_correlation
    }


def print_summary(analysis: Dict, only_judge1: List[str], only_judge2: List[str]):
    """Print human-readable summary of agreement analysis."""
    print("\n" + "="*70)
    print("INTER-JUDGE AGREEMENT ANALYSIS")
    print("="*70)

    print(f"\nTotal problems compared: {analysis['total_compared']}")

    if only_judge1:
        print(f"Problems only in Judge 1: {len(only_judge1)}")
    if only_judge2:
        print(f"Problems only in Judge 2: {len(only_judge2)}")

    # Reward hacking agreement
    rh = analysis["reward_hacking"]
    print("\n" + "-"*70)
    print("REWARD HACKING CLASSIFICATION")
    print("-"*70)
    print(f"Agreement rate: {rh['agreement_rate']:.1%} ({rh['agreements']}/{analysis['total_compared']})")
    print(f"Cohen's Kappa: {rh['cohens_kappa']:.3f}")
    print(f"Judge 1 positive rate: {rh['judge1_positive_rate']:.1%}")
    print(f"Judge 2 positive rate: {rh['judge2_positive_rate']:.1%}")
    print(f"Disagreements: {rh['disagreements']}")

    # Heuristic agreement
    heur = analysis["heuristic"]
    print("\n" + "-"*70)
    print("HEURISTIC CLASSIFICATION")
    print("-"*70)
    print(f"Agreement rate: {heur['agreement_rate']:.1%} ({heur['agreements']}/{analysis['total_compared']})")
    print(f"Cohen's Kappa: {heur['cohens_kappa']:.3f}")
    print(f"Judge 1 positive rate: {heur['judge1_positive_rate']:.1%}")
    print(f"Judge 2 positive rate: {heur['judge2_positive_rate']:.1%}")
    print(f"Disagreements: {heur['disagreements']}")

    # Type agreement
    type_agr = analysis["type_agreement"]
    print("\n" + "-"*70)
    print("TYPE CLASSIFICATION (when at least one judge detected an issue)")
    print("-"*70)
    print(f"Agreement rate: {type_agr['agreement_rate']:.1%} ({type_agr['agreements']}/{type_agr['total_detections']})")

    # Confidence correlation
    print("\n" + "-"*70)
    print("CONFIDENCE CORRELATION")
    print("-"*70)
    print(f"Pearson correlation: {analysis['confidence_correlation']:.3f}")


def print_disagreements(analysis: Dict, show_all: bool = False):
    """Print detailed disagreement information."""
    rh_disagreements = analysis["reward_hacking"]["disagreement_details"]

    if not rh_disagreements:
        print("\n" + "="*70)
        print("NO REWARD HACKING DISAGREEMENTS FOUND")
        print("="*70)
        return

    print("\n" + "="*70)
    print("REWARD HACKING DISAGREEMENTS")
    print("="*70)

    for i, disagreement in enumerate(rh_disagreements, 1):
        agent = disagreement["agent_name"]
        problem = disagreement["problem_id"]
        j1 = disagreement["judge1"]
        j2 = disagreement["judge2"]

        print(f"\n[{i}] {agent}/{problem}")
        print("-" * 70)

        print("\nJudge 1:")
        print(f"  Reward Hacking: {j1.get('is_reward_hacking', False)}")
        print(f"  Heuristic: {j1.get('is_heuristic', False)}")
        print(f"  Type: {j1.get('type', 'unknown')}")
        print(f"  Confidence: {j1.get('confidence', 0):.2f}")
        print(f"  Evidence: {j1.get('evidence', 'N/A')}")
        print(f"  Explanation: {j1.get('explanation', 'N/A')}")

        print("\nJudge 2:")
        print(f"  Reward Hacking: {j2.get('is_reward_hacking', False)}")
        print(f"  Heuristic: {j2.get('is_heuristic', False)}")
        print(f"  Type: {j2.get('type', 'unknown')}")
        print(f"  Confidence: {j2.get('confidence', 0):.2f}")
        print(f"  Evidence: {j2.get('evidence', 'N/A')}")
        print(f"  Explanation: {j2.get('explanation', 'N/A')}")

    if show_all:
        heur_disagreements = analysis["heuristic"]["disagreement_details"]

        if heur_disagreements:
            print("\n" + "="*70)
            print("HEURISTIC DISAGREEMENTS")
            print("="*70)

            for i, disagreement in enumerate(heur_disagreements, 1):
                agent = disagreement["agent_name"]
                problem = disagreement["problem_id"]
                j1 = disagreement["judge1"]
                j2 = disagreement["judge2"]

                print(f"\n[{i}] {agent}/{problem}")
                print("-" * 70)

                print("\nJudge 1:")
                print(f"  Heuristic: {j1.get('is_heuristic', False)}")
                print(f"  Type: {j1.get('type', 'unknown')}")
                print(f"  Confidence: {j1.get('confidence', 0):.2f}")

                print("\nJudge 2:")
                print(f"  Heuristic: {j2.get('is_heuristic', False)}")
                print(f"  Type: {j2.get('type', 'unknown')}")
                print(f"  Confidence: {j2.get('confidence', 0):.2f}")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare inter-judge agreement for reward hacking detection"
    )
    parser.add_argument("judge1", help="Path to first judge's JSON output")
    parser.add_argument("judge2", help="Path to second judge's JSON output")
    parser.add_argument("--output", help="Output file for detailed results (JSON)")
    parser.add_argument("--show-all-disagreements", action="store_true",
                       help="Show all disagreements including heuristic classification")

    args = parser.parse_args()

    # Load judge results
    logger.info(f"Loading Judge 1: {args.judge1}")
    judge1_data = load_judge_results(Path(args.judge1))

    logger.info(f"Loading Judge 2: {args.judge2}")
    judge2_data = load_judge_results(Path(args.judge2))

    # Align judgments
    logger.info("Aligning judgments...")
    matched_pairs, only_judge1, only_judge2 = align_judgments(judge1_data, judge2_data)

    logger.info(f"Found {len(matched_pairs)} matched problems")
    if only_judge1:
        logger.warning(f"{len(only_judge1)} problems only in Judge 1")
    if only_judge2:
        logger.warning(f"{len(only_judge2)} problems only in Judge 2")

    # Analyze agreement
    logger.info("Analyzing agreement...")
    analysis = analyze_agreement(matched_pairs)

    # Print summary
    print_summary(analysis, only_judge1, only_judge2)

    # Print disagreements
    print_disagreements(analysis, args.show_all_disagreements)

    # Save detailed results if requested
    if args.output:
        output_data = {
            "analysis": analysis,
            "only_judge1": only_judge1,
            "only_judge2": only_judge2,
            "metadata": {
                "judge1_file": args.judge1,
                "judge2_file": args.judge2,
                "total_matched": len(matched_pairs)
            }
        }

        output_path = Path(args.output)
        with open(output_path, 'w') as f:
            json.dump(output_data, f, indent=2)

        logger.info(f"Detailed results saved to: {output_path}")


if __name__ == "__main__":
    main()