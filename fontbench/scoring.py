"""Scoring and analysis for FontBench results."""
from collections import defaultdict


def score_mc_results(results):
    correct = sum(1 for r in results if r["parsed_answer"] == r["answer"])
    total = len(results)

    # Per-property accuracy
    per_property = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        prop = r["property"]
        per_property[prop]["total"] += 1
        if r["parsed_answer"] == r["answer"]:
            per_property[prop]["correct"] += 1

    per_property_acc = {
        k: v["correct"] / v["total"] if v["total"] > 0 else 0.0
        for k, v in per_property.items()
    }

    # By difficulty
    by_difficulty = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        if "difficulty" in r:
            d = r["difficulty"]
            by_difficulty[d]["total"] += 1
            if r["parsed_answer"] == r["answer"]:
                by_difficulty[d]["correct"] += 1

    by_difficulty_acc = {
        k: v["correct"] / v["total"] if v["total"] > 0 else 0.0
        for k, v in by_difficulty.items()
    }

    # By script
    by_script = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        if "script" in r:
            s = r["script"]
            by_script[s]["total"] += 1
            if r["parsed_answer"] == r["answer"]:
                by_script[s]["correct"] += 1

    by_script_acc = {
        k: v["correct"] / v["total"] if v["total"] > 0 else 0.0
        for k, v in by_script.items()
    }

    # By source
    by_source = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        if "source" in r:
            s = r["source"]
            by_source[s]["total"] += 1
            if r["parsed_answer"] == r["answer"]:
                by_source[s]["correct"] += 1

    by_source_acc = {
        k: v["correct"] / v["total"] if v["total"] > 0 else 0.0
        for k, v in by_source.items()
    }

    return {
        "overall_accuracy": correct / total if total > 0 else 0.0,
        "total": total,
        "correct": correct,
        "per_property": dict(per_property_acc),
        "by_difficulty": dict(by_difficulty_acc),
        "by_script": dict(by_script_acc),
        "by_source": dict(by_source_acc),
    }


def score_open_ended_results(results):
    """Score open-ended results using exact match per property.

    Each result should have:
      - ground_truth: dict with font_family, font_size, font_style, font_color
      - extracted: dict with same keys (extracted from model response)
    """
    properties = ["font_family", "font_size", "font_style", "font_color"]
    per_property = {p: {"correct": 0, "total": 0} for p in properties}

    for r in results:
        gt = r.get("ground_truth", {})
        ex = r.get("extracted", {})
        for p in properties:
            if p in gt:
                per_property[p]["total"] += 1
                gt_val = str(gt[p]).lower().strip()
                ex_val = str(ex.get(p, "")).lower().strip()
                if gt_val == ex_val or gt_val in ex_val or ex_val in gt_val:
                    per_property[p]["correct"] += 1

    per_property_f1 = {}
    for p in properties:
        t = per_property[p]["total"]
        c = per_property[p]["correct"]
        per_property_f1[p] = c / t if t > 0 else 0.0

    overall = sum(per_property_f1.values()) / len(per_property_f1) if per_property_f1 else 0.0

    return {
        "overall_score": overall,
        "per_property_f1": per_property_f1,
    }
