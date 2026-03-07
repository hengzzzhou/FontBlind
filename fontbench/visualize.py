"""Generate tables and charts from evaluation results."""
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
from fontbench.config import RESULTS_DIR


def load_summary(results_dir=RESULTS_DIR):
    with open(results_dir / "summary.json") as f:
        return json.load(f)


def make_leaderboard_table(summary):
    rows = []
    for name, data in summary.items():
        row = {"Model": data["model"], "Overall MC Acc": data.get("mc_accuracy", 0)}
        for prop, acc in data.get("mc_per_property", {}).items():
            row[prop] = acc
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("Overall MC Acc", ascending=False)
    return df


def plot_per_property(summary, output_path):
    models = []
    properties = ["font_family", "font_size", "font_style", "font_color"]
    data = {p: [] for p in properties}

    for name, s in summary.items():
        models.append(s["model"])
        for p in properties:
            data[p].append(s.get("mc_per_property", {}).get(p, 0))

    x = range(len(models))
    width = 0.2
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, p in enumerate(properties):
        offset = (i - 1.5) * width
        ax.bar([xi + offset for xi in x], data[p], width, label=p)

    ax.set_ylabel("Accuracy")
    ax.set_title("FontBench MC Accuracy by Property")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_by_script(summary, output_path):
    models = []
    scripts = set()
    for s in summary.values():
        for k in s.get("mc_by_script", {}):
            scripts.add(k)
    scripts = sorted(scripts)

    data = {s: [] for s in scripts}
    for name, s in summary.items():
        models.append(s["model"])
        for sc in scripts:
            data[sc].append(s.get("mc_by_script", {}).get(sc, 0))

    x = range(len(models))
    width = 0.25
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, sc in enumerate(scripts):
        offset = (i - len(scripts) / 2 + 0.5) * width
        ax.bar([xi + offset for xi in x], data[sc], width, label=sc)

    ax.set_ylabel("Accuracy")
    ax.set_title("FontBench MC Accuracy by Script")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_by_difficulty(summary, output_path):
    models = []
    difficulties = ["easy", "medium", "hard"]
    data = {d: [] for d in difficulties}

    for name, s in summary.items():
        models.append(s["model"])
        for d in difficulties:
            data[d].append(s.get("mc_by_difficulty", {}).get(d, 0))

    x = range(len(models))
    width = 0.25
    fig, ax = plt.subplots(figsize=(14, 6))

    for i, d in enumerate(difficulties):
        offset = (i - 1) * width
        ax.bar([xi + offset for xi in x], data[d], width, label=d)

    ax.set_ylabel("Accuracy")
    ax.set_title("FontBench MC Accuracy by Difficulty")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_robustness_curves(transform_results, output_dir):
    """Plot accuracy vs. transform severity, one plot per category."""
    for category, transforms in transform_results.items():
        if category == "resolution":
            continue  # handled by plot_resolution_ablation

        fig, ax = plt.subplots(figsize=(10, 6))

        # Collect models from first transform entry
        first_transform = next(iter(transforms.values()))
        models = list(first_transform.keys())

        transform_names = list(transforms.keys())

        for model in models:
            accuracies = []
            for t_name in transform_names:
                acc = transforms[t_name].get(model, {}).get("overall_accuracy", 0)
                accuracies.append(acc)
            ax.plot(range(len(transform_names)), accuracies, marker="o", label=model)

        ax.set_xlabel("Transform Severity")
        ax.set_ylabel("MC Accuracy")
        ax.set_title(f"Robustness: {category.replace('_', ' ').title()}")
        ax.set_xticks(range(len(transform_names)))
        ax.set_xticklabels([n.split("_")[-1] for n in transform_names], rotation=45, ha="right")
        ax.legend(fontsize=8)
        ax.set_ylim(0, 1)
        plt.tight_layout()
        plt.savefig(output_dir / f"robustness_{category}.png", dpi=150)
        plt.close()


def plot_resolution_ablation(transform_results, output_dir):
    """Plot accuracy vs. resolution scale."""
    if "resolution" not in transform_results:
        return

    transforms = transform_results["resolution"]
    # Sort by scale value for logical ordering (0.25x, 0.5x, 1.0x, 2.0x)
    transform_names = sorted(transforms.keys(), key=lambda n: float(n.split("_")[1].rstrip("x")))

    # Collect all models across all scales
    models = set()
    for t_data in transforms.values():
        models.update(t_data.keys())
    models = sorted(models)

    fig, ax = plt.subplots(figsize=(10, 6))
    for model in models:
        accuracies = []
        for t_name in transform_names:
            acc = transforms[t_name].get(model, {}).get("overall_accuracy", 0)
            accuracies.append(acc)
        ax.plot(range(len(transform_names)), accuracies, marker="s", label=model)

    ax.set_xlabel("Resolution Scale")
    ax.set_ylabel("MC Accuracy")
    ax.set_title("Resolution Ablation")
    ax.set_xticks(range(len(transform_names)))
    ax.set_xticklabels([n.split("_")[-1] for n in transform_names], rotation=45, ha="right")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_dir / "resolution_ablation.png", dpi=150)
    plt.close()


def plot_resolution_per_property(transform_results, output_dir):
    """2x2 grid showing per-property accuracy vs. resolution."""
    if "resolution" not in transform_results:
        return

    transforms = transform_results["resolution"]
    # Sort by scale value for logical ordering
    transform_names = sorted(transforms.keys(), key=lambda n: float(n.split("_")[1].rstrip("x")))

    # Collect all models across all scales
    models = set()
    for t_data in transforms.values():
        models.update(t_data.keys())
    models = sorted(models)

    properties = ["font_family", "font_size", "font_style", "font_color"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for idx, prop in enumerate(properties):
        ax = axes[idx // 2][idx % 2]
        for model in models:
            accuracies = []
            for t_name in transform_names:
                acc = transforms[t_name].get(model, {}).get("per_property", {}).get(prop, 0)
                accuracies.append(acc)
            ax.plot(range(len(transform_names)), accuracies, marker="s", label=model)

        ax.set_xlabel("Resolution Scale")
        ax.set_ylabel("Accuracy")
        ax.set_title(f"{prop.replace('_', ' ').title()}")
        ax.set_xticks(range(len(transform_names)))
        ax.set_xticklabels([n.split("_")[-1] for n in transform_names], rotation=45, ha="right")
        ax.legend(fontsize=6)
        ax.set_ylim(0, 1)

    plt.suptitle("Resolution Ablation — Per Property", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "resolution_per_property.png", dpi=150)
    plt.close()


def plot_frb_comparison(frb_results, summary, output_path):
    """Generate FontBench vs. FRB comparison chart."""
    models = []
    fontbench_family = []
    frb_overall = []
    frb_easy = []
    frb_hard = []

    for model_name, frb_data in frb_results.items():
        if model_name in summary:
            models.append(model_name)
            fontbench_family.append(
                summary[model_name].get("mc_per_property", {}).get("font_family", 0)
            )
            frb_overall.append(frb_data["scores"]["overall_accuracy"])
            frb_easy.append(frb_data["scores"]["easy_accuracy"])
            frb_hard.append(frb_data["scores"]["hard_accuracy"])

    x = range(len(models))
    width = 0.2
    fig, ax = plt.subplots(figsize=(14, 6))

    ax.bar([xi - 1.5 * width for xi in x], fontbench_family, width, label="FontBench Family")
    ax.bar([xi - 0.5 * width for xi in x], frb_overall, width, label="FRB Overall")
    ax.bar([xi + 0.5 * width for xi in x], frb_easy, width, label="FRB Easy")
    ax.bar([xi + 1.5 * width for xi in x], frb_hard, width, label="FRB Hard")

    ax.set_ylabel("Accuracy")
    ax.set_title("FontBench (Family) vs. FRB Cross-Benchmark Comparison")
    ax.set_xticks(list(x))
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def generate_all_plots(results_dir=RESULTS_DIR):
    summary = load_summary(results_dir)
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    plot_per_property(summary, plots_dir / "per_property.png")
    plot_by_script(summary, plots_dir / "by_script.png")
    plot_by_difficulty(summary, plots_dir / "by_difficulty.png")

    # Generate transform plots if results exist
    transform_results_path = results_dir / "transform_results.json"
    if transform_results_path.exists():
        with open(transform_results_path) as f:
            transform_results = json.load(f)
        plot_robustness_curves(transform_results, plots_dir)
        plot_resolution_ablation(transform_results, plots_dir)
        plot_resolution_per_property(transform_results, plots_dir)

    # Generate FRB comparison plot if results exist
    frb_results_path = results_dir / "frb_results.json"
    if frb_results_path.exists():
        with open(frb_results_path) as f:
            frb_results = json.load(f)
        plot_frb_comparison(frb_results, summary, plots_dir / "frb_comparison.png")

    table = make_leaderboard_table(summary)
    table.to_csv(results_dir / "leaderboard.csv", index=False)
    print(table.to_string(index=False))
    print(f"\nPlots saved to {plots_dir}")


if __name__ == "__main__":
    generate_all_plots()
