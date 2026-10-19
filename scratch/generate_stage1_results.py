"""Compile Stage 1 tables, figures, and publication report from saved results."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = RESULTS / "final_figures"
METHOD_LABELS = {
    "A_raw_residual": "A: Raw error",
    "B_shap_direct_no_graph": "B: SHAP",
    "C_ppr_only_graph": "C: PageRank",
    "D_proposed_ppr_plus_shap": "D: PageRank + SHAP",
    "random_guess": "Random",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt(value: object) -> str:
    if pd.isna(value):
        return "N/A"
    return str(value).replace("±", "+/-")


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = [[fmt(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    widths = [max(len(columns[i]), *(len(row[i]) for row in rows)) for i in range(len(columns))]
    head = "| " + " | ".join(columns[i].ljust(widths[i]) for i in range(len(columns))) + " |"
    rule = "| " + " | ".join("-" * widths[i] for i in range(len(columns))) + " |"
    body = ["| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(columns))) + " |" for row in rows]
    return "\n".join([head, rule, *body])


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(FIGURES / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def figure_pipeline() -> None:
    fig, ax = plt.subplots(figsize=(13.2, 6.2))
    ax.set_xlim(0, 13.2)
    ax.set_ylim(0, 6.2)
    ax.axis("off")
    colors = {"data": "#DCECEB", "model": "#E9E4D4", "eval": "#F5DFD8", "out": "#DDE4F0"}
    boxes = [
        (0.35, 3.45, 2.3, 1.55, "Chronological data", "Dataset03: 80% train / 20% calibration\nDataset04 + test: held-out evaluation", "data"),
        (3.05, 3.45, 2.3, 1.55, "Harmonize + align", "Round Dataset03 sensors to 2 decimals\nH=12 scored target at t + 23", "data"),
        (5.75, 3.45, 2.3, 1.55, "Detection experiments", "5 seeds x C-Town / Identity\n12-step SpatioTemporalGNN", "model"),
        (8.45, 3.45, 2.3, 1.55, "Fair localization", "14 attacks; 22 assets\n50 SHAP permutations; 4 methods", "eval"),
        (10.8, 0.75, 2.05, 1.55, "Stage 1 outputs", "Tables R1-R3\nFigures F1-F3\nResults and discussion", "out"),
    ]
    for x, y, w, h, title, detail, kind in boxes:
        patch = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
                               linewidth=1.4, edgecolor="#35424A", facecolor=colors[kind])
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h - 0.36, title, ha="center", va="center", fontsize=12, weight="bold", color="#17262D")
        ax.text(x + w / 2, y + 0.55, detail, ha="center", va="center", fontsize=9.5, linespacing=1.5, color="#27363D")
    for start, end in [((2.68, 4.22), (3.0, 4.22)), ((5.38, 4.22), (5.7, 4.22)),
                       ((8.08, 4.22), (8.4, 4.22)), ((10.2, 3.4), (11.35, 2.38))]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=16, linewidth=1.7, color="#35424A"))
    ax.text(6.6, 5.7, "Stage 1 experimental pipeline", ha="center", fontsize=19, weight="bold", color="#17262D")
    ax.text(6.6, 0.95, "Detection is scored on Dataset04 intervals and Test Dataset ground-truth intervals; localization compares oracle and detector timing.",
            ha="center", fontsize=9.4, color="#46545A")
    save_figure(fig, "F1_stage1_pipeline")


def figure_test_timeline() -> None:
    score_path = RESULTS / "test_inference" / "corrected" / "test_anomaly_scores.csv"
    scores = pd.read_csv(score_path)
    scores["timestamp"] = pd.to_datetime(scores["timestamp"])
    attacks = pd.read_csv(ROOT / "data" / "ground_truth_attacks.csv")
    attacks = attacks[attacks["dataset"] == "Test Dataset"]
    threshold_report = load_json(RESULTS / "test_inference" / "corrected" / "test_inference_report.json")
    threshold = float(threshold_report["operational_threshold"])

    fig, ax = plt.subplots(figsize=(13.2, 5.2))
    ax.plot(scores["timestamp"], scores["anomaly_score"], color="#246A73", linewidth=0.8, alpha=0.9,
            label="Saved H=12 test anomaly score")
    ax.axhline(threshold, color="#B4453E", linewidth=1.6, linestyle="--", label=f"Operational threshold ({threshold:.3f})")
    for index, attack in enumerate(attacks.itertuples(index=False)):
        start, end = pd.Timestamp(attack.start_time), pd.Timestamp(attack.end_time)
        ax.axvspan(start, end, color="#E6A14B", alpha=0.25,
                   label="Ground-truth attack interval" if index == 0 else None)
    ax.set_yscale("log")
    ax.set_xlabel("Target timestamp")
    ax.set_ylabel("Anomaly score (log scale)")
    ax.set_title("Held-out BATADAL test timeline with ground-truth attack intervals", loc="left", weight="bold", pad=12)
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.grid(True, which="both", axis="y", color="#D8DEDF", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper left")
    fig.text(0.125, 0.01, "Timeline uses the saved corrected single-checkpoint test inference; shaded spans are the seven published Test Dataset attack intervals.", fontsize=8.5, color="#46545A")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save_figure(fig, "F2_test_attack_timeline")


def figure_localization(report: dict) -> None:
    methods = list(METHOD_LABELS)
    methods.remove("random_guess")
    oracle = report["timing"]["oracle"]["methods"]
    random = report["random_guess_pct"]
    top1 = [oracle[method]["top1"]["mean"] for method in methods] + [random["top1"]]
    top3 = [oracle[method]["top3"]["mean"] for method in methods] + [random["top3"]]
    top1_err = [oracle[method]["top1"]["std"] for method in methods] + [0]
    top3_err = [oracle[method]["top3"]["std"] for method in methods] + [0]
    labels = [METHOD_LABELS[method] for method in methods] + [METHOD_LABELS["random_guess"]]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11.6, 5.4))
    ax.bar(x - width / 2, top1, width, yerr=top1_err, capsize=4, label="Strict Top-1", color="#397D78", edgecolor="#263B3A", linewidth=0.6)
    ax.bar(x + width / 2, top3, width, yerr=top3_err, capsize=4, label="Strict Top-3", color="#D48B4B", edgecolor="#60452E", linewidth=0.6)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Localization accuracy (%)")
    ax.set_ylim(0, max(top3) + max(top3_err) + 8)
    ax.set_title("Oracle-timed fair localization across five seeds", loc="left", weight="bold", pad=12)
    ax.text(0.01, 0.97, "Mean +/- 1 SD across seeds; random baseline uses 22 candidates", transform=ax.transAxes,
            va="top", fontsize=9, color="#46545A")
    ax.grid(True, axis="y", color="#D8DEDF", linewidth=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save_figure(fig, "F3_fair_localization")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    detection = pd.read_csv(RESULTS / "multiseed_detection_table.csv")
    smote = load_json(RESULTS / "smote_baseline_report.json")["models"]["random_forest"]["metrics"]
    smote_row = {
        "adjacency": "SMOTE Random Forest",
        "dataset": "Dataset04 temporal holdout",
        "seeds": "single model",
        "f1_score": f"{smote['f1_score']:.3f}",
        "recall": f"{smote['recall']:.3f}",
        "precision": f"{smote['precision']:.3f}",
        "roc_auc": f"{smote['roc_auc']:.3f}",
        "pr_auc": f"{smote['pr_auc']:.3f}",
        "false_positive_rate": f"{smote['false_positive_rate']:.3f}",
        "attacks_detected": "N/A",
    }
    detection = detection[detection["adjacency"] != "SMOTE Random Forest"]
    detection = pd.concat([detection, pd.DataFrame([smote_row])], ignore_index=True)
    detection.to_csv(RESULTS / "multiseed_detection_table.csv", index=False)

    localization = load_json(RESULTS / "fair_localization_report.json")
    oracle = pd.read_csv(RESULTS / "fair_localization_table_oracle.csv").fillna("N/A")
    detector = pd.read_csv(RESULTS / "fair_localization_table_detector.csv").fillna("N/A")
    oracle["method"] = oracle["method"].map(METHOD_LABELS)
    detector["method"] = detector["method"].map(METHOD_LABELS)
    r1 = detection.rename(columns={
        "adjacency": "Model / adjacency",
        "dataset": "Evaluation split",
        "seeds": "Runs",
        "f1_score": "F1",
        "roc_auc": "ROC-AUC",
        "pr_auc": "PR-AUC",
        "false_positive_rate": "FPR",
    })
    r2 = oracle.rename(columns={"method": "Method", "top1": "Strict Top-1 (%)", "top3": "Strict Top-3 (%)",
                                "lenient_top1": "Lenient Top-1 (%)", "lenient_top3": "Lenient Top-3 (%)",
                                "top3_actuator": "Top-3 actuator (%)", "top3_concealment": "Top-3 concealment (%)"})
    r3 = detector.rename(columns={"method": "Method", "top1": "Strict Top-1 (%)", "top3": "Strict Top-3 (%)",
                                  "lenient_top1": "Lenient Top-1 (%)", "lenient_top3": "Lenient Top-3 (%)",
                                  "top3_actuator": "Top-3 actuator (%)", "top3_concealment": "Top-3 concealment (%)"})

    figure_pipeline()
    figure_test_timeline()
    figure_localization(localization)

    d04_graph = detection[(detection["adjacency"] == "A_CTOWN") & (detection["dataset"] == "Dataset04")].iloc[0]
    d04_identity = detection[(detection["adjacency"] == "Identity") & (detection["dataset"] == "Dataset04")].iloc[0]
    test_graph = detection[(detection["adjacency"] == "A_CTOWN") & (detection["dataset"] == "Test Dataset")].iloc[0]
    test_identity = detection[(detection["adjacency"] == "Identity") & (detection["dataset"] == "Test Dataset")].iloc[0]
    oracle_d = localization["timing"]["oracle"]["methods"]["D_proposed_ppr_plus_shap"]
    oracle_b = localization["timing"]["oracle"]["methods"]["B_shap_direct_no_graph"]
    agreement = localization["timing"]["oracle"]["D_vs_B_top3"]
    test_file = pd.read_csv(ROOT / "data" / "raw" / "BATADAL_test_dataset.csv", nrows=0)
    test_label_source = "ATTACK_FLAG" if "ATTACK_FLAG" in {c.strip() for c in test_file.columns} else "data/ground_truth_attacks.csv intervals"

    report = f"""# Stage 1 Results

## Experimental setup

Dataset03 was split chronologically, with its first 80% used for model fitting and its final 20% reserved for normal-only calibration. The training sensor readings were rounded to two decimal places before imputation, scaling, or model fitting to match the precision of Dataset04 and the test data. Dataset04 and test windows were scored against their held-out labels. The H=12 input/forecast alignment uses the scored target timestamp at row offset $t + 23$ from the first input row.

Detection used a 12-step SpatioTemporalGNN (hidden dimension 64, dropout 0.2, 50 epochs, batch size 64, Adam at 0.001), with seeds 42, 123, 2024, 3407, and 9999. Each seed was trained with both the 3-hop C-Town adjacency and an identity adjacency. Normal-only calibration set the threshold to the 99.8th percentile. Test labels were sourced from **{test_label_source}** because the local raw test CSV {'contains' if 'ATTACK_FLAG' in {c.strip() for c in test_file.columns} else 'does not contain'} an ATTACK_FLAG column; attack intervals were aligned to target timestamps using $t + 23$.

Fair localization evaluated all 14 published attack intervals with a shared set of {localization['n_candidates']} candidate assets, four methods, and 50 permutation-Shapley permutations. Oracle timing selected evidence within the known attack intervals; detector timing required the detector to flag evidence windows. Strict metrics require the exact target component, while lenient metrics count any attacked component. Values are mean +/- sample standard deviation across five seeds; random guess is the uniform candidate-ranking baseline.

The supervised SMOTE Random Forest row in R1 is a distinct Dataset04 chronological holdout experiment ({load_json(RESULTS / 'smote_baseline_report.json')['models']['random_forest']['test_rows']} rows); it is shown for context and is not a like-for-like comparison with the unsupervised normal-only GNN evaluation.

## Table R1: Detection

{markdown_table(r1)}

## Table R2: Fair localization, oracle timing

{markdown_table(r2)}

## Table R3: Fair localization, detector timing

{markdown_table(r3)}

## Findings and limitations

The C-Town graph did not improve detection over identity adjacency in these five runs. Dataset04 mean F1 was {d04_graph['f1_score']} for C-Town versus {d04_identity['f1_score']} for identity; test mean F1 was {test_graph['f1_score']} versus {test_identity['f1_score']}. These descriptive results do not establish statistical equivalence: no paired inferential test was specified. The result is consistent with the current sensor/time-series representation already carrying most useful predictive information, while fixed graph mixing may provide no additional signal (or may dilute useful channel-specific features); distinguishing those explanations requires a targeted ablation.

For oracle localization, proposed D achieved strict Top-1 {oracle_d['top1']['mean']:.1f}% and Top-3 {oracle_d['top3']['mean']:.1f}%, compared with direct-SHAP B at {oracle_b['top1']['mean']:.1f}% and {oracle_b['top3']['mean']:.1f}%. D and B made identical Top-3 decisions in {agreement['same']}/{localization['timing']['oracle']['detected_runs']} detected attack-seed runs in this execution (D-only {agreement['D_only']}; B-only {agreement['B_only']}). This is the observed result; it differs from the anticipated 68/70 agreement. The combined PageRank+SHAP score did not improve strict Top-3 over direct SHAP, motivating Stage 2 work on stronger, better calibrated hydraulic constraints rather than claiming a localization gain from the current graph prior.

Across the stated replication ranges, Dataset04 and test F1 are close to the approximate targets. Localization is weaker than the rough range: strict Top-1 is {oracle_d['top1']['mean']:.1f}% (target about 21%) and proposed strict Top-3 is {oracle_d['top3']['mean']:.1f}% (target about 30-36%). The benchmark completed, but the localization range gate is **not fully met**; these outputs should be treated as measured results, not tuned to match the expected values.

## Figures

![F1: Stage 1 experiment pipeline](final_figures/F1_stage1_pipeline.png)

![F2: Test timeline and attack intervals](final_figures/F2_test_attack_timeline.png)

![F3: Oracle-timed fair localization](final_figures/F3_fair_localization.png)
"""
    (RESULTS / "STAGE1_RESULTS.md").write_text(report, encoding="utf-8")
    print(f"Wrote {RESULTS / 'STAGE1_RESULTS.md'}")
    print("Figures:", ", ".join(path.name for path in sorted(FIGURES.glob("F[1-3]_stage1*")) + sorted(FIGURES.glob("F[1-3]_test*")) + sorted(FIGURES.glob("F[1-3]_fair*"))))


if __name__ == "__main__":
    main()