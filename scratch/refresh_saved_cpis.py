"""Refresh CPIS fields on existing corrected episodes from saved driver summaries.

Historical episode CSVs contain only the top-five sensor residual summary, not
the full residual vector. New inference runs should generate CPIS directly from
all episode sensor residuals instead of using this refresh utility.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import sys

import pandas as pd
import yaml

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from src.impact.cpis import calculate_cpis, derive_episode_impact_factors, parse_dominant_sensor_scores


RESULTS_DIR = ROOT_DIR / "results" / "test_inference" / "corrected"
EPISODES_CSV = RESULTS_DIR / "test_anomaly_episodes.csv"
REPORT_JSON = RESULTS_DIR / "test_inference_report.json"
SUMMARY_MD = RESULTS_DIR / "test_inference_summary.md"
CONFIG_YAML = ROOT_DIR / "configs" / "config.yaml"
THRESHOLD = float(json.loads(REPORT_JSON.read_text(encoding="utf-8"))["operational_threshold"])
HISTORICAL_BASIS = "Top-five stored dominant sensor residual summaries (historical artifact)"


def main() -> None:
    config = yaml.safe_load(CONFIG_YAML.read_text(encoding="utf-8"))
    impact_config = config.get("impact", {})
    episodes = pd.read_csv(EPISODES_CSV)

    cpis_records = []
    for _, episode in episodes.iterrows():
        sensor_scores = parse_dominant_sensor_scores(episode["dominant_sensors_summary"])
        factors = derive_episode_impact_factors(
            sensor_scores,
            float(episode["duration_hours"]),
            THRESHOLD,
            impact_config,
        )
        cpis_records.append(
            {
                **calculate_cpis(factors, impact_config),
                "cpis_evidence_basis": HISTORICAL_BASIS,
            }
        )

    cpis_frame = pd.DataFrame(cpis_records, index=episodes.index)
    legacy_columns = [
        "cpis_anomaly_component",
        "cpis_persistence_component",
        "cpis_network_component",
        "cpis_topology_node_count",
    ]
    episodes = episodes.drop(columns=legacy_columns, errors="ignore")
    for column in cpis_frame.columns:
        episodes[column] = cpis_frame[column]
    episodes.to_csv(EPISODES_CSV, index=False)

    report = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
    episode_records = json.loads(episodes.to_json(orient="records"))
    for record in episode_records:
        nodes = record.get("spatial_topology_nodes")
        if isinstance(nodes, str):
            try:
                record["spatial_topology_nodes"] = ast.literal_eval(nodes)
            except (SyntaxError, ValueError):
                pass
    report["episodes_ranked_by_severity"] = episode_records
    report.pop("cpis_definition", None)
    report["cpis_methodology"] = {
        "scale": "0-1",
        "formula": "weighted sum of pressure impact, flow impact, service disruption, infrastructure criticality",
        "weights": impact_config.get("weights", {}),
        "severity_thresholds": impact_config.get("severity_thresholds", {}),
        "asset_criticality_priors": impact_config.get("asset_criticality", {}),
        "evidence_basis": HISTORICAL_BASIS,
        "caveat": "Historical factors are estimated from saved top-five sensor summaries; they are not recalculated from full residual vectors.",
    }
    REPORT_JSON.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    summary = SUMMARY_MD.read_text(encoding="utf-8", errors="replace")
    summary = re.sub(
        r"CPIS is a project-specific[^\n]*",
        "CPIS is a configured 0-1 weighted score of pressure impact, flow impact, service disruption, and infrastructure criticality. These historical factors use saved top-five sensor residual summaries; they are prioritization estimates, not probabilities.",
        summary,
    )
    old_cpis_section = "## Cyber-Physical Impact Score (CPIS)"
    if old_cpis_section in summary:
        summary = summary.split(old_cpis_section, 1)[0].rstrip() + "\n"
    summary = re.sub(
        r"\| Rank \| Episode ID \|[^\n]*\n\|[-:| ]+\n(?:\|[^\n]*\n?)+",
        lambda _: _markdown_episode_table(episodes),
        summary,
        count=1,
    )
    SUMMARY_MD.write_text(summary, encoding="utf-8")
    print(f"Updated CPIS for {len(episodes)} historical episodes using saved top-five sensor summaries.")


def _markdown_episode_table(episodes: pd.DataFrame) -> str:
    lines = [
        "| Rank | Episode ID | Duration (h) | CPIS | Severity | Pressure | Flow | Service | Infrastructure |",
        "| :---: | :--- | ---: | ---: | :--- | ---: | ---: | ---: | ---: |",
    ]
    for _, row in episodes.sort_values("severity_rank").iterrows():
        lines.append(
            f"| {int(row['severity_rank'])} | `{row['episode_id']}` | {row['duration_hours']:.1f} "
            f"| **{row['cpis_score']:.3f}** | {row['cpis_severity']} "
            f"| {row['cpis_pressure_impact']:.3f} | {row['cpis_flow_impact']:.3f} "
            f"| {row['cpis_service_disruption']:.3f} | {row['cpis_infrastructure_criticality']:.3f} |"
        )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()