from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_no_unsupported_rf_table_is_reported():
    """Ensure the repository only publishes metrics reproduced by its own scripts."""
    doc_paths = [
        ROOT_DIR / "README.md",
        ROOT_DIR / "results" / "README.md",
        ROOT_DIR / "results" / "final_results_summary.md",
    ]

    forbidden_markers = [
        "rf f1 = 0.9532",
        "roc-auc = 0.9922",
        "random forest f1 = 0.9532",
        "table 5: rf",
    ]

    for doc_path in doc_paths:
        text = doc_path.read_text(encoding="utf-8", errors="ignore").lower()
        for marker in forbidden_markers:
            assert marker not in text, f"Unsupported benchmark claim found in {doc_path.name}: {marker}"
