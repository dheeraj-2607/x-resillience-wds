"""Verify SHA-256 checksums for the authoritative BATADAL data files."""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
EXPECTED_SHA256 = {
    "data/raw/BATADAL_dataset03.csv": "8ca6cb851242254d2605b5a53ba3ba5009e2213a545d65396615f1ce0b426e1b",
    "data/raw/BATADAL_dataset04.csv": "4746beab2cfcdb5e68c7fa197a7522aecb70189d61dcc3eb1f870d2ce387068b",
    "data/raw/BATADAL_test_dataset.csv": "184dfc1581df52193b2acb79164f7ce094f2bb00b1bc2e3c2e70050095065fa5",
    "data/raw/CTOWN.INP": "9198b0bb43f551b29bfb550e206ff95f1df220118fc0068c7f130cec78d58d9e",
}


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest for a file, streaming in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    """Print hash comparisons and return nonzero if any file is missing or changed."""
    all_match = True
    for relative_path, expected_hash in EXPECTED_SHA256.items():
        path = ROOT_DIR / relative_path
        if not path.is_file():
            print(f"MISSING  {relative_path}")
            all_match = False
            continue

        actual_hash = sha256_file(path)
        matches = actual_hash == expected_hash
        status = "MATCH" if matches else "MISMATCH"
        print(f"{status:8} {relative_path} {actual_hash}")
        all_match = all_match and matches

    return 0 if all_match else 1


if __name__ == "__main__":
    raise SystemExit(main())