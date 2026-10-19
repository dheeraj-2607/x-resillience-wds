# X-Resilience-WDS evaluation package (review of 19 Oct 2026)

Copy the contents of this folder into the root of the repository
(github.com/dheeraj-2607/x-resillience-wds). It adds files and replaces one data file:

| Path | What it is |
|---|---|
| src/evaluation/labels.py | ground-truth labels, official test labels, sensor -> component mapping |
| src/evaluation/shapley.py | permutation Shapley (same estimator as shap.PermutationExplainer), 50 permutations by default |
| src/evaluation/data_prep.py | builds all splits in one place; fixes the dataset03 precision mismatch |
| src/evaluation/detection_eval.py | model-agnostic detection scoring, multi-seed table |
| src/evaluation/fair_localization.py | fair localization benchmark (4 methods, same candidates, same evidence, top-1/top-3, lenient, random) |
| scratch/run_detection_multiseed.py | trains 5 seeds x {C-Town, identity}, saves checkpoints, writes tables |
| scratch/run_fair_localization.py | runs the fair benchmark on the saved checkpoints |
| tests/test_evaluation_package.py | 8 correctness tests |
| data/ground_truth_attacks.csv | attack 12 end corrected to 2017-02-28 08:00 (matches official labels) |
| reference/ | NumPy re-implementation used in the review + its result files, for comparison only |

Requirements: the existing requirements.txt. Data: data/raw/ must contain
BATADAL_dataset03.csv, BATADAL_dataset04.csv, CTOWN.INP and the LABELLED test file
BATADAL_test_dataset.csv (has an ATTACK_FLAG column;
SHA-256 8a786183003ed3fe0fcec287ab4c585c5ec9c32adadc79586720070583c2e59c).

Run, from the repository root:
    python -m pytest tests/test_evaluation_package.py -q
    python scratch/run_detection_multiseed.py
    python scratch/run_fair_localization.py

Expected ranges (from the review's NumPy replication; PyTorch numbers will differ
slightly, compare as ranges):
  Detection, C-Town graph, harmonised precision: Dataset04 F1 ~0.46, ROC-AUC ~0.79;
  test set F1 ~0.30, ROC-AUC ~0.73.
  Localization (oracle timing): all four methods top-1 ~21%, top-3 ~30-36%; random 4.5% / 13.6%.
If your numbers fall far outside these ranges, stop and investigate before writing.
