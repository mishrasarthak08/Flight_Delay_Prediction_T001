# Phase 2 evidence and coverage matrix

| Rubric / claim | Evidence | Qualification |
|---|---|---|
| Population, EDA and missingness | results/reports/{dataset_profile.json,raw_dataset_audit.json}; docs/DATA_CARD.md; src/data.py | Missing arrival is not confirmed cancellation; roster differs from labeled sample. |
| Time and leakage controls | src/{timeutils,features,splitting}.py; tests/test_temporal_features.py; split_manifest.json | 24-hour lag and complete timetable are assumptions, not verified operational availability. |
| Split, training and selection | scripts/run_pipeline.py; src/{train,experiments,calibration}.py; split_manifest.json; temporal_search.json | Prior test inspection; corrected retrospective evaluation. Saved source hash matches the base revision; current changes in hashed files are import cleanups. Dirty-tree contents outside the hash are unknown. |
| ML, neural, hybrid architecture | src/{models,train,preprocessing}.py; pipeline calls; pytorch_training_history.json | Final hybrid uses MLP, not ResNet; fusion and calibrator share fit data. |
| Results, calibration and uncertainty | evaluation_metrics.json; model_comparison.csv; prediction CSVs; calibration_selection.json | AP is not trapezoidal PR area. Hybrid does not beat either component on raw test AP. |
| Feature and neural ablations | feature_ablation.csv; ablation_studies.csv; src/ablation.py and pipeline calls | Fusion ablation evaluated in sample; architecture, capacity, training budget and random variation confounded. |
| Rescheduling | src/reschedule.py; scenario sample; app/app.py | Offline fixed-background sensitivity; no causal or operational outcome. |
| Literature | Primary papers, official nycflights13 documentation and Kaggle source listed in bibliography | Dataset mirror page did not expose readable content; repository records its exact URL and checksum. |
| Software and reproducibility | tests/; scripts/verify_artifacts.py; run_summary.json | Raw data absent; no full rerun or artifact replay claimed. Saved hash matches the base revision, but the run had a dirty tree. |
| Team / viva | README.md and docs/VIVA_VOCE_GUIDE.md | Names and IDs agree across repository records; role claims not independently verified. Paper supports but cannot replace presentation or individual viva. |
