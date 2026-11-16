# Kshitiz — Individual Viva Study Guide

**Student:** Kshitiz  
**Enrollment number:** 2401010242  
**Role:** Ablation experiments, Streamlit application, test integration, and artifact verification

This guide documents the parts of the project most directly associated with this
role. It is intended to support an individual viva without changing the
experimental or inference code.

## 1. Contribution summary

My contribution was the validation and demonstration layer of the project:

- implemented the Phase 2 ablation suite in `src/ablation.py`;
- developed the Streamlit demonstrator in `app/app.py`;
- added deterministic metric, threshold, and inference replay checks in
  `scripts/verify_artifacts.py`;
- supported demo generation and application/integration tests;
- documented the limitations of model-based schedule sensitivity.

The work made the model comparison auditable and made the saved artifacts
usable through a reproducible historical demonstration.

## 2. Ablation studies

An ablation study changes or removes one component so that its contribution can
be measured rather than assumed. The implementation evaluates PR-AUC, ROC-AUC,
and Brier score using `evaluate_predictions_compact`.

### Hybrid paradigm ablation

| Variant | PR-AUC | ROC-AUC | Brier score |
|---|---:|---:|---:|
| Standalone XGBoost | 0.3397 | 0.6872 | 0.1427 |
| Standalone Deep ResNet | 0.2938 | 0.6568 | 0.1517 |
| Naive 50/50 averaging | 0.3075 | 0.6785 | 0.1447 |
| Learned gated stacking | 0.3374 | **0.6884** | **0.1419** |

The proposed fusion combines raw margins in logit space:

```text
z = w_xgb * logit_xgb + w_dl * logit_dl + intercept
p = sigmoid(z)
```

The learned development weights were approximately `w_xgb = 1.289`,
`w_dl = 0.060`, and `intercept = 0.473`. This indicates that XGBoost supplied
most of the signal for this feature configuration while the neural stream
provided a smaller complementary adjustment.

The hybrid did not win every metric: standalone XGBoost had slightly higher
PR-AUC in this ablation. The defensible conclusion is that learned fusion
improved development ROC-AUC and probability quality, not that it universally
dominates every model.

### Architecture-depth ablation

The study compares a shallow MLP, a standard three-layer MLP, a one-block
ResNet, and a three-block ResNet. A residual block uses:

```text
h_(l+1) = h_l + F(h_l)
```

The identity path helps preserve gradient flow through deeper tabular
representations. In the recorded short ablation, the three-block ResNet
achieved the strongest ROC-AUC among the tested architectures.

### Regularization ablation

The study compares no regularization, dropout, weight decay, and the combined
configuration. Dropout reduces co-adaptation during training, while weight
decay adds an L2 penalty. The measured results were mixed, so the correct
interpretation is empirical: regularization changes the precision, ranking, and
probability-quality trade-off and should not be claimed to improve every
metric automatically.

## 3. Streamlit application

The application is a historical demonstrator, implemented as sidebar pages:

1. **Overview:** dataset scope and model comparison.
2. **Predict a historical flight:** selects a saved model and returns calibrated
   delay probability, threshold, alert decision, and risk tier.
3. **Phase 2 dashboard:** displays learned hybrid weights and ablation results.
4. **Model evaluation:** shows metrics, curves, confusion matrices, and feature
   ablations.
5. **Training-data EDA:** displays descriptive delay patterns.
6. **Rescheduling sensitivity:** compares model risk under schedule shifts.
7. **Method and limitations:** explains the prediction contract and caveats.

The app loads saved models, preprocessors, historical state, calibrators, and
threshold metadata. It does not retrain models when opened.

## 4. Schedule-sensitivity limitation

The simulator changes the schedule of one historical flight while keeping the
background timetable and original historical information fixed. It reports:

```text
risk difference = shifted predicted probability - baseline probability
```

This is predictive sensitivity, not causal evidence. The model does not
represent aircraft rotations, crews, gates, slots, passenger connections,
capacity, or reactions from other flights. Therefore, the app must not be
presented as an airline scheduling optimizer or as proof that rescheduling
reduces real delays.

## 5. Deterministic artifact verification

`scripts/verify_artifacts.py` independently checks each saved model by:

- confirming test prediction IDs are unique;
- recomputing Average Precision and ROC-AUC;
- recomputing raw and calibrated Brier scores;
- checking operational predictions against saved thresholds;
- recomputing the threshold from the threshold-validation block;
- confirming calibration, selection, threshold, and test records are disjoint;
- replaying saved inference on 12 fixed-seed sampled test flights;
- comparing raw and calibrated probabilities with tolerance `1e-7`.

The actual test label is used only after prediction for verification. It is not
included in the replay feature matrix.

The tolerance allows negligible floating-point differences while still
detecting a changed preprocessing path, model artifact, calibration artifact,
or threshold.

## 6. Reproducibility and testing

The verification process is supported by:

- `bundle_metadata.json` for package, policy, and artifact metadata;
- `run_summary.json` for the experiment summary;
- split and prediction artifacts for replayable evidence;
- pinned dependencies and recorded dataset hashes;
- ablation tests for metric keys, variants, and plot generation;
- Streamlit tests for page rendering, prediction feedback, and input errors.

The key distinction is that verification audits saved results; it does not
retrain the models or select a better result after seeing the final test
metrics.

## 7. Viva answers to memorize

**What was your main contribution?**  
I implemented the ablation and demonstration layer: systematic hybrid,
architecture, and regularization comparisons; the Streamlit app; and
deterministic verification of metrics, thresholds, and saved inference.

**Why was naive averaging insufficient?**  
It assumes equal model reliability and averages probabilities without learning
their relative contribution. Gated stacking learns a regularized combination of
the two model logits from held-out predictions.

**Did the hybrid win everything?**  
No. It had the best development Brier score and ROC-AUC in the hybrid ablation,
but XGBoost had slightly higher development PR-AUC and the final test results
also showed that different models lead on different metrics.

**Can the app be deployed in an airport today?**  
No. It is a historical 2013 demonstrator for three origins. It lacks current
operational feeds, weather, rotations, crew, capacity, and prospective
validation.

**Is the rescheduling result causal?**  
No. It is an offline model-sensitivity analysis under a fixed timetable and
does not establish that changing a departure time will reduce actual delay.

**How did you verify the saved predictions?**  
I recomputed the metrics and threshold independently, checked disjoint
evaluation blocks, and replayed saved inference on fixed-seed test samples,
matching probabilities within `1e-7`.

## 8. Closing statement

> My contribution made the project experimentally defensible, user-demonstrable,
> and reproducible. The ablations measured which design choices mattered, the
> Streamlit app exposed the system and its limitations, and deterministic
> verification confirmed that the saved metrics and inference artifacts were
> internally consistent.
