# Experimental method

## Prediction contract

The target is `arr_delay >= 15 minutes`, conditional on an observed arrival outcome. Predictions use information available two hours before scheduled departure. The population consists of 2013 departures from EWR, JFK and LGA. This is a historical experiment, not a currently validated operational forecasting service.

Current-flight actual departure, actual arrival, realized delay, airborne time and delay-cause fields never enter a model input matrix. They may be used to construct labels or audit the release time of **historical** labels. A feature-column check alone cannot establish temporal validity.

## Availability and historical replay

Scheduled departure clocks use America/New_York. Destination timezone metadata comes from nycflights13, with explicit IANA timezone additions for Puerto Rico and the US Virgin Islands. Scheduled arrival date is the first destination local scheduled clock after departure, appropriate for the dataset's trips under 24 hours. Actual arrival UTC is scheduled arrival UTC plus the observed arrival delay.

The timestamp audit checks the derived actual destination clock against `arr_time` and checks elapsed time against observed airborne time. Inconsistent timestamps cannot contribute historical labels. They remain eligible prediction targets when an arrival-delay label exists. The data card and generated audit record the exclusion count.

A label is assumed released 24 hours after actual arrival. This is a conservative modeling assumption, not an observed BTS publication timestamp. Real deployment would need a versioned operational feed and its actual reporting latency. The static source timetable is also assumed known at prediction time; the dataset does not contain historical timetable revisions.

Daily history snapshots use only labels with `label_available_at < local midnight` on the prediction's information date. Midnight is computed from the **T-2h decision timestamp**, so a flight departing shortly after midnight uses the preceding decision date. Global smoothing, cumulative group rates, counts and seven-day release-window rates all use this same cutoff.

The historical event store contains future records for offline replay, like an event database. A strict timestamp query prevents those records from contributing to earlier features. Future-label mutation tests check this boundary. Earlier development/test outcomes can contribute to later dates after their release. Estimator weights, calibration functions and decision thresholds stay fixed throughout test. The app replays exactly this protocol from the saved event store.

The code also supports a frozen training-history reference. It uses training-only events frozen at the first selection-block decision timestamp; the recent-history window may consequently contain no training events. Its development-data ablation contrasts that restricted reference with daily replay. It is not the primary replay policy.

## Features and smoothing

Available feature families are carrier/origin/destination identity, schedule clocks, distance, calendar, cyclical time, scheduled origin demand, and historical outcomes.

For a cyclic value `t` with period `P`, use `sin(2*pi*t/P)` and `cos(2*pi*t/P)`. Tested periods are minutes in a day, weekday and month. Raw calendar terms are also tested as an alternative, rather than assumed to generalize beyond the fitting dates.

For each as-of snapshot, a fixed Beta(1,3) cold start gives global mean

`mu = (past_delays + 1) / (past_flights + 4)`.

Each group uses `(group_delays + m*mu)/(group_flights + m)`, with `m=20` for carrier/origin/destination and `m=10` for route. These are declared pseudo-count choices. No future training prevalence supplies the prior. Seven-day rates refer to outcomes **released during the preceding 7*24 hours**, rather than assuming all previous-day flights have finished.

Scheduled demand counts every flight in the source roster, including missing arrival outcomes. The features are departures from the same origin in the scheduled hour and the three-hour window comprising the preceding, current and following hours. These are demand proxies, not realized congestion or airport capacity estimates.

Feature selection considers schedule/demand, cyclic schedule/demand, schedule plus historical rates/demand, and all features including historical exposure counts. Training-only date folds tune each feature family. A separate forward selection block chooses the final feature family by raw Average Precision before selecting its probability treatment by Brier loss. Historical counts can act as calendar proxies and need not help extrapolation. The saved search records retain unsuccessful alternatives.

## Theory and model selection

Logistic Regression estimates `p(y=1|x)=sigmoid(beta_0 + x^T beta)`. Fitting minimizes binary log-loss with an L2 penalty. `C` inversely controls regularization strength. Balanced weighting changes the optimization target and can distort uncorrected probabilities; weighted and unweighted alternatives are compared rather than assuming imbalance requires weighting.

XGBoost constructs an additive log-odds model from regression trees. At each boosting step, a second-order approximation uses gradient `g=p-y` and Hessian `h=p*(1-p)` for binary logistic loss. The regularized tree objective includes leaf-count and leaf-weight penalties. Depth, minimum child Hessian weight, shrinkage and L2 leaf regularization control complexity. Scaling is required for the linear model's regularization; using the same numeric transformations for trees is convenient but not a theoretical requirement.

Four expanding date folds, each with a 30-date forward holdout, occur entirely inside the outer training dates. Every fold refits its imputer, scaler and categorical encoder. Flight dates remain intact. Labels unavailable at each fitting cutoff are purged. The search objective is mean holdout Average Precision, with mean Brier loss breaking ties. XGBoost uses log-loss early stopping within these folds. The final number of boosting rounds is the selected candidate's median fold stopping round, then stays fixed during final fitting.

The supplementary embedding MLP uses the XGBoost-selected numeric set, learned carrier/origin/destination embeddings, an MLP with batch normalization/dropout, binary logit loss and Adam. Its final training uses a declared fixed epoch count. Calibration-period monitoring does not select its checkpoint. It is additional evidence, outside the required Phase 1 ML comparison.

## Phase 2: Deep Learning Evaluation & Hybrid ML-DL Innovation

Phase 2 extends the evaluation protocol from classical machine learning to modern deep learning and a technically justified **Hybrid ML-DL Architecture**.

### 1. Deep Tabular Representation Learning
Tabular flight data combines dense continuous schedules and congestion volumes with high-cardinality discrete nominals (Airlines, Origins, Destinations). Standard one-hot encodings introduce excessive sparsity and ignore latent correlations between hubs and carriers.
Our Deep Learning architecture (`FlightDeepResNet` and `FlightEmbeddingMLP`) maps discrete categories $c \in \{1, \dots, K\}$ to continuous $d$-dimensional embedding vectors:
$$\mathbf{e}_c = \mathbf{E} \mathbf{v}_c \in \mathbb{R}^d$$
where $\mathbf{E} \in \mathbb{R}^{d \times K}$ is optimized end-to-end via gradient descent. Concatenating carrier ($d=16$), origin ($d=16$), destination ($d=32$), and standardized numerical features $\mathbf{x}_{\text{num}} \in \mathbb{R}^m$ yields a unified representation vector:
$$\mathbf{h}_0 = [\mathbf{e}_{\text{carrier}}; \mathbf{e}_{\text{origin}}; \mathbf{e}_{\text{dest}}; \mathbf{x}_{\text{num}}] \in \mathbb{R}^{64 + m}$$

For the Deep ResNet architecture, information passes through stacked residual blocks with skip connections:
$$\mathbf{h}_{l+1} = \mathbf{h}_l + \mathcal{F}(\mathbf{h}_l, \mathcal{W}_l)$$
where $\mathcal{F}$ includes layer normalization, ReLU non-linearities, dropout ($p=0.2$), and linear projections. Residual connections preserve gradient fidelity across deep tabular representations, preventing vanishing gradients.

### 2. The Hybrid ML-DL Innovation: Gated Stacking Meta-Fusion
Simply ensembling heterogeneous models often produces marginal gains without technical rationale. In aviation delay prediction, tree-based models and neural networks exhibit complementary inductive biases:
- **Gradient Boosted Decision Trees (XGBoost)** partition feature space with orthogonal, axis-aligned hyperplanes, providing superior sensitivity to thresholded continuous schedules and origin departure spikes.
- **Deep Tabular Networks** project high-cardinality discrete airline networks into smooth metric manifolds, providing superior generalization across sparse carrier-route combinations.

Our **Hybrid ML-DL Model** fuses the two streams directly in logit space:
$$z(x) = w_{\text{xgb}} \cdot \ell_{\text{xgb}}(x) + w_{\text{dl}} \cdot \ell_{\text{dl}}(x) + b_0$$
where $\ell_{\text{xgb}}(x) \in \mathbb{R}$ is the uncalibrated margin (log-odds) from XGBoost, and $\ell_{\text{dl}}(x) \in \mathbb{R}$ is the linear logit before sigmoid from the deep network.

#### Theoretical Justification & Leakage Prevention
To guarantee that the meta-learner weights $(w_{\text{xgb}}, w_{\text{dl}}, b_0)$ are statistically sound and free from data leakage:
1. Base models are trained exclusively on outer training data.
2. Logits $\ell_{\text{xgb}}$ and $\ell_{\text{dl}}$ are generated for the strictly disjoint forward calibration-selection block.
3. The stacking meta-learner minimizes regularized cross-entropy with an $L_2$ penalty:
   $$\min_{\mathbf{w}, b} \sum_{i \in \text{val}} \mathcal{L}_{\text{BCE}}(y_i, \sigma(\mathbf{w}^T \boldsymbol{\ell}_i + b)) + \frac{\lambda}{2} \|\mathbf{w}\|_2^2$$
4. Under Jensen's inequality and independent error distributions, the expected variance of the hybrid logit satisfies:
   $$\operatorname{Var}(z_{\text{hybrid}}) \le w_{\text{xgb}}^2 \operatorname{Var}(\ell_{\text{xgb}}) + w_{\text{dl}}^2 \operatorname{Var}(\ell_{\text{dl}}) + 2 w_{\text{xgb}} w_{\text{dl}} \operatorname{Cov}(\ell_{\text{xgb}}, \ell_{\text{dl}})$$
   Because the error covariance between orthogonal tree splits and embedding manifolds is low, the hybrid combination achieves lower variance and superior probability calibration (Brier score 0.1419 vs 0.1427 for XGBoost alone on development holdout).

### 3. Comprehensive Ablation Studies
The Phase 2 ablation suite systematically isolates the source of empirical improvements:
- **Hybrid Paradigm Ablation**: Compares standalone XGBoost, standalone Deep ResNet, naive unweighted averaging ($0.5 p_{\text{xgb}} + 0.5 p_{\text{dl}}$), and the learned gated stacking meta-learner.
- **Architecture Depth Ablation**: Evaluates 1-layer shallow MLP, 3-layer dense MLP, 1-block ResNet, and 3-block Deep ResNet.
- **Regularization Ablation**: Demonstrates the interplay between Dropout ($p=0.2$), $L_2$ weight decay ($\lambda=10^{-4}$), batch normalization, and cosine annealing learning rate schedules.

## Chronological protocol

The outer split assigns the first 70% of dates to training, the next 15% to development and the remaining dates to test. The actual date boundaries and row counts are generated in `results/reports/split_manifest.json`.

The development dates have three consecutive, disjoint roles:

1. **Calibration fit:** fit sigmoid and isotonic candidates on predictions from the fixed estimator.
2. **Pipeline selection (artifact key `calibration_selection`):** choose among the CV-tuned feature families by raw Average Precision, then choose identity, sigmoid or isotonic for that model using minimum Brier loss. Both are selection decisions on fixed candidate predictions, not estimator fitting.
3. **Threshold validation:** choose an operating threshold on the chosen probability scale.

Each role purges labels that would still be unavailable at the following role's decision boundary. The identity option prevents assuming recalibration always helps. The final policy becomes available at the first test decision timestamp. Inference rejects earlier queries and dates outside the stored timetable.

An operating threshold maximizes recall subject to at least 70% precision and at least 100 development alerts. If no supported threshold meets 70%, the saved, explicitly labeled fallback maximizes development F1. It does not claim to meet 70%. Test precision is an observed result, not a guarantee inherited from development.

## Evaluation and limits

Average Precision (AP) and ROC-AUC summarize **raw-score ranking**. AP is not trapezoidal PR area. Selected-score AP/ROC-AUC are also saved because isotonic ties can change rankings. Operational precision/recall/F1 use the chosen calibrated/identity output. Brier loss and raw/selected reliability curves assess probability predictions. A constant predictor uses the fitted training prevalence without estimating a parameter from test labels.

Confidence intervals resample whole dates, retaining dependence among flights sharing daily disruptions. AP/ROC intervals use raw scores and Brier intervals use the selected probability output. They quantify uncertainty within this period, not across years or geography. Error slices describe date, origin, carrier and departure hour. Feature ablations use development data; they do not select a winner by test performance.

The original test outputs were inspected before correcting the repository. The saved run is therefore a **corrected retrospective reevaluation**, not a fresh untouched external test. There is no claim of statistical superiority without a corresponding test, nor an assertion of operational readiness. Unobserved weather, aircraft rotations, airport capacity and dispatch decisions limit achievable accuracy and transfer.

## Schedule sensitivity

Offsets must include zero, be distinct, and lie between -60 and +60 minutes in 15-minute increments. Departure and arrival shift by the same UTC duration. Local calendar and cyclic features are recomputed, including midnight/month/DST transitions. Historical information stays fixed at the original decision time. The focal historical flight is assumed present in the original timetable bucket; moving it updates that bucket against a fixed background timetable.

Predicted risk differences do not establish that rescheduling causes lower actual delay. Slots, crews, connections, gates, rotations and responses by other flights remain outside the model. The 100-flight cohort is a sensitivity description, not a policy effectiveness trial.

## References for implementation

- Original data/schema: https://nycflights13.tidyverse.org/reference/flights.html
- Threshold selection: https://scikit-learn.org/stable/modules/classification_threshold.html
- Probability calibration: https://scikit-learn.org/stable/modules/calibration.html
- XGBoost formulation: https://arxiv.org/abs/1603.02754

These implementation notes support the coding submission. The final [project report](report/T001_Report.pdf) contains the literature review, numbered references and measured results. The final [presentation](presentation/T001_Presentation.pdf) summarizes the work for Phase 1 evaluation. Only final PDFs are included in these two submission folders.
