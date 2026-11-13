# Phase 2 Individual Viva Voce Assessment Guide
**Team T001 · Applied Machine Learning & Deep Learning Capstone**
**Project:** Flight Delay Prediction and Operational Rescheduling via GBDT-Deep Learning Hybridization

---

## 1. Executive Summary & Architecture Overview

### Problem Statement
Predicting whether a commercial flight departing from the New York metropolitan area (EWR, JFK, LGA) will experience an arrival delay of $\ge 15$ minutes (`is_delayed = 1`), evaluated strictly **two hours before scheduled departure** ($T-2\text{h}$) conditional on an observed arrival outcome.

### Phase 2 Innovation: The GBDT + Deep Learning Hybrid Model
Flight delay prediction exhibits a dual nature:
1. **Dense continuous operational features** (scheduled departure minutes, airport hourly departure demand, route distance): Tree-based models (XGBoost) excel at forming sharp, axis-aligned orthogonal partitions.
2. **Sparse high-cardinality categorical entities** (Airline Carrier, Origin Airport, Destination Airport): Deep Neural Networks with learned entity embeddings project discrete categories into continuous latent geometric spaces that capture semantic airline network topology.

Our **Hybrid ML-DL Model** unifies both models in logit space:
$$z(x) = w_{\text{xgb}} \cdot \ell_{\text{xgb}}(x) + w_{\text{dl}} \cdot \ell_{\text{dl}}(x) + b_0$$
where $\ell_{\text{xgb}}(x) \in \mathbb{R}$ is the XGBoost margin (uncalibrated log-odds) and $\ell_{\text{dl}}(x) \in \mathbb{R}$ is the linear logit before activation from the Deep Neural Network. Stacking weights are fit strictly on held-out development validation data using $L_2$-regularized logistic regression, guaranteeing **zero target leakage** and optimal variance reduction.

---

## 2. Team Member Contributions & Individual Viva Profiles

| Member | Enrollment No. | Primary Role & Core Contribution | Code Ownership |
|---|---|---|---|
| **Sarthak Mishra** | 2401010421 | Team Lead, Deep Learning & Hybrid Architecture Design | `src/models.py`, `src/train.py` |
| **Shane** | 2401020062 | Data Pipeline, Temporal Splitting & Leakage Safeguards | `src/data.py`, `src/splitting.py`, `src/timeutils.py` |
| **Shitanshu Tiwari** | 2401010440 | GBDT (XGBoost) Modeling & Temporal Feature Engineering | `src/features.py`, `src/experiments.py` |
| **Preetish Ubhrani** | 2401020050 | Probability Calibration & Operational Decision Thresholds | `src/calibration.py`, `src/evaluate.py` |
| **Kshitiz** | 2401010242 | Ablation Experiment Suite, Streamlit App & Replay Verification | `src/ablation.py`, `app/app.py`, `scripts/verify_artifacts.py` |

---

## 3. Individual Question & Answer Defense Matrix

---

### Member 1: Sarthak Mishra (2401010421)
**Focus:** Deep Learning Architecture, Tabular Entity Embeddings & Hybrid Meta-Fusion

#### Q1: Why did you design a hybrid model instead of using a standalone Deep Learning model or XGBoost alone?
> **Answer:** "A standalone XGBoost model struggles with rare categories because tree splitting on one-hot features leads to high variance and sparse leaf nodes. Conversely, standard Deep Neural Networks often struggle on tabular data with axis-aligned decision boundaries (such as sharp congestion spikes at specific departure hours). 
> Our hybrid approach leverages **complementary inductive biases**: XGBoost captures hard operational thresholds on continuous features, while our Deep Embedding network maps categorical entities into smooth continuous embedding spaces ($\mathbb{R}^{16}$ and $\mathbb{R}^{32}$). By stacking them in logit space ($z = w_{\text{xgb}} \ell_{\text{xgb}} + w_{\text{dl}} \ell_{\text{dl}} + b$), the hybrid model achieves a superior Brier score (0.1419 vs 0.1427 on validation holdout) and higher discrimination than either constituent model alone."

#### Q2: How does your Entity Embedding layer work mathematically?
> **Answer:** "For a discrete nominal category $c \in \{1, \dots, K\}$, we maintain a trainable weight matrix $\mathbf{E} \in \mathbb{R}^{d \times K}$. The forward pass performs an index lookup equivalent to multiplying $\mathbf{E}$ by a one-hot vector $\mathbf{v}_c$:
> $$\mathbf{e}_c = \mathbf{E} \mathbf{v}_c$$
> The gradients backpropagate through the embeddings:
> $$\frac{\partial \mathcal{L}}{\partial \mathbf{E}_{*, c}} = \frac{\partial \mathcal{L}}{\partial \mathbf{e}_c}$$
> This allows the model to learn semantic geometric distances between airports and carriers—e.g., major hub airports cluster together in latent space without manual feature engineering."

#### Q3: What is the mathematical formulation of your Deep Tabular ResNet?
> **Answer:** "In `FlightDeepResNet`, we employ stacked residual blocks. For layer $l$:
> $$\mathbf{h}_{l+1} = \mathbf{h}_l + \text{Dropout}(\mathbf{W}_2 \text{ReLU}(\text{BatchNorm}(\mathbf{W}_1 \text{BatchNorm}(\mathbf{h}_l))))$$
> The identity skip connection $\mathbf{h}_l$ ensures that $\frac{\partial \mathcal{L}}{\partial \mathbf{h}_l} = \frac{\partial \mathcal{L}}{\partial \mathbf{h}_{l+1}} + \frac{\partial \mathcal{F}}{\partial \mathbf{h}_l}$, which allows uninterrupted gradient flow and prevents vanishing gradients across deep tabular layers."

---

### Member 2: Shane (2401020062)
**Focus:** Chronological Partitioning, Strict Availability & Information Leakage Prevention

#### Q1: Why did you forbid standard K-fold cross-validation in this project?
> **Answer:** "Standard random K-fold cross-validation assumes independent and identically distributed (i.i.d.) observations. In aviation operations, flights exhibit strong temporal autocorrelation (weather fronts, cascading reactionary delays, seasonal traffic). Random splitting would allow flights from tomorrow to predict flights from yesterday, leaking future congestion and outcomes into the past. 
> We enforced a **strict chronological partition**: the first 70% of dates for training, the next 15% for development (disjointly split into calibration fit, pipeline selection, and threshold validation), and the final 15% for unseen test holdout. Date boundaries are never crossed."

#### Q2: What is the T-2h prediction horizon, and how did you prevent target leakage?
> **Answer:** "Our prediction contract specifies that delay forecasts must be made exactly 2 hours before scheduled departure (`sched_dep_time - 2h`). 
> We maintain an explicit `FORBIDDEN_COLUMNS` audit in `src/config.py` that strictly excludes actual departure time (`dep_time`), departure delay (`dep_delay`), taxi times (`taxi_out`, `taxi_in`), airborne time (`air_time`), and delay causes (`weather_delay`, `nas_delay`). Our automated validation script `audit_feature_matrix_for_forbidden_columns` runs before every train and inference pass and throws an exception if any non-predictive column appears."

#### Q3: How does the historical outcome reporting lag work?
> **Answer:** "In real-world aviation, official BTS outcome tables are not published instantly upon landing. We implemented a conservative **24-hour reporting lag**: a flight's arrival delay label is only released at $\text{actual\_arrival} + 24\text{h}$. 
> Furthermore, historical features use local-midnight snapshots: a flight departing on date $D$ at 08:00 only sees outcomes released prior to midnight preceding the $T-2\text{h}$ decision timestamp."

---

### Member 3: Shitanshu Tiwari (2401010440)
**Focus:** XGBoost GBDT, Feature Engineering & Expanding Date-Fold Tuning

#### Q1: What objective function and optimization technique does XGBoost use?
> **Answer:** "XGBoost minimizes the regularized objective:
> $$\mathcal{L}^{(t)} = \sum_{i=1}^n l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t)$$
> where $\Omega(f_t) = \gamma T + \frac{1}{2} \lambda \sum_{j=1}^T w_j^2$. Using a second-order Taylor expansion:
> $$\mathcal{L}^{(t)} \approx \sum_{i=1}^n \left[ g_i f_t(x_i) + \frac{1}{2} h_i f_t^2(x_i) \right] + \Omega(f_t)$$
> For binary logistic loss, the first and second derivatives are:
> $$g_i = p_i - y_i, \quad h_i = p_i (1 - p_i)$$
> The optimal leaf weight is $w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$, and tree splits maximize split gain."

#### Q2: How did you design cyclical time features?
> **Answer:** "Time features like departure hour (0–23), day of week (0–6), and month (1–12) have periodic boundaries (e.g., 23:59 is adjacent to 00:01). Linear representations treat 23 and 0 as distant extremes. We transformed each time variable $t$ with period $P$ into orthogonal trigonometric components:
> $$x_{\sin} = \sin\left(\frac{2\pi t}{P}\right), \quad x_{\cos} = \cos\left(\frac{2\pi t}{P}\right)$$
> This allows linear models and neural networks to treat cyclic transitions smoothly."

#### Q3: What is empirical Bayes historical smoothing and why was it necessary?
> **Answer:** "Raw historical delay rates for low-volume routes or carriers exhibit high variance (e.g., a route with 1 delayed flight out of 1 has an unregularized rate of 100%). We applied Empirical Bayes smoothing with pseudo-counts $m$:
> $$\hat{p}_{\text{group}} = \frac{\text{delays}_{\text{group}} + m \cdot \mu_{\text{prior}}}{\text{flights}_{\text{group}} + m}$$
> where $\mu_{\text{prior}} = \frac{\text{past\_delays} + 1}{\text{past\_flights} + 4}$ from a $\text{Beta}(1, 3)$ cold start. We set $m=20$ for carriers/airports and $m=10$ for routes, pulling small samples towards the historical prior."

---

### Member 4: Preetish Ubhrani (2401020050)
**Focus:** Probability Calibration, Operational Thresholding & Decision Theory

#### Q1: Why is calibration critical for operational flight rescheduling?
> **Answer:** "Uncalibrated machine learning models may rank flights well (high ROC-AUC) but output distorted probabilities (e.g. predicting 0.80 when true empirical risk is only 0.35). For operational schedule sensitivity and passenger rebooking, decisions depend on true risk differences $\Delta p = p_{\text{shifted}} - p_{\text{baseline}}$. If probabilities are poorly calibrated, risk estimates are economically meaningless. 
> We evaluate calibration using **Brier Score** ($\frac{1}{N}\sum (p_i - y_i)^2$) and Reliability Diagrams, comparing Platt scaling (Sigmoid) and non-parametric Isotonic Regression."

#### Q2: What is the difference between Platt Scaling and Isotonic Regression?
> **Answer:** "Platt Scaling fits a parametric logistic transformation on the raw scores:
> $$p_{\text{cal}} = \frac{1}{1 + \exp(A \cdot s + B)}$$
> It assumes the uncalibrated scores are normally distributed within classes, preserving monotonicity with only 2 parameters.
> Isotonic Regression fits a non-parametric piecewise constant isotonic step function using the Pool Adjacent Violators Algorithm (PAVA). It makes no parametric assumptions but can overfit on small samples. In our pipeline, the forward selection block selects the calibration method that minimizes Brier score on unseen development data."

#### Q3: How did you select the operational decision threshold?
> **Answer:** "Rather than defaulting to an arbitrary 0.50 cutoff (which is suboptimal under severe class imbalance), we formulated an operational objective: achieve **70% precision** subject to minimum alert volume support. 
> On the dedicated threshold validation block, we swept candidate thresholds. When 70% precision is mathematically infeasible due to irreducible noise, our automated policy switches to maximizing development $F_1$, explicitly documenting the fallback to avoid silent operational degradation."

---

### Member 5: Kshitiz (2401010242)
**Focus:** Ablation Study Experiments, Streamlit Dashboard & Deterministic Replay

#### Q1: What were the key findings of your Phase 2 Ablation Studies?
> **Answer:** "We conducted four systematic ablation studies in `src/ablation.py`:
> 1. **Hybrid Paradigm Ablation**: Standalone XGBoost achieved Brier score 0.1427. Standalone Deep ResNet achieved Brier 0.1517. Naive probability averaging achieved Brier 0.1447. The **Learned Gated Stacking Hybrid** achieved the lowest Brier score (**0.1419**) and highest ROC-AUC (**0.6884**), proving that learned logit fusion outperforms simple heuristic ensembles.
> 2. **Architecture Depth Ablation**: Deep ResNet with residual skip connections outperformed shallow 1-layer networks, demonstrating that residual blocks preserve representation capacity in deep tabular networks.
> 3. **Regularization Ablation**: Combining Dropout ($p=0.2$) and Weight Decay ($\lambda=10^{-4}$) prevented overfitting on the training partition."

#### Q2: How does your Streamlit interactive application demonstrate the Phase 2 Hybrid Model?
> **Answer:** "In `app/app.py`, we created the **Phase 2 Deep Learning & Hybrid Dashboard**. Users can select any historical flight from the test holdout and inspect:
> - Real-time delay probability predictions from the Hybrid model alongside XGBoost and Logistic Regression.
> - Learned fusion weights ($w_{\text{xgb}}, w_{\text{dl}}, b$) showing how the model weights tree decisions against deep neural representations.
> - Interactive ablation tables and plots showing empirical justification.
> - Schedule sensitivity simulator showing predicted delay risk reduction under departure shifts (e.g. $\pm 30$ minutes)."

#### Q3: How do you verify artifact integrity and prevent model drift?
> **Answer:** "We built `scripts/verify_artifacts.py`. It independently recomputes PR-AUC, ROC-AUC, Brier score, and threshold precision/recall against the saved predictions, and executes 12 end-to-end inference replays matching test records to float precision ($10^{-7}$). 
> Furthermore, all pipeline runs produce `bundle_metadata.json` and `run_summary.json` containing SHA-256 hashes of the dataset, package versions, and Git revisions."

---

## 4. Viva Voce Examiner "Trap" Questions & Defense Strategies

| Trap Question | Bad / Flawed Answer | Correct Top-Scoring Defense |
|---|---|---|
| *"Why didn't you achieve 90% accuracy?"* | *"The dataset was too small or our model needed more layers."* | *"Accuracy is misleading under class imbalance (~26% delay prevalence). A naive model predicting 'no delay' achieves 74% accuracy but zero recall. We evaluate Average Precision (PR-AUC) and Brier Score, which penalize distorted probabilities."* |
| *"Can an airline use your rescheduling simulator to eliminate delays tomorrow?"* | *"Yes, airlines can just move the flights to the recommended times."* | *"No. Our schedule sensitivity model is explicitly **non-causal**. Moving a flight changes network congestion, crew rotations, and gate capacity, which our static model cannot observe. We include an explicit causal disclaimer in our app and report."* |
| *"Why not train XGBoost and Deep Learning on all data together?"* | *"To save time and get higher accuracy."* | *"Fitting meta-learners on training data introduces severe optimistic bias (leakage). The meta-learner must be fit on held-out validation predictions to learn true out-of-sample generalization weights."* |

