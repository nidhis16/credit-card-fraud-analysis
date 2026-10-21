# Credit Card Fraud Analysis: Project Report

**Purpose:** Understand where and how card fraud occurs, build a model that flags it, and translate the result into decisions a bank (or a supervisor) can act on.
**Data:** ULB / Worldline "Credit Card Fraud Detection": 284,807 transactions by European cardholders over 48 hours (September 2013). Columns: `Time`, `V1`-`V28` (anonymised PCA components), `Amount`, `Class` (1 = fraud).
**Code:** `fraud_pipeline.py`. Every number below comes from `outputs/results.json`.

---

## 1. The project in one paragraph

Fraud is rare (0.17% of transactions), so a model that says "nothing is fraud" is 99.83% accurate and completely useless. The project therefore (a) cleans and validates the data, (b) tests statistically how fraud differs from normal activity, (c) trains three models using a **time-ordered** split so results resemble real deployment, (d) picks the alert threshold using a **business cost model** instead of a default 0.5, and (e) reports results with **confidence intervals**, because with only 74 test-period frauds, a single number would overstate certainty.

---

## 2. What was done, step by step, and why

### Step 1: Load and validate
- Checked that columns match the expected schema, that there are no nulls, that `Class` contains only 0/1, and that no amounts are negative. The pipeline stops with an error if any check fails.
- Found **1,081 exact duplicate rows** and removed them. *Why:* if a duplicate lands in training and its twin in testing, the model is graded on rows it has already seen, which inflates the score.
- Result: 283,726 rows, 473 frauds (0.167%).

### Step 2: Statistical EDA (exploratory data analysis)
Rather than only drawing charts, each pattern was tested for statistical significance.

| Question | Test used | Why this test |
|---|---|---|
| Do fraud amounts differ from legitimate amounts? | Mann-Whitney U | Amounts are heavily skewed, so a t-test is inappropriate |
| Are tiny (<= 1) transactions over-represented in fraud? | Fisher's exact test + odds ratio | Compares two proportions, exact even with small counts |
| Does fraud rate vary by time of day? | Chi-square on 4 six-hour blocks | Compares fraud rates across categories |
| Which anonymised features separate fraud best? | Single-feature AUC + Kolmogorov-Smirnov | Measures how different the two distributions are |

### Step 3: Feature engineering
- `log(1 + Amount)`: compresses the long right tail.
- `hour_sin`, `hour_cos`: encodes hour-of-day as a circle so 23:00 and 00:00 are neighbours.
- Raw `Time` was **deliberately excluded**. It is a running counter from the start of the dataset, so a model using it could not work on future data.

### Step 4: Chronological split
| Set | Share of time | Rows | Frauds | Role |
|---|---|---|---|---|
| Train | first 60% | 170,235 | 342 | Fit the models |
| Validation | next 20% | 56,745 | 57 | Choose the model and the threshold |
| Test | final 20% | 56,746 | 74 | Final, untouched evaluation |

*Why not a random split?* Fraud patterns change over time. A random split lets the model "see the future." Training on the past and testing on the future is how a bank would actually use it.

### Step 5: Models
Four candidates, all trained on the training set only:
1. **Baseline** (always predicts the base rate): the floor any model must beat.
2. **Logistic Regression**: simple and interpretable.
3. **Random Forest**: many decision trees voting.
4. **Gradient Boosting**: trees built sequentially, each correcting the previous.

Class imbalance was handled with **class weights**, which make a missed fraud cost more during training. I chose weights over SMOTE (synthetic oversampling) because they add no artificial data and avoid leakage risks.

**Metric choice:** PR-AUC (area under the precision-recall curve). With 0.17% fraud, ROC-AUC looks excellent almost regardless of quality, while PR-AUC reflects how many false alarms you accept for each fraud caught.

### Step 6: Business-cost threshold
A model outputs a risk score from 0 to 1; someone must decide the cut-off. I used this rule, fitted **on validation data only**:

> Net savings = (value of fraud caught) − (review cost × number of alerts)

with an **assumed** review cost of 5 currency units per alert (editable in `CONFIG`). The chosen threshold (0.953) was then applied unchanged to the test set.

### Step 7: Test evaluation + uncertainty
Bootstrap (300 resamples of the test set) gives 95% confidence intervals.

### Step 8: Business analysis
Alert-budget table (what if analysts can only review the top X%?), permutation feature importance, and drift monitoring using the Population Stability Index (PSI).

---

## 3. Results

### 3.1 Model comparison (test set)

| Model | Test PR-AUC | 95% CI | Test ROC-AUC |
|---|---|---|---|
| Baseline | 0.001 | n/a | 0.500 |
| Logistic Regression | 0.745 | 0.62 - 0.85 | 0.981 |
| Random Forest | 0.814 | 0.72 - 0.89 | 0.937 |
| **Gradient Boosting (selected)** | 0.798 | 0.71 - 0.88 | 0.984 |

**Honest reading:** Gradient Boosting was chosen because it had the best *validation* PR-AUC (0.792). On the test set, Random Forest scored slightly higher, but the confidence intervals overlap heavily. **The three models cannot be statistically separated with 74 test frauds.** Claiming "Gradient Boosting is best" would be overclaiming. What is clear is that all three are hundreds of times better than the baseline.

### 3.2 Operating point on the test set (threshold = 0.953)

| Metric | Value |
|---|---|
| Frauds caught / missed | 54 / 20 |
| False alarms | 8 (out of 56,672 legitimate transactions) |
| Precision | 87.1% (95% CI 79 - 94%) |
| Recall | 73.0% (95% CI 63 - 83%) |
| Alerts per 10,000 transactions | 10.9 |
| Fraud value caught | 4,841 of 7,728 (**62.6%**) |
| Review cost | 310 |
| Net savings | 4,531 over the 7.7-hour test window |

---

## 4. Business insights

**Insight 1: Fraud skews to very small transactions ("card testing").**
36.2% of fraud transactions are <= 1 currency unit, versus 10.6% of legitimate ones (odds ratio 4.75, Fisher p < 1e-40). The fraud median amount (9.82) is below the legitimate median (22.00). The fraud rate is 56 per 10,000 for transactions <= 1, versus 6-10 per 10,000 in the middle bands. This is consistent with criminals testing whether a stolen card works before using it. *Caveat: this is a hypothesis consistent with the data. The dataset has no card ID, so I cannot link a test to a later large fraud.*

**Insight 2: Fraud risk is U-shaped by amount.**
The rate climbs again for large transactions (28 per 10,000 at 200-1,000 and 31 per 10,000 above 1,000, though the top band has only 9 frauds). Two different fraud behaviours exist: cheap probing and high-value cash-out.

**Insight 3: Night-time fraud rates are about 3-4 times higher.**
In the first six hours of the daily cycle, fraud is 48 per 10,000 vs 11-17 in other blocks (chi-square = 162, p < 1e-30). This holds on **both days** (44.7 and 52.0 per 10,000), so it is not a single burst. *Caveat: the dataset does not disclose the start clock-time, so "hour" means hours since the data began, not local time.* By contrast, the spike at hour 11 is **one-day only** (43 frauds on day 1 vs 10 on day 2), which looks like an isolated attack burst, not a pattern. Only two days of data exist, so time-of-day findings are indicative, not conclusive.

**Insight 4: A small review team captures most of the benefit.**
Reviewing only the top **0.1%** of transactions by risk (57 alerts in the test window) catches 69% of frauds at 89% precision and protects 61% of fraud value. Going to 1% review lifts recall to 88%, and going from 1% to 2% adds nothing. There are sharply diminishing returns, which is useful for staffing and alert-capacity planning.

**Insight 5: Counting frauds and counting rupees tell different stories.**
Recall by count is 73%, but by value it is 62.6%. The 28 transactions at the very top of the risk ranking are all genuine frauds (100% precision) yet represent only 9% of fraud value, because they are mostly small card-testing transactions. Meanwhile, **the largest missed fraud (1,097) received a risk score of 0.0005**, so the model did not even suspect it. A supervisor judging a bank's fraud system on "number of frauds detected" alone could be misled; **value-weighted detection should be reported alongside count-based detection.**

**Insight 6: The review-cost assumption matters.**
The median caught fraud is only 1.80, below the assumed 5-unit review cost. So the cost model treats many alerts as loss-making. In practice, blocking a test transaction may prevent a much larger later fraud, which this dataset cannot show. The real value of catching small frauds is therefore **understated**, and the 5-unit cost is an assumption to be replaced with a real figure.

**Insight 7: Detection rests on a few signals.**
Permutation importance is dominated by V14 (a drop of 0.142 in PR-AUC when shuffled), then V4 (0.054), then V12, V17 and V10 (about 0.01 each). Concentration in a few features means that if fraudsters change behaviour on those signals, performance could fall quickly, which is a case for ongoing monitoring.

**Insight 8: Monitoring.**
The score distribution was stable between validation and test (PSI = 0.012, below the 0.1 "stable" threshold). Many individual features showed PSI > 0.1, but the extreme PSI on `hour_sin/hour_cos` (about 5) is an **artefact**: the training window covers hours 0-33, while the test window covers only evening-to-night hours. This illustrates why drift alerts need interpretation before action.

---

## 5. Limitations (state these yourself in an interview)

1. **Old, European, anonymised data (2013).** The V1-V28 features are PCA outputs, so I cannot say what they mean economically. Findings describe this dataset and do not directly transfer to Indian payment systems such as UPI.
2. **Very few frauds in the evaluation window** (57 validation, 74 test). Hence the wide confidence intervals and the inability to rank the models.
3. **Only two days.** Weekly or seasonal fraud patterns are not observable.
4. **No customer, card or merchant identifiers.** Velocity features (e.g. "5 transactions in 2 minutes") cannot be built, and they are the strongest real-world fraud signals.
5. **Assumed review cost** (5 units) and an **unknown currency** (presumably euros).
6. **Fraud labels are assumed correct and complete.** Undetected fraud would be labelled legitimate.
7. The threshold was tuned on a validation window with 57 frauds, so it is itself uncertain.

---

## 6. Resume bullets (use only what you can defend)

- Built an end-to-end fraud analytics pipeline on 283,726 card transactions (0.17% fraud): schema validation, deduplication, statistical EDA, and a leakage-free chronological train/validation/test design.
- Used Mann-Whitney, Fisher's exact and chi-square tests to show that fraud concentrates in micro-transactions (OR = 4.75) and night hours (about 3-4x baseline rate).
- Trained and compared logistic regression, random forest and gradient boosting with class weighting; evaluated with PR-AUC and bootstrap 95% CIs (best test PR-AUC about 0.80 vs 0.001 baseline).
- Selected the alert threshold via a cost-benefit model; at the chosen point, caught 73% of frauds at 87% precision, and showed that reviewing 0.1% of transactions captures 69% of fraud.
- Identified a gap between count-based recall (73%) and value-based recall (63%), and added PSI drift monitoring.

---

## 7. Likely interview questions and answers

**Why not use accuracy?**
Predicting "legitimate" for everything gives 99.83% accuracy while catching zero fraud. PR-AUC, precision and recall measure what matters.

**Why a time-based split?**
It prevents the model from learning from the future and mimics deployment. A random split would give optimistic numbers.

**Why did you remove duplicates?**
Identical rows in both train and test make the test score unreliable.

**Why class weights instead of SMOTE?**
SMOTE invents synthetic fraud points, and if applied before splitting it leaks information. Class weights reweight the loss without touching the data.

**Why choose Gradient Boosting if Random Forest scored higher on test?**
Selection was made on validation data only, to avoid tuning to the test set. The test difference is also well within the bootstrap confidence intervals, so the models are statistically indistinguishable.

**What does the PSI tell you?**
It measures how much a variable's distribution has shifted between two periods. Under 0.1 is stable, 0.1-0.25 is moderate, and above 0.25 is major. I used it on the model score and on features.

**How would a regulator or supervisor use this?**
Fraud reporting is a supervisory data problem. The same techniques apply to analysing reported fraud returns, comparing banks, spotting unusual reporting patterns, and recommending value-weighted metrics.

**What would you do next with more data?**
Add card/merchant identifiers for velocity features, use rolling-window retraining, calibrate the probabilities, and replace the assumed review cost with a measured one.

---

## 8. File guide

| File | What it is |
|---|---|
| `fraud_pipeline.py` | The whole pipeline, with `# %%` cell markers for notebook-style use |
| `outputs/results.json` | All numerical results |
| `outputs/figures/01-08_*.png` | Charts used in this report |
| `outputs/tables/*.csv` | Model comparison, alert-budget gain table, importance, PSI |
| `outputs/model/fraud_model.joblib` | Trained model, threshold, feature list |
