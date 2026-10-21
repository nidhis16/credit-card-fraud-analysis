"""
Credit Card Fraud Analysis - End-to-End Pipeline
=================================================
Dataset : ULB / Worldline "Credit Card Fraud Detection" (European cardholders,
          two days of transactions, Sept 2013). Columns: Time, V1..V28 (PCA-
          anonymised), Amount, Class (1 = fraud).

Stages  : 1. Load + data-quality validation
          2. Statistical EDA (hypothesis tests, effect sizes)
          3. Feature engineering
          4. Chronological train / validation / test split (no leakage)
          5. Model training + comparison (imbalance-aware)
          6. Cost-based threshold selection (on validation only)
          7. Final test evaluation + bootstrap confidence intervals
          8. Business analysis: alert budget, value captured, drift (PSI)
          9. Save figures, tables, model and results.json

Run     : python fraud_pipeline.py --data data/creditcard.csv --out outputs
Cells   : '# %%' markers let you run it block-by-block in VS Code / Jupyter.
"""

# %% [0] Imports and configuration
import argparse
import json
import time
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

CONFIG = {
    "seed": 42,
    "train_frac": 0.60,       # chronological split: first 60% of time
    "val_frac": 0.20,         # next 20% (threshold selection)  -> last 20% = test
    "review_cost": 5.0,       # ASSUMPTION: cost (currency units) to manually review one alert
    "min_precision_target": 0.80,
    "n_bootstrap": 300,
    "alert_budgets": [0.0005, 0.001, 0.002, 0.005, 0.01, 0.02],  # share of txns reviewed
}

EXPECTED_COLUMNS = (
    ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount", "Class"]
)
V_COLS = [f"V{i}" for i in range(1, 29)]


def jsonable(obj):
    """Convert numpy / pandas types so results can be dumped to JSON."""
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# %% [1] Load and validate
def load_and_validate(path):
    """Load the CSV and run explicit data-quality checks. Fails loudly on bad data."""
    df = pd.read_csv(path)

    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError("Unexpected columns. Expected: %s" % EXPECTED_COLUMNS)

    report = {
        "rows_raw": int(len(df)),
        "null_cells": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "class_values": sorted(df["Class"].unique().tolist()),
        "negative_amounts": int((df["Amount"] < 0).sum()),
        "zero_amounts": int((df["Amount"] == 0).sum()),
        "time_sorted": bool(df["Time"].is_monotonic_increasing),
        "time_span_hours": float(df["Time"].max() / 3600),
    }

    if report["null_cells"] > 0:
        raise ValueError("Dataset contains null values.")
    if report["class_values"] != [0, 1]:
        raise ValueError("Class column must contain only 0 and 1.")
    if report["negative_amounts"] > 0:
        raise ValueError("Negative transaction amounts found.")

    # Exact duplicates would let the same row appear in train AND test -> optimistic metrics.
    df = df.drop_duplicates().sort_values("Time", kind="stable").reset_index(drop=True)
    report["rows_clean"] = int(len(df))
    report["fraud_count_clean"] = int(df["Class"].sum())
    report["fraud_rate_clean_pct"] = float(df["Class"].mean() * 100)
    return df, report


# %% [2] Statistical EDA
def run_eda(df, fig_dir):
    out = {}
    y = df["Class"].values
    fraud, legit = df.loc[y == 1, "Amount"], df.loc[y == 0, "Amount"]

    # --- 2a. Amount: fraud vs legitimate -------------------------------------------
    u = stats.mannwhitneyu(fraud, legit, alternative="two-sided")
    auc_amount = u.statistic / (len(fraud) * len(legit))  # P(fraud amount > legit amount)
    out["amount"] = {
        "fraud_mean": float(fraud.mean()), "legit_mean": float(legit.mean()),
        "fraud_median": float(fraud.median()), "legit_median": float(legit.median()),
        "fraud_p25": float(fraud.quantile(0.25)), "legit_p25": float(legit.quantile(0.25)),
        "fraud_max": float(fraud.max()), "legit_max": float(legit.max()),
        "mannwhitney_p": float(u.pvalue),
        "prob_fraud_amount_exceeds_legit": float(auc_amount),
        "share_fraud_le_1": float((fraud <= 1).mean()),
        "share_legit_le_1": float((legit <= 1).mean()),
        "share_fraud_le_10": float((fraud <= 10).mean()),
        "share_legit_le_10": float((legit <= 10).mean()),
        "total_fraud_value": float(fraud.sum()),
        "total_value": float(df["Amount"].sum()),
    }
    # share of fraud with amount <= 1: test vs legit via Fisher exact
    a = int((fraud <= 1).sum()); b = int(len(fraud) - a)
    c = int((legit <= 1).sum()); d = int(len(legit) - c)
    odds, p_fisher = stats.fisher_exact([[a, b], [c, d]])
    out["amount"]["micro_txn_odds_ratio"] = float(odds)
    out["amount"]["micro_txn_fisher_p"] = float(p_fisher)

    # --- 2b. Fraud rate by amount band ---------------------------------------------
    bins = [-0.01, 1, 10, 50, 200, 1000, np.inf]
    labels = ["<=1", "1-10", "10-50", "50-200", "200-1000", ">1000"]
    band = pd.cut(df["Amount"], bins=bins, labels=labels)
    bt = df.groupby(band, observed=False).agg(
        txns=("Class", "size"), frauds=("Class", "sum"), value=("Amount", "sum"))
    bt["fraud_rate_per_10k"] = bt["frauds"] / bt["txns"] * 1e4
    out["amount_bands"] = bt.reset_index().rename(columns={"Amount": "band"}).to_dict("records")

    # --- 2c. Time-of-day pattern ---------------------------------------------------
    # 'Time' = seconds since first transaction (start clock-time is not disclosed),
    # so 'hour' is hour-of-day RELATIVE to the data start, not local clock time.
    hour = ((df["Time"] % 86400) // 3600).astype(int)
    ht = df.groupby(hour).agg(txns=("Class", "size"), frauds=("Class", "sum"))
    ht["fraud_rate_per_10k"] = ht["frauds"] / ht["txns"] * 1e4
    out["hourly"] = ht.reset_index().rename(columns={"Time": "hour"}).to_dict("records")
    block = pd.cut(hour, [-1, 5, 11, 17, 23], labels=["h00-05", "h06-11", "h12-17", "h18-23"])
    ct = pd.crosstab(block, df["Class"])
    chi2, p_chi, dof, _ = stats.chi2_contingency(ct)
    out["time_blocks"] = {
        "crosstab": ct.reset_index().to_dict("records"),
        "chi2": float(chi2), "p": float(p_chi), "dof": int(dof),
        "fraud_rate_per_10k": (ct[1] / ct.sum(axis=1) * 1e4).to_dict(),
    }

    # --- 2d. Which anonymised features separate fraud from legit? ------------------
    rows = []
    for c_ in V_COLS:
        auc = roc_auc_score(y, df[c_])
        ks = stats.ks_2samp(df.loc[y == 1, c_], df.loc[y == 0, c_])
        rows.append({"feature": c_, "separation_auc": max(auc, 1 - auc),
                     "direction": "higher in fraud" if auc > 0.5 else "lower in fraud",
                     "ks_stat": float(ks.statistic)})
    sep = pd.DataFrame(rows).sort_values("separation_auc", ascending=False).reset_index(drop=True)
    out["feature_separation_top10"] = sep.head(10).to_dict("records")

    # ---- Figures ------------------------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    counts = df["Class"].value_counts().sort_index()
    ax[0].bar(["Legitimate", "Fraud"], counts.values, color=["#4C72B0", "#C44E52"])
    ax[0].set_yscale("log"); ax[0].set_ylabel("Transactions (log scale)")
    ax[0].set_title("Extreme class imbalance")
    for i, v in enumerate(counts.values):
        ax[0].text(i, v, f"{v:,}\n({v / len(df) * 100:.2f}%)", ha="center", va="bottom")
    ax[0].set_ylim(top=counts.max() * 8)
    bins_ = np.logspace(-2, np.log10(df["Amount"].max() + 1), 50)
    ax[1].hist(legit.clip(lower=0.01), bins=bins_, alpha=.6, density=True, label="Legitimate", color="#4C72B0")
    ax[1].hist(fraud.clip(lower=0.01), bins=bins_, alpha=.6, density=True, label="Fraud", color="#C44E52")
    ax[1].set_xscale("log"); ax[1].set_xlabel("Amount (log scale)"); ax[1].set_ylabel("Density")
    ax[1].set_title("Transaction amount distribution"); ax[1].legend()
    plt.tight_layout(); plt.savefig(fig_dir / "01_imbalance_and_amount.png", dpi=140); plt.close()

    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.bar(ht.index, ht["txns"], color="#9CB4D8", label="Transactions")
    ax.set_xlabel("Hour of day (relative to start of data)"); ax.set_ylabel("Transactions")
    ax2 = ax.twinx()
    ax2.plot(ht.index, ht["fraud_rate_per_10k"], color="#C44E52", marker="o", label="Fraud per 10k txns")
    ax2.set_ylabel("Frauds per 10,000 transactions")
    ax.set_title("Volume falls overnight, but fraud rate rises")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper left")
    plt.tight_layout(); plt.savefig(fig_dir / "02_fraud_by_hour.png", dpi=140); plt.close()

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(bt.index.astype(str), bt["fraud_rate_per_10k"], color="#C44E52")
    for i, (r, n) in enumerate(zip(bt["fraud_rate_per_10k"], bt["frauds"])):
        ax.text(i, r, f"{r:.1f}\n(n={n})", ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Frauds per 10,000 transactions"); ax.set_xlabel("Amount band")
    ax.set_title("Fraud rate by transaction amount band")
    ax.set_ylim(top=bt["fraud_rate_per_10k"].max() * 1.25)
    plt.tight_layout(); plt.savefig(fig_dir / "03_fraud_by_amount_band.png", dpi=140); plt.close()

    top = sep.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(top["feature"], top["separation_auc"], color="#55A868")
    ax.axvline(0.5, color="grey", ls="--"); ax.set_xlim(0.5, 1.0)
    ax.set_xlabel("Single-feature separation (AUC; 0.5 = no signal)")
    ax.set_title("Top 10 features separating fraud from legitimate")
    plt.tight_layout(); plt.savefig(fig_dir / "04_feature_separation.png", dpi=140); plt.close()

    return out


# %% [3] Feature engineering and splitting
def make_features(df):
    """V1..V28 + log(amount) + cyclical hour-of-day. Raw 'Time' is deliberately excluded:
    it is a running counter, so a model could not use it on future data."""
    X = df[V_COLS].copy()
    X["log_amount"] = np.log1p(df["Amount"])
    hr = (df["Time"] % 86400) / 3600.0
    X["hour_sin"] = np.sin(2 * np.pi * hr / 24)
    X["hour_cos"] = np.cos(2 * np.pi * hr / 24)
    return X


def chronological_split(df, train_frac, val_frac):
    n = len(df)
    i1, i2 = int(n * train_frac), int(n * (train_frac + val_frac))
    return df.iloc[:i1], df.iloc[i1:i2], df.iloc[i2:]


# %% [4] Metrics helpers
def recall_at_precision(y, s, min_precision):
    p, r, _ = precision_recall_curve(y, s)
    mask = p[:-1] >= min_precision
    return float(r[:-1][mask].max()) if mask.any() else 0.0


def evaluate_scores(y, s, min_precision):
    return {
        "pr_auc": float(average_precision_score(y, s)),
        "roc_auc": float(roc_auc_score(y, s)),
        f"recall_at_precision_{int(min_precision * 100)}": recall_at_precision(y, s, min_precision),
    }


def psi(expected, actual, bins=10):
    """Population Stability Index. <0.1 stable, 0.1-0.25 moderate shift, >0.25 major shift."""
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.clip(np.histogram(expected, edges)[0] / len(expected), 1e-6, None)
    a = np.clip(np.histogram(actual, edges)[0] / len(actual), 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


# %% [5] Cost model and thresholding
def net_savings(y, amount, s, thr, review_cost):
    """Net benefit of flagging every txn with score >= thr.
    Benefit  = value of fraud caught (loss prevented)
    Cost     = review_cost x number of alerts (every alert is manually reviewed)."""
    flag = s >= thr
    caught = amount[(y == 1) & flag].sum()
    return float(caught - review_cost * flag.sum())


def pick_threshold(y, amount, s, review_cost):
    cand = np.unique(np.quantile(s, np.linspace(0.90, 0.99995, 600)))
    vals = np.array([net_savings(y, amount, s, t, review_cost) for t in cand])
    best = int(np.flatnonzero(vals == vals.max())[-1])  # highest threshold among ties
    return float(cand[best]), cand, vals


def confusion_at(y, s, thr):
    flag = s >= thr
    tp = int(((y == 1) & flag).sum()); fp = int(((y == 0) & flag).sum())
    fn = int(((y == 1) & ~flag).sum()); tn = int(((y == 0) & ~flag).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": prec, "recall": rec,
            "alerts": tp + fp, "false_positive_rate": fp / (fp + tn)}


# %% [6] Main pipeline
def main(data_path, out_dir):
    t0 = time.time()
    cfg = CONFIG
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    model_dir = out_dir / "model"
    for d in (fig_dir, tab_dir, model_dir):
        d.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(cfg["seed"])
    results = {"config": cfg}

    # ---- 1. Load ---------------------------------------------------------------
    print("[1/8] Loading and validating data ...")
    df, quality = load_and_validate(data_path)
    results["data_quality"] = quality
    print(f"      {quality['rows_raw']:,} raw rows -> {quality['rows_clean']:,} after removing "
          f"{quality['duplicate_rows']:,} duplicates; fraud rate {quality['fraud_rate_clean_pct']:.3f}%")

    # ---- 2. EDA ----------------------------------------------------------------
    print("[2/8] Statistical EDA ...")
    results["eda"] = run_eda(df, fig_dir)

    # ---- 3. Split --------------------------------------------------------------
    print("[3/8] Chronological split ...")
    tr, va, te = chronological_split(df, cfg["train_frac"], cfg["val_frac"])
    Xtr, Xva, Xte = make_features(tr), make_features(va), make_features(te)
    ytr, yva, yte = tr["Class"].values, va["Class"].values, te["Class"].values
    ava, ate = va["Amount"].values, te["Amount"].values
    results["split"] = {
        name: {"rows": int(len(part)), "frauds": int(part["Class"].sum()),
               "hours": [float(part["Time"].min() / 3600), float(part["Time"].max() / 3600)]}
        for name, part in (("train", tr), ("validation", va), ("test", te))
    }
    if min(yva.sum(), yte.sum()) == 0:
        raise ValueError("Validation/test split contains no fraud cases.")

    # ---- 4. Models -------------------------------------------------------------
    print("[4/8] Training models (imbalance handled with class weights) ...")
    models = {
        "Baseline (prevalence)": DummyClassifier(strategy="prior"),
        "Logistic Regression": Pipeline([
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(class_weight="balanced", max_iter=3000, C=0.5))]),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=14, min_samples_leaf=3, max_features="sqrt",
            class_weight="balanced_subsample", n_jobs=-1, random_state=cfg["seed"]),
        "Gradient Boosting": HistGradientBoostingClassifier(
            max_iter=250, learning_rate=0.05, max_leaf_nodes=31, l2_regularization=1.0,
            class_weight="balanced", early_stopping=False, random_state=cfg["seed"]),
    }
    val_scores, test_scores, comp_rows = {}, {}, []
    for name, m in models.items():
        t1 = time.time()
        m.fit(Xtr, ytr)
        val_scores[name] = m.predict_proba(Xva)[:, 1]
        test_scores[name] = m.predict_proba(Xte)[:, 1]
        mv = evaluate_scores(yva, val_scores[name], cfg["min_precision_target"])
        mt = evaluate_scores(yte, test_scores[name], cfg["min_precision_target"])
        comp_rows.append({"model": name,
                          **{f"val_{k}": v for k, v in mv.items()},
                          **{f"test_{k}": v for k, v in mt.items()},
                          "fit_seconds": time.time() - t1})
        print(f"      {name:<24} val PR-AUC={mv['pr_auc']:.3f} | test PR-AUC={mt['pr_auc']:.3f}")
    comp = pd.DataFrame(comp_rows)
    comp.to_csv(tab_dir / "model_comparison.csv", index=False)
    results["model_comparison"] = comp.to_dict("records")

    # Model selection uses VALIDATION only (never peeks at test).
    cand = comp[comp["model"] != "Baseline (prevalence)"]
    best_name = cand.loc[cand["val_pr_auc"].idxmax(), "model"]
    best = models[best_name]
    results["best_model"] = best_name
    print(f"      -> selected on validation PR-AUC: {best_name}")

    # ---- 5. Threshold (validation) -> apply on test ------------------------------
    print("[5/8] Cost-based threshold selection ...")
    thr, cand_thr, cand_val = pick_threshold(yva, ava, val_scores[best_name], cfg["review_cost"])
    s_va, s_te = val_scores[best_name], test_scores[best_name]
    cm_va, cm_te = confusion_at(yva, s_va, thr), confusion_at(yte, s_te, thr)
    fraud_value_te = float(ate[yte == 1].sum())
    caught_te = float(ate[(yte == 1) & (s_te >= thr)].sum())
    saving_te = net_savings(yte, ate, s_te, thr, cfg["review_cost"])
    results["operating_point"] = {
        "threshold": thr, "validation": cm_va, "test": cm_te,
        "test_fraud_value_total": fraud_value_te,
        "test_fraud_value_caught": caught_te,
        "test_value_caught_pct": caught_te / fraud_value_te * 100,
        "test_review_cost": cfg["review_cost"] * cm_te["alerts"],
        "test_net_savings": saving_te,
        "test_alerts_per_10k_txns": cm_te["alerts"] / len(yte) * 1e4,
        "test_missed_fraud_value": fraud_value_te - caught_te,
    }

    # ---- 6. Uncertainty: bootstrap on test --------------------------------------
    print("[6/8] Bootstrap confidence intervals ...")
    boot = {n: [] for n in models if n != "Baseline (prevalence)"}
    boot_rec, boot_prec = [], []
    idx_all = np.arange(len(yte))
    for _ in range(cfg["n_bootstrap"]):
        ix = rng.choice(idx_all, size=len(idx_all), replace=True)
        if yte[ix].sum() == 0:
            continue
        for n in boot:
            boot[n].append(average_precision_score(yte[ix], test_scores[n][ix]))
        c = confusion_at(yte[ix], s_te[ix], thr)
        boot_rec.append(c["recall"]); boot_prec.append(c["precision"])
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    results["bootstrap_95ci"] = {
        "pr_auc": {n: ci(v) for n, v in boot.items()},
        "best_model_recall": ci(boot_rec), "best_model_precision": ci(boot_prec),
    }

    # ---- 7. Business views ------------------------------------------------------
    print("[7/8] Business analysis (alert budgets, importance, drift) ...")
    order = np.argsort(-s_te)
    total_fraud, total_value = yte.sum(), fraud_value_te
    gain_rows = []
    for b in cfg["alert_budgets"]:
        k = max(int(round(b * len(yte))), 1)
        top = order[:k]
        tp = int(yte[top].sum())
        gain_rows.append({
            "review_budget_pct": b * 100, "alerts": k, "frauds_caught": tp,
            "recall_pct": tp / total_fraud * 100, "precision_pct": tp / k * 100,
            "fraud_value_caught_pct": float(ate[top][yte[top] == 1].sum() / total_value * 100),
            "lift_vs_random": (tp / k) / (total_fraud / len(yte)),
        })
    gain = pd.DataFrame(gain_rows)
    gain.to_csv(tab_dir / "alert_budget_gain_table.csv", index=False)
    results["alert_budget"] = gain.to_dict("records")

    # Permutation importance on a rebalanced sample (all frauds + 10k legit) for speed
    legit_ix = rng.choice(np.flatnonzero(yte == 0), size=min(10000, int((yte == 0).sum())), replace=False)
    samp = np.sort(np.concatenate([np.flatnonzero(yte == 1), legit_ix]))
    pi = permutation_importance(best, Xte.iloc[samp], yte[samp], scoring="average_precision",
                                n_repeats=3, random_state=cfg["seed"], n_jobs=1)
    imp = pd.DataFrame({"feature": Xte.columns, "importance": pi.importances_mean,
                        "std": pi.importances_std}).sort_values("importance", ascending=False)
    imp.to_csv(tab_dir / "permutation_importance.csv", index=False)
    results["top_features_by_permutation_importance"] = imp.head(10).to_dict("records")

    # Drift: train vs test
    drift = [{"feature": c, "psi": psi(Xtr[c].values, Xte[c].values)} for c in Xtr.columns]
    drift = pd.DataFrame(drift).sort_values("psi", ascending=False)
    drift.to_csv(tab_dir / "drift_psi_train_vs_test.csv", index=False)
    results["drift"] = {
        "score_psi_val_vs_test": psi(s_va, s_te),
        "features_psi_gt_0.1": drift[drift["psi"] > 0.1]["feature"].tolist(),
        "max_feature_psi": drift.head(3).to_dict("records"),
    }

    # Where do the misses come from?  (false negatives on test)
    fn_mask = (yte == 1) & (s_te < thr)
    results["missed_frauds_test"] = {
        "count": int(fn_mask.sum()),
        "median_amount": float(np.median(ate[fn_mask])) if fn_mask.any() else None,
        "share_amount_le_10": float((ate[fn_mask] <= 10).mean()) if fn_mask.any() else None,
        "caught_median_amount": float(np.median(ate[(yte == 1) & (s_te >= thr)]))
        if ((yte == 1) & (s_te >= thr)).any() else None,
    }

    # ---- Figures: modelling -----------------------------------------------------
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    for n in models:
        if n.startswith("Baseline"):
            continue
        p, r, _ = precision_recall_curve(yte, test_scores[n])
        ax[0].plot(r, p, label=f"{n} (AP={average_precision_score(yte, test_scores[n]):.2f})")
    ax[0].axhline(yte.mean(), color="grey", ls="--", label=f"Baseline ({yte.mean():.4f})")
    ax[0].set_xlabel("Recall"); ax[0].set_ylabel("Precision")
    ax[0].set_title("Precision-Recall curves (test set)"); ax[0].legend(loc="lower left", fontsize=8)
    names = [n for n in boot]
    data = [boot[n] for n in names]
    ax[1].boxplot(data, showmeans=True)
    ax[1].set_xticklabels([n.replace(" ", "\n") for n in names])
    ax[1].set_ylabel("PR-AUC (bootstrap)"); ax[1].set_title("Uncertainty: 300 bootstrap resamples of test set")
    plt.tight_layout(); plt.savefig(fig_dir / "05_model_comparison.png", dpi=140); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    ax[0].plot(cand_thr, cand_val, color="#4C72B0")
    ax[0].axvline(thr, color="#C44E52", ls="--", label=f"Chosen threshold = {thr:.3f}")
    ax[0].set_xlabel("Score threshold"); ax[0].set_ylabel("Net savings on validation (currency units)")
    ax[0].set_title("Choosing the alert threshold by business cost"); ax[0].legend()
    cmx = np.array([[cm_te["tn"], cm_te["fp"]], [cm_te["fn"], cm_te["tp"]]])
    ax[1].imshow(cmx, cmap="Blues")
    for (i, j), v in np.ndenumerate(cmx):
        ax[1].text(j, i, f"{v:,}", ha="center", va="center",
                   color="white" if v > cmx.max() / 2 else "black", fontsize=13)
    ax[1].set_xticks([0, 1], ["Predicted\nlegit", "Predicted\nfraud"])
    ax[1].set_yticks([0, 1], ["Actual\nlegit", "Actual\nfraud"])
    ax[1].set_title("Test-set confusion matrix at chosen threshold")
    plt.tight_layout(); plt.savefig(fig_dir / "06_threshold_and_confusion.png", dpi=140); plt.close()

    ks = np.arange(1, len(yte) + 1)
    cum = np.cumsum(yte[order]) / total_fraud
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    ax[0].plot(ks / len(yte) * 100, cum * 100, color="#C44E52", label="Model")
    ax[0].plot([0, 100], [0, 100], color="grey", ls="--", label="Random review")
    ax[0].set_xlim(0, 5); ax[0].set_ylim(0, 100)
    ax[0].set_xlabel("% of transactions sent for review (highest risk first)")
    ax[0].set_ylabel("% of all frauds caught"); ax[0].set_title("Gain curve (zoomed to top 5%)")
    ax[0].legend()
    ax[1].bar(gain["review_budget_pct"].map(lambda v: f"{v:.2f}%"), gain["fraud_value_caught_pct"], color="#55A868")
    ax[1].set_xlabel("Review budget (% of transactions)"); ax[1].set_ylabel("% of fraud VALUE caught")
    ax[1].set_title("Fraud value protected vs. review workload")
    plt.tight_layout(); plt.savefig(fig_dir / "07_alert_budget.png", dpi=140); plt.close()

    topi = imp.head(10).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(topi["feature"], topi["importance"], xerr=topi["std"], color="#8172B2")
    ax.set_xlabel("Drop in PR-AUC when feature is shuffled")
    ax.set_title(f"Permutation importance ({best_name})")
    plt.tight_layout(); plt.savefig(fig_dir / "08_feature_importance.png", dpi=140); plt.close()

    # ---- 8. Save artefacts ------------------------------------------------------
    print("[8/8] Saving model and results ...")
    joblib.dump({"model": best, "threshold": thr, "features": list(Xtr.columns)},
                model_dir / "fraud_model.joblib")
    results["runtime_seconds"] = time.time() - t0
    with open(out_dir / "results.json", "w") as f:
        json.dump(jsonable(results), f, indent=2)
    print(f"Done in {time.time() - t0:.0f}s. Outputs in: {out_dir.resolve()}")
    return results


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/creditcard.csv")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()
    main(args.data, args.out)