"""
Project 2: Regression and Regularized Regression via statsmodels.

Rubric Section 1 (Regression): full multiple OLS regression y ~ all predictors
for all 3 UCI datasets: Auto MPG, Concrete Compressive Strength, Airfoil Self-Noise.

Rubric Section 2 (Regularized Regression): Ridge (L2) and Lasso (L1) via
statsmodels' OLS.fit_regularized, AutoMPG only. The penalty weight (alpha) for
each is chosen by 5-fold cross-validated MSE over a small grid, then the final
model is refit on the full dataset at the chosen alpha.

Outputs are written to project 2/results/<dataset>/:
  - ols_summary_regression.txt         full OLS summary (Section 1)
  - fit_vs_actual_regression.png       predicted vs. actual scatter (Section 1)
  - ridge_summary.txt / lasso_summary.txt   coefficients + CV/fit diagnostics (Section 2, AutoMPG only)
  - ridge_alpha_search.csv / lasso_alpha_search.csv   CV grid search results (Section 2, AutoMPG only)
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.model_selection import KFold

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
RESULTS_DIR = os.path.join(BASE, "results")

DATASETS = [
    {"name": "auto_mpg", "csv": "auto_mpg.csv", "target": "mpg"},
    {"name": "concrete", "csv": "concrete.csv", "target": "concrete_compressive_strength"},
    {"name": "airfoil", "csv": "airfoil.csv", "target": "scaled_sound_pressure_level"},
]

ALPHAS = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]


def section1_regression(cfg, out_dir):
    """Full multiple OLS regression on all predictors."""
    df = pd.read_csv(os.path.join(DATA_DIR, cfg["csv"]))
    y = df[cfg["target"]].values
    X = df.drop(columns=[cfg["target"]])
    fname = list(X.columns)
    Xc = sm.add_constant(X.values)

    model = sm.OLS(y, Xc).fit()
    with open(os.path.join(out_dir, "ols_summary_regression.txt"), "w") as f:
        f.write(f"Regression: {cfg['target']} ~ {' + '.join(fname)}\n\n")
        f.write(str(model.summary()))

    yhat = model.predict(Xc)
    plt.figure(figsize=(6, 6))
    plt.scatter(y, yhat, s=12, alpha=0.6)
    lims = [min(y.min(), yhat.min()), max(y.max(), yhat.max())]
    plt.plot(lims, lims, "r--", linewidth=1.5)
    plt.xlabel(f"actual {cfg['target']}")
    plt.ylabel(f"predicted {cfg['target']}")
    plt.title(f"{cfg['name']}: OLS Regression, predicted vs. actual (statsmodels)")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "fit_vs_actual_regression.png"), dpi=150)
    plt.close()

    print(f"  [Section 1] {cfg['name']}: R^2={model.rsquared:.4f}, "
          f"adjR^2={model.rsquared_adj:.4f}, AIC={model.aic:.2f}")
    return fname


def cv_mse(X, y, alpha, L1_wt, n_splits=5, seed=0):
    """5-fold CV MSE for OLS.fit_regularized(alpha, L1_wt) on standardized X."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    mses = []
    for train_idx, test_idx in kf.split(X):
        Xtr, Xte = X[train_idx], X[test_idx]
        ytr, yte = y[train_idx], y[test_idx]
        mu, sd = Xtr.mean(axis=0), Xtr.std(axis=0)
        sd[sd == 0] = 1.0
        Xtr_s = (Xtr - mu) / sd
        Xte_s = (Xte - mu) / sd
        res = sm.OLS(ytr, Xtr_s).fit_regularized(alpha=alpha, L1_wt=L1_wt)
        yhat = Xte_s @ res.params
        mses.append(np.mean((yte - yhat) ** 2))
    return float(np.mean(mses))


def section2_regularized(cfg, out_dir, fname):
    """Ridge (L1_wt=0) and Lasso (L1_wt=1) via statsmodels fit_regularized, AutoMPG only."""
    df = pd.read_csv(os.path.join(DATA_DIR, cfg["csv"]))
    y = df[cfg["target"]].values
    X = df.drop(columns=[cfg["target"]]).values
    mu, sd = X.mean(axis=0), X.std(axis=0)
    Xs = (X - mu) / sd
    yc = y - y.mean()

    for label, l1_wt in [("ridge", 0.0), ("lasso", 1.0)]:
        rows = []
        best_alpha, best_mse = None, np.inf
        for a in ALPHAS:
            mse = cv_mse(X, y, a, l1_wt)
            rows.append({"alpha": a, "cv_mse": mse})
            if mse < best_mse:
                best_mse, best_alpha = mse, a
        pd.DataFrame(rows).to_csv(os.path.join(out_dir, f"{label}_alpha_search.csv"), index=False)

        res = sm.OLS(yc, Xs).fit_regularized(alpha=best_alpha, L1_wt=l1_wt)
        yhat = Xs @ res.params + y.mean()
        sse = float(np.sum((y - yhat) ** 2))
        sst = float(np.sum((y - y.mean()) ** 2))
        r2 = 1 - sse / sst

        with open(os.path.join(out_dir, f"{label}_summary.txt"), "w") as f:
            f.write(f"{label.title()} Regression (statsmodels OLS.fit_regularized): "
                    f"{cfg['target']} ~ {' + '.join(fname)}\n")
            f.write(f"L1_wt = {l1_wt} ({'Lasso/L1' if l1_wt == 1.0 else 'Ridge/L2'})\n")
            f.write(f"alpha chosen by 5-fold CV over {ALPHAS} -> alpha = {best_alpha} "
                    f"(cv_mse = {best_mse:.4f})\n\n")
            f.write("Standardized coefficients (x centered/scaled, y centered):\n")
            for nm, b in zip(fname, res.params):
                f.write(f"  {nm:>20s}: {b: .6f}\n")
            f.write(f"\nIn-sample (full-data) fit at chosen alpha:\n")
            f.write(f"  SSE = {sse:.4f}\n  R^2 = {r2:.4f}\n")

        print(f"  [Section 2] {cfg['name']} {label}: best alpha={best_alpha}, "
              f"cv_mse={best_mse:.4f}, in-sample R^2={r2:.4f}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for cfg in DATASETS:
        out_dir = os.path.join(RESULTS_DIR, cfg["name"])
        os.makedirs(out_dir, exist_ok=True)
        print(f"=== {cfg['name']} ===")
        fname = section1_regression(cfg, out_dir)
        if cfg["name"] == "auto_mpg":
            section2_regularized(cfg, out_dir, fname)

    print("\nAll statsmodels results written under", RESULTS_DIR)


if __name__ == "__main__":
    main()
