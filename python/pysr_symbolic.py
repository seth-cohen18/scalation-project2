"""
Project 2, Rubric Section 5: Symbolic Regression via PySR.

Runs PySR on all 3 UCI datasets: Auto MPG, Concrete Compressive Strength,
Airfoil Self-Noise, searching over a small library of operators for a compact
closed-form expression that predicts the target from the predictors.

Outputs are written to project 2/results/<dataset>/:
  - pysr_equations.csv        the Pareto front of (complexity, loss, equation)
  - pysr_summary.txt          best equation, its score/loss/R^2, and predictor mapping
  - pysr_fit_vs_actual.png    predicted vs. actual scatter for the best equation
"""

import platform
platform.machine = lambda: "AMD64"          # this Windows-on-ARM64 machine has no native
                                             # Julia build; force resolution of the x64
                                             # build, which runs fine under Windows' x64 emulation

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pysr import PySRRegressor

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE, "data")
RESULTS_DIR = os.path.join(BASE, "results")

DATASETS = [
    {"name": "auto_mpg", "csv": "auto_mpg.csv", "target": "mpg"},
    {"name": "concrete", "csv": "concrete.csv", "target": "concrete_compressive_strength"},
    {"name": "airfoil", "csv": "airfoil.csv", "target": "scaled_sound_pressure_level"},
]


def run_one(cfg):
    out_dir = os.path.join(RESULTS_DIR, cfg["name"])
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv(os.path.join(DATA_DIR, cfg["csv"]))
    y = df[cfg["target"]].values
    X = df.drop(columns=[cfg["target"]])
    fname = list(X.columns)

    print(f"=== {cfg['name']}: PySR symbolic regression, n={len(df)}, "
          f"predictors={fname} ===")

    model = PySRRegressor(
        niterations=60,
        populations=24,
        population_size=33,
        maxsize=24,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["square", "sqrt", "exp", "log"],
        model_selection="best",
        loss="loss(x, y) = (x - y)^2",
        random_state=0,
        deterministic=True,
        parallelism="serial",
        temp_equation_file=True,
        verbosity=1,
        progress=False,
    )
    model.fit(X.values, y, variable_names=fname)

    eq_df = model.equations_[["complexity", "loss", "score", "equation"]]
    eq_df.to_csv(os.path.join(out_dir, "pysr_equations.csv"), index=False)

    yhat = model.predict(X.values)
    sse = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1 - sse / sst
    best = model.get_best()

    with open(os.path.join(out_dir, "pysr_summary.txt"), "w") as f:
        f.write(f"PySR Symbolic Regression: {cfg['target']} ~ f({', '.join(fname)})\n\n")
        f.write(f"Best equation (by PySR's default complexity/loss trade-off score):\n")
        f.write(f"  {cfg['target']} = {best['equation']}\n\n")
        f.write(f"  complexity = {best['complexity']}, training loss (MSE) = {best['loss']:.6f}\n")
        f.write(f"  in-sample R^2 = {r2:.4f}\n\n")
        f.write("Pareto front (complexity, loss, equation):\n")
        f.write(eq_df.to_string(index=False))

    plt.figure(figsize=(6, 6))
    plt.scatter(y, yhat, s=12, alpha=0.6)
    lims = [min(y.min(), yhat.min()), max(y.max(), yhat.max())]
    plt.plot(lims, lims, "r--", linewidth=1.5)
    plt.xlabel(f"actual {cfg['target']}")
    plt.ylabel(f"predicted {cfg['target']}")
    plt.title(f"{cfg['name']}: PySR best equation, predicted vs. actual")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "pysr_fit_vs_actual.png"), dpi=150)
    plt.close()

    print(f"  best equation: {cfg['target']} = {best['equation']}")
    print(f"  complexity={best['complexity']}, loss={best['loss']:.6f}, R^2={r2:.4f}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for cfg in DATASETS:
        run_one(cfg)
    print("\nAll PySR results written under", RESULTS_DIR)


if __name__ == "__main__":
    main()
