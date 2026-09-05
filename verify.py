"""
Bonus / optional script (not required by the assignment, but the prompt
mentions "Additional code / maths used to estimate / extract the variables
will be a plus" — this covers that).

What it does:
1. Re-runs the fit from fit_curve.py to get theta, M, X.
2. Computes an L1-distance metric similar to what the grading criteria
   describes ("L1 distance between uniformly sampled points between
   expected and predicted curve"). NOTE: we don't have the grader's true
   (expected) theta/M/X, so as a stand-in we treat the provided xy_data.csv
   points as ground truth and measure how far our fitted curve is from them
   using the same style of metric (uniform sampling + nearest point L1
   distance). This is a self-check, not the actual grading number.
3. Saves fit_plot.png overlaying the fitted curve on the data scatter, for
   a quick visual sanity check to include in the README.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

from fit_curve import model_pts, fit  # reuse the exact fitting logic

DATA_PATH = "xy_data.csv"
N_UNIFORM = 2000  # uniform samples along t for the L1 metric


def l1_curve_to_points(params, xdata, ydata, n_samples=N_UNIFORM):
    """L1 distance from each data point to its nearest point on the
    uniformly-t-sampled fitted curve, summed (and averaged)."""
    t = np.linspace(6, 60, n_samples)
    xm, ym = model_pts(params, t)
    tree = cKDTree(np.column_stack([xm, ym]))
    dist, _ = tree.query(np.column_stack([xdata, ydata]), p=1)  # L1 (Manhattan)
    return dist.sum(), dist.mean(), dist.max()


def main():
    df = pd.read_csv(DATA_PATH)
    x, y = df["x"].values, df["y"].values

    rng = np.random.default_rng(0)
    best = None
    for _ in range(6):
        init = [rng.uniform(0, np.radians(50)), rng.uniform(-0.05, 0.05), rng.uniform(0, 100)]
        params, dist = fit(x, y, init)
        cost = np.mean(dist ** 2)
        if best is None or cost < best[0]:
            best = (cost, params)
    _, params = best
    theta, M, X = params

    total_l1, mean_l1, max_l1 = l1_curve_to_points(params, x, y)
    print(f"theta = {np.degrees(theta):.4f} deg, M = {M:.6f}, X = {X:.6f}")
    print(f"L1 (proxy, data vs fitted curve): total={total_l1:.4f}  "
          f"mean={mean_l1:.6f}  max={max_l1:.6f}")

    # --- plot ---
    t = np.linspace(6, 60, N_UNIFORM)
    xm, ym = model_pts(params, t)
    plt.figure(figsize=(7, 6))
    plt.scatter(x, y, s=6, alpha=0.4, label="given data (xy_data.csv)")
    plt.plot(xm, ym, color="red", linewidth=1.5, label="fitted curve")
    plt.title(f"Fitted curve: θ={np.degrees(theta):.2f}°, M={M:.4f}, X={X:.2f}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.tight_layout()
    plt.savefig("fit_plot.png", dpi=150)
    print("Saved fit_plot.png")


if __name__ == "__main__":
    main()
