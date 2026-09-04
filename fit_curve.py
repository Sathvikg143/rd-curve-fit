"""
Fits theta, M, X for the parametric curve:
    x(t) = t*cos(theta) - e^(M|t|) * sin(0.3t) * sin(theta) + X
    y(t) = 42 + t*sin(theta) + e^(M|t|) * sin(0.3t) * cos(theta)
to the unlabeled point cloud in xy_data.csv (points are NOT ordered by t).

Approach: since each row's t-value is unknown and the rows are shuffled,
this is solved as an Iterative Closest Point (ICP) / orthogonal-distance
regression problem:
  1. Given current (theta, M, X), densely sample the curve over t in [6,60]
     and build a KD-tree of curve points.
  2. For every data point, find its nearest point on the curve -> this
     recovers an estimate of that point's t.
  3. With those t estimates fixed, re-fit (theta, M, X) via nonlinear
     least squares (scipy.optimize.least_squares, bounded to the given ranges).
  4. Repeat steps 1-3 until convergence.
Multiple random restarts are used to avoid local minima.
"""
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.spatial import cKDTree

DATA_PATH = "xy_data.csv"
BOUNDS = ([0.0, -0.05, 0.0], [np.radians(50), 0.05, 100.0])  # theta(rad), M, X


def model_pts(params, t):
    theta, M, X = params
    ex = np.exp(M * np.abs(t))
    xm = t * np.cos(theta) - ex * np.sin(0.3 * t) * np.sin(theta) + X
    ym = 42 + t * np.sin(theta) + ex * np.sin(0.3 * t) * np.cos(theta)
    return xm, ym


def nearest_t(params, xdata, ydata, n_samples=50000):
    t = np.linspace(6, 60, n_samples)
    xm, ym = model_pts(params, t)
    tree = cKDTree(np.column_stack([xm, ym]))
    dist, idx = tree.query(np.column_stack([xdata, ydata]))
    return t[idx], dist


def resid(params, tvals, xdata, ydata):
    xm, ym = model_pts(params, tvals)
    return np.concatenate([xm - xdata, ym - ydata])


def fit(xdata, ydata, init, n_iter=25):
    params = np.array(init, dtype=float)
    for _ in range(n_iter):
        tvals, _ = nearest_t(params, xdata, ydata)
        params = least_squares(
            resid, params, args=(tvals, xdata, ydata),
            bounds=BOUNDS, xtol=1e-15, ftol=1e-15
        ).x
    tvals, dist = nearest_t(params, xdata, ydata)
    return params, dist


def main():
    df = pd.read_csv(DATA_PATH)
    x, y = df["x"].values, df["y"].values

    rng = np.random.default_rng(0)
    best = None
    for _ in range(6):  # random restarts
        init = [rng.uniform(0, np.radians(50)), rng.uniform(-0.05, 0.05), rng.uniform(0, 100)]
        params, dist = fit(x, y, init)
        cost = np.mean(dist ** 2)
        if best is None or cost < best[0]:
            best = (cost, params)

    _, (theta, M, X) = best
    print(f"theta = {np.degrees(theta):.4f} deg  ({theta:.6f} rad)")
    print(f"M     = {M:.6f}")
    print(f"X     = {X:.6f}")


if __name__ == "__main__":
    main()
