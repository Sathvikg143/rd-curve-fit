# R&D Assignment — Parametric Curve Fit

## Answer

| Variable | Value |
|---|---|
| θ | **30°** (π/6 rad = 0.523599) |
| M | **0.03** |
| X | **55** |

**Desmos-format string:**
```
\left(t*\cos(0.5236)-e^{0.03\left|t\right|}\cdot\sin(0.3t)\sin(0.5236)+55,42+t*\sin(0.5236)+e^{0.03\left|t\right|}\cdot\sin(0.3t)\cos(0.5236)\right)
```
Domain: `6 ≤ t ≤ 60`

Fit quality: mean point-to-curve distance ≈ 0.0003 (i.e. residual is just numerical grid noise — an effectively exact match).

## Problem
`xy_data.csv` gives 1500 `(x, y)` points sampled from the curve for `6 < t < 60`, but the rows are **not** ordered by `t` (row order ≠ curve order) and no `t` value is given per row. So this isn't a simple curve_fit — the correspondence between each data point and its `t` is itself unknown and has to be recovered.

## Approach
Treated it as an **Iterative Closest Point (ICP) / orthogonal-distance regression** problem, since we effectively have two unknowns per data point (its latent `t`) plus the three global unknowns (θ, M, X):

1. **Initialize** θ, M, X randomly within the given bounds.
2. **Assign t's:** densely sample the candidate curve (50k points over t ∈ [6,60]), build a KD-tree, and for every data point find its nearest point on the curve. That nearest point's `t` becomes the point's estimated `t`.
3. **Refit parameters:** with those per-point `t` estimates now fixed, run bounded nonlinear least squares (`scipy.optimize.least_squares`) to re-solve for θ, M, X that minimize the sum of squared (x,y) residuals.
4. **Repeat** steps 2–3 until convergence (curve stops moving, ~15-25 iterations).
5. **Random restarts** (6 different random initializations) to avoid local minima — all converged to the same basin, confirming the solution: θ=30°, M=0.03, X=55.

## Files
- `fit_curve.py` — full working script (reproduces the answer)
- `xy_data.csv` — original data, copied for convenience

## Run it
```bash
pip install numpy pandas scipy
python fit_curve.py
```
