---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# <i class="fa-solid fa-wave-square"></i> Polynomial and Flexible Regression

Let us have a quick recap of the first session of the semester, where regression models were introduced. In linear regression, we assume that the relationship between a predictor $x$ and a response $y$ is a straight line:

$$
y \approx \beta_0 + \beta_1 x.
$$

In many real-world problems, this assumption is too restrictive. The true relationship may be curved, bend differently in different regions of $x$, or vary smoothly but non-linearly.

In this chapter, we look at four families of methods that allow more flexible regression functions:

- Polynomial regression
- Stepwise regression
- Spline regression
- Local regression

Let's start by creating some example data. Imagine we have one predictor $x$ and a response $y$ generated as:

$$
y = \sin(x) + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2).
$$


```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

n = 200
X = np.linspace(0, 10, n)
eps = rng.normal(scale=0.4, size=n)
y = np.sin(X) + 0.3 * X + eps
data = pd.DataFrame({"x": X, "y": y})

fig, ax = plt.subplots()
ax.scatter(data["x"], data["y"], alpha=0.5)
ax.set(xlabel="x", ylabel="y");
```

If we fit a straight line to such data, the model will clearly miss the oscillating pattern. Flexible regression methods aim to recover such non-linear patterns while still being based on the same least squares framework.

---

## Polynomial Regression

Polynomial regression extends the linear model by including powers of $x$ as additional predictors:

* **Degree 1** (ordinary linear regression):

  $$
  y \approx \beta_0 + \beta_1 x
  $$

* **Degree 2** (quadratic):

  $$
  y \approx \beta_0 + \beta_1 x + \beta_2 x^2
  $$

* **Degree $d$**:

  $$
  y \approx \beta_0 + \beta_1 x + \beta_2 x^2 + \dots + \beta_d x^d.
  $$

```{admonition}
:class: note 

Polynomial regression models are still **linear models in the parameters** $(\beta_0, \dots, \beta_d)$ – we just feed them transformed inputs $(x, x^2, \dots, x^d)$.
```

Pros:

* Simple and easy to fit with ordinary least squares
* Can capture smooth non-linear trends with low-degree polynomials

Cons:

* High-degree polynomials can oscillate wildly (especially at the boundaries)
* Global basis: changing the fit in one region can affect the entire curve

---

We can conveniently generate polynomial features using `PolynomialFeatures` from `sklearn.preprocessing` and then fit a standard `LinearRegression` model.

```{code-cell} ipython3
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

def fit_poly_regression(x, y, degree):
    """Fit a polynomial regression of given degree and return the model + grid predictions."""
    x = np.asarray(x).reshape(-1, 1)
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(x)

    model = LinearRegression()
    model.fit(X_poly, y)

    # For visualisation: predictions on a dense grid
    x_grid = np.linspace(x.min(), x.max(), 300).reshape(-1, 1)
    X_grid_poly = poly.transform(x_grid)
    y_hat = model.predict(X_grid_poly)

    return model, x_grid.ravel(), y_hat

degrees = [1, 2, 4, 16]

fig, ax = plt.subplots()
ax.scatter(data["x"], data["y"], alpha=0.3, label="Data")

for d in degrees:
    _, xg, yg = fit_poly_regression(data["x"], data["y"], degree=d)
    ax.plot(xg, yg, label=f"Degree {d}")
ax.set(xlabel="x", ylabel="y", title="Polynomial regression with different degrees")
plt.legend();
```

Things to observe:

* Degree 1 (straight line) cannot follow the curvature.
* Degree 2 or 3 often works well for gently curved relationships.
* Higher degree models can start to wiggle and overfit the noise.

---

## Stepwise Functions (0th-Order Splines)

The simple idea:

> Divide the range of $x$ into intervals using **cut points** (knots) and fit a **constant mean** within each interval.

Concretely, choose cut points $c_1 < c_2 < \dots < c_K$ and define indicator variables

$$
I_1(x) = \mathbf{1}(x \le c_1), \quad
I_2(x) = \mathbf{1}(c_1 < x \le c_2), \dots,
I_{K+1}(x) = \mathbf{1}(x > c_K).
$$

Then the model is

$$
y \approx \beta_0 + \beta_1 I_1(x) + \dots + \beta_{K+1} I_{K+1}(x),
$$

which produces a **piecewise constant** (stepwise) regression function.

This is equivalent to a **B-spline of degree 0** (a “zero-order spline”).

Pros:

* Very simple and interpretable: one mean per age/`x`-interval
* Useful as a starting point to understand more flexible splines

Cons:

* The fit is discontinuous at the cut points
* Sensitive to the choice and placement of cut points
* Can look rough; higher-order splines often give smoother fits

We can build a 0th-order spline (step function) basis using `patsy.dmatrix` with `bs(..., degree=0)` and then fit a linear model with `statsmodels`:

```{code-cell} ipython3
import patsy
import statsmodels.api as sm

# Choose some cut points (knots) over the x-range
cut_points = (2, 4, 6, 8)

# Build a B-spline basis of degree 0 (step function)
transformed_x = patsy.dmatrix(
    "bs(x, knots=cut_points, degree=0, include_intercept=False)",
    {"x": data["x"], "cut_points": cut_points},
    return_type="dataframe",
)
```

Fit the model:

```{code-cell} ipython3
step_model = sm.OLS(data["y"], transformed_x)
step_fit = step_model.fit()

print(step_fit.summary())
```

Visualise the stepwise fit:

```{code-cell} ipython3
xp = np.linspace(data["x"].min(), data["x"].max(), 300)
xp_trans = patsy.dmatrix(
    "bs(xp, knots=cut_points, degree=0, include_intercept=False)",
    {"xp": xp, "cut_points": cut_points},
    return_type="dataframe",
)

pred_step = step_fit.predict(xp_trans)

plt.figure(figsize=(10, 6))
plt.scatter(data["x"], data["y"], alpha=0.4, label="Data")
plt.plot(xp, pred_step, color="red", label="Stepwise fit (degree 0)")
for c in cut_points:
    plt.axvline(c, color="black", linestyle="--", alpha=0.6)

plt.xlabel("x")
plt.ylabel("y")
plt.title("Stepwise regression (zero-order spline)")
plt.legend();
```

---

## Spline Regression

The previous section introduced "stepwise regression" as 0th-order splines. However, when we talk about spline regression, we usually mean higher-order splines, which will smooth these steps into continuous and differentiable curves. Again, the key idea is the similar:

> Approximate the regression function by **piecewise polynomials** that are smoothly joined at pre-defined points called **knots**.

For example, a cubic spline with knots at $t_1, \dots, t_K$ is a function that is:

* A cubic polynomial on each interval $(-\infty, t_1)$, $(t_1, t_2)$, …, $(t_K, \infty)$
* Continuously differentiable up to a certain order at the knots

We usually do not work with the piecewise form directly. Instead, we represent splines as a linear combination of **basis functions**:

$$
f(x) = \sum_{j=1}^{M} \theta_j B_j(x),
$$

where $B_j(x)$ are spline basis functions (B-splines). This again gives a linear model in the parameters $\theta_j$.

Two common options:

* **B-splines** (flexible general basis)
* **Natural splines** (add boundary constraints to avoid wild extrapolation)

Again, we use the convenient `bs()` function from `patsy` to create B-spline bases. We can then plug these into `statsmodels` for ordinary least squares:

```{code-cell} ipython3
from patsy import dmatrix

# Build a cubic B-spline basis with 6 degrees of freedom
spline_basis = dmatrix(
    "bs(x, df=6, degree=3, include_intercept=False)",
    {"x": data["x"]},
    return_type="dataframe",
)
```

Fit an OLS model:

```{code-cell} ipython3
spline_model = sm.OLS(data["y"], spline_basis).fit()
print(spline_model.summary())
```

Visualise the spline fit:

```{code-cell} ipython3
x_grid = np.linspace(data["x"].min(), data["x"].max(), 300)
spline_grid = dmatrix(
    "bs(x, df=6, degree=3, include_intercept=False)",
    {"x": x_grid},
    return_type="dataframe",
)
y_spline_hat = spline_model.predict(spline_grid)

fig, ax = plt.subplots()
ax.scatter(data["x"], data["y"], alpha=0.3, label="Data")
ax.plot(x_grid, y_spline_hat, label="Cubic B-spline (df = 6)", linewidth=2)
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Spline regression")
ax.legend();
```

---

## Local Regression (LOWESS)

Polynomial and spline regression still use a **global basis**: one set of parameters applies to the whole range of $x$. Local regression takes a different view:

> Fit a **separate regression** in a neighbourhood around each target point, using only nearby observations (with weights).

For a target point $x_0$:

1. Define a neighbourhood around $x_0$, e.g. the closest $\alpha \cdot n$ observations, where $\alpha$ is a smoothing parameter between 0 and 1.
2. Assign **weights** to observations, typically higher for points closer to $x_0$.
3. Fit a **weighted least squares** regression (often linear or quadratic) using those weighted points.
4. The fitted value at $x_0$ is the prediction from this local model.

Repeat this for many $x_0$ to obtain a smooth curve.

Key parameters:

* **Span / fraction (`frac`)**: proportion of data used in each local fit

  * Small `frac` → very flexible, risk of overfitting
  * Large `frac` → smoother, less flexible
* **Degree** of local polynomial (often 1 or 2)


`statsmodels` provides a convenient implementation of LOWESS (locally weighted scatterplot smoothing).

```{code-cell} ipython3
from statsmodels.nonparametric.smoothers_lowess import lowess

x = data["x"].to_numpy()
y = data["y"].to_numpy()

# Try different fractions
frac_list = [0.2, 0.5, 0.7]

fig, ax = plt.subplots()
ax.scatter(x, y, alpha=0.3, label="Data")

for frac in frac_list:
  result = lowess(y, x, frac=frac, return_sorted=True)
  ax.plot(result[:, 0], result[:, 1], label=f"LOWESS, frac = {frac}", linewidth=2)

ax.set(xlabel="x", ylabel="y", title="Local regression (LOWESS) with different spans")
ax.legend();
```

Observe:

* `frac = 0.2` follows the data more closely (more wiggles).
* `frac > 0.5` give a more general trend.

Local regression methods are very useful for **exploratory analysis**: they give a flexible, data-driven summary of the trend without a strong global parametric assumption.

---

## Summary and Quiz

| Method | Key Idea | Flexibility | Pros | Cons |
|--------|----------|-------------|------|------|
| **Polynomial Regression** | Add powers of $x$ as predictors: $x, x^2, \dots, x^d$ | Controlled by degree $d$ | Simple to fit with OLS; interpretable for low degrees | Oscillates at boundaries; global changes affect entire curve; high degrees overfit |
| **Stepwise Regression** | Fit constant values in intervals defined by knots | Controlled by number and placement of knots | Very simple and interpretable; good starting point for understanding splines | Discontinuous at knots; sensitive to knot placement; rough appearance |
| **Spline Regression** | Piecewise polynomials smoothly joined at knots | Controlled by number of knots and degree | Smooth and flexible; local changes stay local; handles complex patterns | Requires choosing knot locations and number; can overfit with too many knots |
| **Local Regression (LOWESS)** | Fit weighted regression in neighborhood of each point | Controlled by span/fraction parameter | Data-driven; no global assumptions; excellent for exploration | Computationally intensive; requires choosing span; harder to interpret than parametric models |


In practice, these methods are often combined or used alongside regularisation and cross-validation to control overfitting and select tuning parameters (degree, knots, span).

---

```{code-cell} ipython3
:tags: ["remove-input"]
from jupyterquiz import display_quiz
display_quiz("quiz/BeyondLinear.json", shuffle_answers=True)
display_quiz("quiz/BeyondLinear2.json", shuffle_answers=True)
```