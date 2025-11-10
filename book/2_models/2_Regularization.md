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
myst:
  substitutions:
    lambda: 1
---

# <i class="fa-solid fa-puzzle-piece"></i> Regularisation, PCR & PLS

Building on subset selection, an alternative approach is to include all *p* predictors in the model but apply regularization, shrinking the coefficient estimates toward zero relative to the least squares estimates. This reduces model complexity without fully discarding variables. Though it introduces some bias, it often lowers variance and improves test performance. 

3 approaches are commonly used to regularize the predictors:

- Ridge Regression (L2 Regression)
- Lasso Regression (L1 Regression)
- Elastic Net Regression

Beyond penalising coefficients, we can also transform the predictor space:
- Principal Component Regression (PCR)
- Partial Least Squares (PLS)

We will explore all of these models in today's session. For the initial practical demonstration, we will again use the `Hitters` dataset:

```{code-cell} ipython3
import statsmodels.api as sm 

hitters = sm.datasets.get_rdataset("Hitters", "ISLR").data

# Keep 15 variables - the target ´Salary´ and 14 features.
hitters_subset = hitters[["Salary", "AtBat", "Runs","RBI", "CHits", "CAtBat", "CRuns", "CWalks", "Assists", "Hits", "HmRun", "Years", "Errors", "Walks"]].copy()
hitters_subset = hitters_subset.drop(columns=["CRuns", "CAtBat"]) # Remove highly correlated features (see previous session)
hitters_subset.dropna(inplace=True) # drop rows containing missing data

hitters_subset.head()
```

Once again, let's take a look at the correlation between predictors to check for potential multicollinearity, which can affect the stability of linear regression models:

```{code-cell} ipython3
import seaborn as sns

sns.heatmap(hitters_subset.corr(), annot=True, cmap="coolwarm", fmt=".2f");
```

We now continue to prepare our data by defining target and features and split it into training and test set:

```{code-cell} ipython3
from sklearn.model_selection import train_test_split

y = hitters_subset["Salary"]
X = hitters_subset.drop(columns=["Salary"])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

Great! We are now ready for using the data in our regression models.


## Ridge Regression

The loss function for ridge regression is defined as:

$$L(\boldsymbol\beta)=\sum_{i=1}^{n}\bigl(y_i - \mathbf X_i\,\boldsymbol\beta\bigr)^2\;+\;\lambda\sum_{j=1}^{p} \beta_j^2$$

Where:

- $n$ is the number of observations
- $p$ is the number of predictors
- $y_i$​ is the target for observation $i$
- $X_i$ is the row vector of features for observation $i$
- $\beta$ are the regression coefficients
- $\lambda$ is the regularization strength

Combined:

- $\sum_{i=1}^{n}\bigl(y_i - \mathbf X_i\,\boldsymbol\beta\bigr)^2$ is the residual sum of squares (RSS)
- $\lambda\sum_{j=1}^{p} \beta_j^2$ is the L2 penalty on the coefficients

```{admonition} The λ parameter
:class: note 

λ controls the regulariztation strength:

- λ = 0: The penalty term has no effect (normal OLS regression)
- λ > 0: The impact of the penalty increases proportinal to λ

Lambda is a hyperparameter which we need to chose ourselves (through e.g. cross validation). 
```

We can implement Ridge regression as follows:

**Step 1:** We first standardize the predictors as ridge regression is sensitive to scaling. This can be done by using e.g. `StandardScaler` from `scikit-learn`: 

```{code-cell} ipython3
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler() # Scale predictors to mean=0 and std=1
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Step 2:** Next, we set up a range of values for λ. We will test 100 values between 0.001 and 150:

```{code-cell} ipython3
import numpy as np 

lambda_range = np.linspace(0.001, 150, 100)
```

**Step 3:** For each lambda, we will perform ridge regression on the training data. Cross-validation is then used to identify the optimal lambda that minimizes prediction error:

```{margin}
Note: In `sklearn`, the lamda parameter is called alpha.
```

```{code-cell} ipython3
from sklearn.linear_model import RidgeCV
import pandas as pd

# Get the optimal lambda
ridge_cv = RidgeCV(alphas=lambda_range, cv=5)
ridge_cv.fit(X_train_scaled, y_train) 

print(f"Optimal alpha: {ridge_cv.alpha_}")

# Get training R²
train_score_ridge= ridge_cv.score(X_train_scaled, y_train)
print(f"Training R²: {train_score_ridge}")
```

We can then inspect the coefficients of the regression model:

```{code-cell} ipython3
# Put the coefficients into a nicely formatted df for visualization
coef_table = pd.DataFrame({
    'Predictor': X_train.columns,
    'Beta': ridge_cv.coef_
})

coef_table = coef_table.reindex(coef_table['Beta'].abs().sort_values(ascending=False).index)
print(coef_table)
```

**Step 4:** Finally, we can evaluate the model on the testing data to assess its ability for generalization:

```{code-cell} ipython3
test_score_ridge = ridge_cv.score(X_test_scaled, y_test)

print(f"Test R²: {test_score_ridge}")
```

**Additional visualization:** To get a better intuition about how different lambda values affect the beta estimates, we can simply plot them for each model: 

```{code-cell} ipython3
from sklearn.linear_model import Ridge
import numpy as np 

coefs = []
alphas = np.linspace(0.001, 50, 100)

for a in alphas:
  ridgereg = Ridge(alpha=a)
  ridgereg.fit(X_train_scaled,y_train)
  coefs.append(ridgereg.coef_)

# Plot
fig, ax = plt.subplots()
ax.plot(alphas, coefs)
ax.set(title="Ridge coefficients", xlabel="Lambda", ylabel="Beta");
```

## Lasso Regression

In contrast to ridge regression, lasso regression will not retain all predictors in the final model but force some of them to zero. The loss function for lasso is given as:

$$L(\boldsymbol\beta)=\sum_{i=1}^{n}\bigl(y_i - \mathbf X_i\,\boldsymbol\beta\bigr)^2\;+\;\lambda\sum_{j=1}^{p}\bigl\lvert \beta_j \bigr\rvert$$

Where:

- $\sum_{i=1}^{n}\bigl(y_i - \mathbf X_i\,\boldsymbol\beta\bigr)^2$ is the residual sum of squares (RSS)
- $\lambda\sum_{j=1}^{p}\bigl\lvert \beta_j \bigr\rvert$ is the L1 penalty on the coefficients

You can see, that the concept is nearly identical, with the only difference being the form of the penalty term. Lasso uses the L1 norm ($\bigl\lvert \beta_j \bigr\rvert$) while Ridge uses the L2 norm ($\beta_j^2$).
Because of this geometric difference in the penalty, Lasso can produce sparse models by zeroing out irrelevant features, whereas Ridge keeps all predictors in the model but with smaller weights. 
Both balance the trade‐off between model fit (the RSS term) and complexity via the tuning parameter λ, typically chosen by cross‐validation.

```{code-cell} ipython3
:tags: [remove-input]
from sklearn.linear_model import Lasso
import numpy as np 

coefs = []
alphas = np.linspace(0.001, 50, 100)

for a in alphas:
  lassoreg = Lasso(alpha=a)
  lassoreg.fit(X_train_scaled,y_train)
  coefs.append(lassoreg.coef_)

# Plot
fig, ax = plt.subplots()
ax.plot(alphas, coefs)
ax.set(title="Lasso coefficients", xlabel="Lambda", ylabel="Beta");
```


### Why Ridge and Lasso Behave Differently: The Budget

When we use Ridge or Lasso regression, we are not just trying to fit the data perfectly — we are also trying to control how large the coefficients $\beta_j$ become. If we let them grow without any limits, the model could become too complex and overfit the data.

To prevent this, Ridge and Lasso add a kind of budget:
We are only allowed to spend a certain amount on the size of the coefficients. You can think of it like having a fixed amount of money that you have to divide up among all your coefficients.

This budget is enforced by a mathematical condition. For Lasso, the total absolute value of the coefficients must stay below a certain limit $t$:
    
  $$\sum_{j=1}^p \lvert \beta_j \rvert \;\le\; t$$

For Ridge, the total squared value of the coefficients must stay below this limit:
    
  $$\sum_{j=1}^p \beta_j^2 \;\le\; t$$

Even though both Ridge and Lasso limit the size of the coefficients, how they limit them is different — and this changes the behaviour of the models.

- In Lasso regression (left plot), the constraint $\sum |\beta_j| \le t$ forms a diamond shape (or a "pointy" version in higher dimensions). Because of these sharp corners, Lasso often drives some coefficients exactly to zero — which is why Lasso can create sparse models (feature selection).

- In Ridge regression (right plot), the constraint $\sum \beta_j^2 \le t$ forms a circular shape (or a sphere in higher dimensions). This means that Ridge tends to shrink coefficients smoothly but rarely sets them exactly to zero.


```{figure} figures/Budget_LassoRidge.png 
:name: Budget Lasso vs Ridge Regression
:alt: Budget
:align: center

The budget for Lasso and Ridge regression.
```


## Elastic Net Regression

Neither Ridge nor Lasso will universally be better than the other. The main difference between them is the coefficient selection:

- Lasso tends to shrink some coefficients all the way to zero, effectively performing feature selection by excluding less relevant variables.
- Ridge also shrinks coefficients toward zero, but does not eliminate any completely. Instead of selecting variables, it distributes the effect across all features.

Another point to note is their response to multicollinearity:

  - Ridge shares the weight among collinear features, keep them all (shrink but don’t zero)
  - Lasso chooses a subset (often just one) from each collinear group and zeroes out the rest

This is where elastic net regression comes into play. Elastic net is a combination of Ridge and Lasso:

$$
L(\boldsymbol\beta) \;=\;
\sum_{i=1}^{n}\bigl(y_i - \mathbf X_i\,\boldsymbol\beta\bigr)^2
\;+\;
\lambda
\Biggl[
\alpha \sum_{j=1}^{p}\bigl\lvert\beta_j\bigr\rvert
\;+\;
\frac{1 - \alpha}{2}\sum_{j=1}^{p}\beta_j^2
\Biggr]
$$

Here, the mixing parameter $\alpha\in[0,1]$ determines how much weight to give each penalty: when  $\alpha=1$ you recover pure Lasso (only an L1 penalty), and when  $\alpha=0$ you recover pure Ridge (only an L2 penalty); values in between blend the two. By combining these penalties, Elastic Net can both shrink coefficients (like Ridge) and select variables (like Lasso), while also tending to select or drop groups of correlated features together. In practice, you tune both λ (overall regularization strength) and $\alpha$ via cross-validation to find the best compromise between bias, variance, and sparsity for your particular dataset.

As a summary, we can look at coefficients estimated by all introduced methods:

```{code-cell}  ipython3
:tags: [remove-input]
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, RidgeCV, ElasticNetCV, LinearRegression
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import statsmodels.api as sm

# Data related processing
hitters = sm.datasets.get_rdataset("Hitters", "ISLR").data
hitters_subset = hitters[[
    "Salary", "AtBat", "Runs", "RBI", "CHits", "CWalks",
    "Assists", "Hits", "HmRun", "Years", "Errors", "Walks"
]].copy()
hitters_subset.dropna(inplace=True)
y = hitters_subset["Salary"]
X = hitters_subset.drop(columns=["Salary"])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Scale predictors
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# Regularization paths
lambda_range = np.linspace(0.001, 50, 100)

# Lasso CV
lasso_cv = LassoCV(alphas=lambda_range, cv=5).fit(X_train_scaled, y_train)

# Ridge CV
ridge_cv = RidgeCV(alphas=lambda_range, cv=5).fit(X_train_scaled, y_train)

# ElasticNet CV (blend of L1 & L2)
enet_cv = ElasticNetCV(
    alphas=lambda_range,
    l1_ratio=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],  # try a few mixing ratios
    cv=5
).fit(X_train_scaled, y_train)

# Ordinary Linear Regression
lr = LinearRegression().fit(X_train_scaled, y_train)

# Plotting coefficients
features = X.columns.tolist()
x_pos    = np.arange(len(features))

plt.scatter(x_pos, ridge_cv.coef_, alpha=0.8, label='Ridge Regression')
plt.scatter(x_pos, lasso_cv.coef_, alpha=0.8, label='Lasso Regression')
plt.scatter(x_pos, enet_cv.coef_,  alpha=0.8, label='Elastic Net Regression')
plt.scatter(x_pos, lr.coef_,       alpha=0.4, label='Linear Regression')

plt.axhline(0, color='gray', linestyle='--')
plt.xticks(x_pos, features, rotation=45, ha='right')
plt.xlabel("Feature")
plt.ylabel("Coefficient")
plt.title("Comparison between regularization methods")
plt.legend()
plt.tight_layout()
plt.show()
```

**Now it's your turn:** Head to the [exercise section](Exercises) for Exercise 2, in which you will implement a Lasso model.


## Principal Component Regression

Regularised regression models such as Ridge and Lasso address issues with correlated features or high-dimensional predictors by shrinking the regression coefficients. Another way to handle these issues is to transform the predictor space itself
before regression. This leads to Principal Component Regression (PCR).

Principal Component Regression (PCR) first performs Principal Component Analysis (PCA) on the predictor matrix `X`. Then, it fits a linear regression model on the PCA-transformed data. This approach works well when directions of high variance in `X` are also predictive for `y`. However, since PCA is unsupervised, it may drop components that explain little variance but have high predictive power.

Let's have a look at this simulated data:

```{code-cell} ipython3
---
tags:
  - hide-input
---
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

rng = np.random.RandomState(0)
n_samples = 500
cov = [[3, 3], [3, 4]]
X = rng.multivariate_normal(mean=[0, 0], cov=cov, size=n_samples)

fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], alpha=0.3)
ax.set(xlabel="x1", ylabel="x2", title="Simulated data");
```

We perform a PCA to get and visualize the two directions of highest variance:

```{code-cell} ipython3
pca = PCA(n_components=2).fit(X)

fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], alpha=0.3)
for i, (comp, var) in enumerate(zip(pca.components_, pca.explained_variance_)):
    ax.plot([0, comp[0]*var], [0, comp[1]*var], lw=5, c=f"C{i + 2}", label=f"PC{i+1}")
ax.set(aspect="equal", xlabel="x1", ylabel="x2", title="Principal components")
plt.legend();
```

For the purpose of this example, we now define the target to be aligned with low variance. This makes `y` strongly correlated with PC2 (the low variance direction):

```{code-cell} ipython3
y = X.dot(pca.components_[1]) + rng.normal(size=n_samples) / 2

fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].scatter(X.dot(pca.components_[0]), y, alpha=0.3)
ax[0].set(xlabel="Projection on PC1", ylabel="y", title="High-variance direction")
ax[1].scatter(X.dot(pca.components_[1]), y, alpha=0.3)
ax[1].set(xlabel="Projection on PC2", ylabel="y", title="Low-variance direction")
plt.tight_layout();
```

On this data, we can then perform PCR. Let's first start with one component:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng)

pcr = make_pipeline(StandardScaler(), PCA(n_components=1), LinearRegression())
pcr.fit(X_train, y_train)

pca_step = pcr.named_steps["pca"]
plt.scatter(pca_step.transform(X_test), y_test, alpha=0.3, label="Ground truth")
plt.scatter(pca_step.transform(X_test), pcr.predict(X_test), alpha=0.3, label="PCR predictions")
plt.xlabel("First PCA component")
plt.ylabel("y")
plt.title("PCR (1 component)")
plt.legend()
plt.show()

print(f"PCR R²: {pcr.score(X_test, y_test):.3f}")
```

Because PCA is unsupervised, it focuses on directions of high variance (PC1), even though PC2 contains most of the predictive signal. Hence, PCR performs poorly with one component. If we add a second component, we see that the PCR now captures the predictive direction (PC2) and performs much better:

```{code-cell} ipython3
pcr_2 = make_pipeline(StandardScaler(), PCA(n_components=2), LinearRegression())
pcr_2.fit(X_train, y_train)
print(f"PCR (2 components) R²: {pcr_2.score(X_test, y_test):.3f}")
```

## Partial Least Squares Regression

While PCR focuses purely on the variance structure of `X`, Partial Least Squares (PLS) introduces supervision into the process and can be thought of as a supervised extension of PCR.

Instead of finding directions that maximise variance in `X`, PLS finds latent components that maximise the covariance between `X` and `y`. This allows PLS to identify directions that are both informative about `y` and stable under multicollinearity, even if they explain only modest variance in `X`. As a result, PLS often performs better than PCR when the most predictive information lies in low-variance directions (a situation where PCR tends to fail).

Fitting a model is straightforward:

```{code-cell} ipython3
from sklearn.cross_decomposition import PLSRegression

pls = PLSRegression(n_components=1)
pls.fit(X_train, y_train)

plt.scatter(pls.transform(X_test), y_test, alpha=0.3, label="Ground truth")
plt.scatter(pls.transform(X_test), pls.predict(X_test), alpha=0.3, label="PLS predictions")
plt.xlabel("First PLS component")
plt.ylabel("y")
plt.title("PLS (1 component)")
plt.legend()
plt.show()

print(f"PLS R²: {pls.score(X_test, y_test):.3f}")
```

You can see, even with a single component, PLS can align with the predictive direction because it uses target information. Thus, it achieves a high $R^2$, unlike PCR with one component.