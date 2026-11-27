---
short_title: GAMs
kernelspec:
  name: python3
  display_name: Python 3
---
# ➕ Generalized Additive Models

Generalized Additive Models (GAMs) extend linear models by allowing **non-linear relationships** between each predictor and the outcome. Instead of the linear model

$$ y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_p x_p + \epsilon $$

a GAM can be written as

$$ y = \beta_0 + f_1(x_1) + f_2(x_2) + \dots + f_p(x_p) + \epsilon $$

Each fixed slope $\beta_j$ is replaced by a flexible function $f_j$, usually a spline as in the previous session. In the previous session, we used splines to predict $y$ from a single predictor $x$. GAMs combine several of these functions, one for each predictor.

The model remains *additive*: the effects of the predictors are simply added up. This makes GAMs easy to interpret, because we can look at each function $f_j$ on its own, while the other predictors are held constant. The price is that a GAM does not model interactions between predictors, unless we add them explicitly.

## Today's data: The Diabetes dataset

For today's practical demonstration, we will work with the `Diabetes` dataset from `scikit-learn`, which you already know from the PCA session. This dataset contains medical information collected from 442 diabetes patients, including:

- Features: 10 baseline measures from the beginning of the study: age, sex, Body Mass Index (BMI), average blood pressure, as well as six blood serum measurements (e.g. cholesterol, blood sugar, etc.). All features are already centered and scaled.
- Target: A quantitative measure of disease progression one year after the baseline measurements were taken.

You can find more information [here](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_diabetes.html).


```{code-cell} ipython3
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Get data
diabetes = datasets.load_diabetes(as_frame=True)
X = diabetes.data
y = diabetes.target

# Split the data into training and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

To explore the relationships between each feature and the target, we plot each predictor against the disease progression outcome. These scatter plots with simple linear regressions help us to visually assess whether the relationship between a feature and the target is linear or if we need a more flexible model approach:

```{code-cell} ipython3
:tags: [remove-input]

import matplotlib.pyplot as plt
import seaborn as sns

# Combine into a single DataFrame for easy plotting
df = X.copy()
df['target'] = y

features_to_plot = X.columns

# Set up the plot grid
num_features = len(features_to_plot)
cols = 3
rows = (num_features + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(15, 4 * rows))
axes = axes.flatten()

# Plot each feature against the target
for i, col in enumerate(features_to_plot):
    sns.regplot(x=col, y='target', data=df, ax=axes[i], scatter_kws={'s': 10}, line_kws={'color': 'red'})
    axes[i].set_title(f"{col} vs target")

# Hide any unused subplots
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()
```

Most relationships look roughly linear, but we cannot rule out some curvature. Let's see whether a more flexible model helps.


## GAMs in Python

There are multiple options for implementing GAMs. We will here use `statsmodels`, as you are already familiar with it from the previous sessions. If you need more information than provided in this session, check out the [documentation](https://www.statsmodels.org/dev/gam.html).

Our workflow will be the following:

1. Separate smooth (continuous) and categorical features
2. Create spline basis functions for the continuous features using B-splines
3. Fit the GAM with the smooth terms, the categorical predictors, and an intercept

We leave out `s1` (total cholesterol). It is almost completely determined by the other serum measurements: a linear regression of `s1` on `s2` to `s5` reaches an $R^2$ of 0.98. With flexible functions, such nearly redundant predictors can cancel each other out, and their estimated effects become huge and meaningless (just like the correlated predictors in the regularization session).

```{code-cell} ipython3
import statsmodels.api as sm
from statsmodels.gam.api import GLMGam, BSplines

spline_features = ['age', 'bmi', 'bp', 's2', 's3', 's4', 's5', 's6']
categorical_features = ['sex']

# Place the outer knots at the range of the full data, so that we can later also
# predict test observations that lie outside the range of the training data
knot_kwds = [dict(lower_bound=X[f].min(), upper_bound=X[f].max()) for f in spline_features]

# Create smoother for continuous variables
bs = BSplines(
    X_train[spline_features],
    df=[6]*len(spline_features), 
    degree=[3]*len(spline_features),
    knot_kwds=knot_kwds,
    constraints='center'  # center each smooth function around zero (see below)
)

# Fit GAM with the smoother, and sex + intercept as linear terms
gam = GLMGam(y_train, exog=sm.add_constant(X_train[categorical_features]), smoother=bs)
res = gam.fit()

print(res.summary())
```

```{margin}
Remember: B-splines are a series of piecewise polynomial functions that are joined smoothly at certain points called knots.
```

For the B-splines we choose:

- `df=[6]*len(spline_features)` -> 6 basis functions per feature
- `degree=[3]*len(spline_features)` -> cubic splines (degree 3)
- `constraints='center'` -> each smooth function has an average of zero over the data. Without such a constraint, we could shift any function up or down and the intercept down or up, without changing the predictions. The constraint removes one parameter, so each feature gets 5 coefficients (e.g. `bmi_s0` to `bmi_s4` in the summary)

The `coef` column shows the estimated coefficients for the intercept, `sex`, and all spline basis functions. The coefficients of the individual basis functions are not interpreted directly. Instead, we look at the function they form together. Due to the additive structure of GAMs, we can plot the effect of each smooth term on its own, while all other predictors are held constant:

```{code-cell} ipython3
fig, ax = plt.subplots(3, 3, figsize=(10, 9))

for i, feature in enumerate(spline_features):
    res.plot_partial(i, plot_se=True, cpr=True, include_constant=False, ax=ax.flat[i])
    ax.flat[i].set_title(f"Partial Effect: {feature}")

ax.flat[-1].axis("off")  # we only have 8 smooth terms
plt.tight_layout()
```

Each subplot shows the estimated contribution of one predictor to the predictions of the model.

- `plot_se=True` adds a confidence interval for the estimated function. It is narrow where there are many observations and wider towards the edges, where there are only a few
- `cpr=True` adds a component-plus-residual plot (a scatterplot of partial residuals)

Because the functions are centered, a value of 0 means an average contribution to the prediction, and positive or negative values show how much higher or lower the predicted disease progression is compared to this average.

Disease progression clearly increases with `bmi` (slightly more steeply for high values) and with blood pressure (`bp`), and it decreases with `s3` (HDL, the "good" cholesterol). The other functions bend and wiggle a little, but their confidence intervals are wide, so these curves may mostly reflect noise.

## Does the GAM predict better?

The GAM is more flexible than a linear model, but it also has many more parameters: 42 instead of 10. To see whether this pays off, we compare it with a linear regression on the same features, using the test set. We also fit a second GAM with fewer basis functions (`df=4`):

```{code-cell} ipython3
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def fit_gam(df):
    """Fit a GAM with the given df for all spline features and return the train and test R²."""
    bs = BSplines(X_train[spline_features], df=[df]*len(spline_features),
                  degree=[3]*len(spline_features), knot_kwds=knot_kwds, constraints='center')
    res = GLMGam(y_train, exog=sm.add_constant(X_train[categorical_features]), smoother=bs).fit()
    y_pred = res.predict(exog=sm.add_constant(X_test[categorical_features], has_constant="add"),
                         exog_smooth=X_test[spline_features])
    return r2_score(y_train, res.fittedvalues), r2_score(y_test, y_pred)

features = spline_features + categorical_features
lr = LinearRegression().fit(X_train[features], y_train)

results = pd.DataFrame({
    "Linear regression": [lr.score(X_train[features], y_train), lr.score(X_test[features], y_test)],
    "GAM (df=4)": fit_gam(4),
    "GAM (df=6)": fit_gam(6),
}, index=["Train R²", "Test R²"])

results.round(3)
```

The GAM with `df=6` fits the training data best, but it performs worse than linear regression on the test set: with 42 parameters for 309 patients, it overfits. With fewer basis functions (`df=4`), the GAM performs about as well as linear regression. This fits the partial effects above: most relationships in this dataset are close to linear, so there is little to gain from more flexibility.

In practice, we choose the flexibility of a GAM with cross-validation, just like the degree of a polynomial or the $\lambda$ of a Ridge model. Instead of changing `df`, `statsmodels` also allows a smoothing penalty (the `alpha` argument of `GLMGam`), which shrinks the spline functions towards smoother curves.

## Summary

- GAMs replace the linear terms of a regression model with flexible functions, usually splines. You do not have to assume a linear relationship for every predictor.
- Thanks to their additive structure, GAMs stay interpretable: you can inspect each smooth effect with `.plot_partial()`, and the linear terms directly in the model summary.
- `statsmodels` makes it easy to combine smooth terms (B-splines or cyclic cubic splines) with categorical or linear predictors.
- More flexibility is not always better: as with all flexible models, check the performance on new data.

```{code-cell} ipython3
:tags: [remove-input]
import sys; sys.path.append("..")
from quizpager import display_quiz_paged
display_quiz_paged('quiz/GAM.json')
```

**Now it's your turn:** head to the [exercise section](Exercises.ipynb) for Exercise 8, in which you will explore the flexibility of GAMs and model wages with several predictors.
