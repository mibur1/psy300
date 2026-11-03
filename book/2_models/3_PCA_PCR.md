---
short_title: PCA, PCR & PLS
kernelspec:
  name: python3
  display_name: Python 3
---
# 🔎 PCA, PCR & PLS

In the last session, we dealt with many or correlated predictors by selecting a subset of them or by shrinking their coefficients. Today we take a third route: we first combine the predictors into a few new variables and then use these for the regression.

## Principal Component Analysis

Principal Component Analysis (PCA) is an *unsupervised* method for *dimensionality reduction*. It describes a dataset with fewer variables while losing as little information as possible. There is no target: PCA only looks at the variables themselves.

PCA finds new axes for the data, the *principal components* (PCs). PC1 points in the direction in which the data vary most. PC2 is perpendicular to PC1 and captures the largest remaining variance, and so on. With $p$ variables, there are at most $p$ components. Mathematically, the components are the eigenvectors of the covariance matrix of the data.

Each component is a weighted sum of the original variables:

$$
Z_1 = \phi_{11} X_1 + \phi_{21} X_2 + \dots + \phi_{p1} X_p
$$

$X_1, \dots, X_p$ are the (centered) variables. The weights $\phi_{11}, \dots, \phi_{p1}$ are the *loadings*: they tell us how much each variable contributes to PC1, and together they form its direction (the red arrow in the plot below). If we plug in the values of one observation, we get its *score* on PC1, i.e. its position along this axis. Keeping only the first few components, we can describe each observation with a few scores instead of all $p$ values. In scikit-learn, `pca.components_` holds the loadings and `pca.transform()` returns the scores.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Simulate centered 2D data
rng = np.random.RandomState(0)
X = rng.multivariate_normal(mean=[0, 0], cov=[[3, 3], [3, 4]], size=500)

# Run PCA: extract the components and the standard deviation along each of them
pca2d = PCA().fit(X)
pcs, scales = pca2d.components_, np.sqrt(pca2d.explained_variance_)

# Plot data
fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], alpha=0.3, label='Data')
mean = X.mean(axis=0)

# Plot PCs
ax.arrow(*mean, *(pcs[0] * scales[0] * 3), head_width=0.2, head_length=0.3, color='r', linewidth=2, label='PC1') # *3 scaling just for visualization
ax.arrow(*mean, *(pcs[1] * scales[1] * 3), head_width=0.2, head_length=0.3, color='b', linewidth=2, label='PC2') # *3 scaling just for visualization
ax.set(xlabel="Feature 1", ylabel="Feature 2", title="PCA for 2D data")
ax.axis('equal')
ax.legend();

print("Explained variance ratio:", pca2d.explained_variance_ratio_.round(3))
```

The two features are strongly correlated, so most of the variation lies along a single diagonal direction (PC1, 93% of the variance). PC2 only captures the remaining 7%. If we kept only PC1, we would describe each observation with one number instead of two and still keep most of the information.

### PCA on the Iris dataset

Next, we apply PCA to the Iris dataset (4 features), which you already know from the resampling session. A pairplot shows how the features relate to each other and to the species:

```{code-cell} ipython3
import seaborn as sns
from sklearn.datasets import load_iris

# Load Iris as a DataFrame for easy plotting
iris = load_iris(as_frame=True)
iris.frame["target"] = iris.target_names[iris.target]
sns.pairplot(iris.frame, hue="target");
```

Petal length and petal width separate the three species most clearly. They are also strongly correlated with each other and with sepal length, so the four features contain a lot of redundant information. Let's fit a PCA with three components:

```{code-cell} ipython3
import pandas as pd

# Prepare data arrays
X_iris, y_iris = iris.data, iris.target

# Fit PCA to retain the first 3 principal components
pca = PCA(n_components=3).fit(X_iris)

# Loadings: one row per component, one column per original feature
loadings = pd.DataFrame(pca.components_, columns=iris.feature_names, index=["PC1", "PC2", "PC3"])
print(loadings.round(2))
print("\nExplained variance ratio:", pca.explained_variance_ratio_.round(3))
```

* **`pca.components_`** contains the loadings. Each row is one component, and each entry tells us how strongly a feature contributes to it. PC1 is dominated by petal length (0.86), while PC2 is mostly made up of the two sepal measures.
* **`pca.explained_variance_ratio_`** is the fraction of the total variance that each component explains. PC1 alone explains 92.5%.

:::{important} Standardize before PCA
PCA looks for directions of large variance, and variance depends on the units of the features. A feature measured in millimeters instead of centimeters would have a 100 times larger variance and would take over PC1. For this reason, we usually standardize the features first (mean 0, standard deviation 1).

In Iris, all four features are measured in centimeters, so we left them as they are. Petal length simply varies the most, which is why it dominates PC1. On standardized features, PC1 would explain 73% instead of 92.5% of the variance.
:::

### How many components?

To decide how many components to keep, we can plot the explained variance of each component (a *scree plot*) and the cumulative explained variance:

```{code-cell} ipython3
pca_full = PCA().fit(X_iris)
evr = pca_full.explained_variance_ratio_
components = np.arange(1, len(evr) + 1)

fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].bar(components, evr)
ax[0].set(xlabel="Component", ylabel="Explained variance ratio", title="Scree plot", xticks=components)
ax[1].plot(components, evr.cumsum(), marker="o")
ax[1].axhline(0.95, color="gray", linestyle="--")
ax[1].set(xlabel="Number of components", ylabel="Cumulative explained variance", title="Cumulative", xticks=components, ylim=(0, 1.05))
plt.tight_layout();
```

A common rule of thumb is to look for the "elbow" in the scree plot, or to keep enough components to reach a certain amount of explained variance (e.g. 80% or 95%). For Iris, one or two components are enough. These rules are only heuristics. When the components are used for a prediction task, as in PCR below, we choose their number with cross-validation instead.

### Projecting the data

Finally, we can project the data onto the components with `.transform()`. This centers the data (subtracts each feature's mean) and then computes the scores for each component. The result `X_pca` has the shape `(n_samples, 3)`:

```{code-cell} ipython3
# Project (transform) the data into the first 3 PCs
X_pca = pca.transform(X_iris)

# Plot
fig = plt.figure(figsize=(12, 4), constrained_layout=True)

ax1 = fig.add_subplot(1, 3, 1)
scatter1 = ax1.scatter(X_pca[:, 0], np.zeros_like(X_pca[:, 0]), c=y_iris, cmap='viridis', s=40)
ax1.set(title='1 Component', xlabel='PC1')
ax1.get_yaxis().set_visible(False)

ax2 = fig.add_subplot(1, 3, 2)
ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=y_iris, cmap='viridis', s=40)
ax2.set(title='2 Components', xlabel='PC1', ylabel='PC2')

ax3 = fig.add_subplot(1, 3, 3, projection='3d', elev=-150, azim=110)
ax3.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:, 2], c=y_iris, cmap='viridis', s=40)
ax3.set(title='3 Components', xlabel='PC1', ylabel='PC2', zlabel='PC3')

handles, labels = scatter1.legend_elements()
legend = ax1.legend(handles, iris.target_names, loc='upper left', title='Species')
ax1.add_artist(legend);
```

The 3D panel is hard to read on a flat page, so here is the same projection as a figure you can **rotate and zoom** with the mouse. If you rotate it, you will see that almost all of the separation between the species lies along PC1:

```{code-cell} ipython3
:tags: [hide-input]

import plotly.graph_objects as go
import plotly.io as pio

tpl = pio.templates["plotly_white"]
tpl.layout.paper_bgcolor = "rgba(0,0,0,0)"
tpl.layout.plot_bgcolor = "rgba(128,128,128,0.08)"
tpl.layout.font.color = "#888888"
pio.templates["psy300"] = tpl
pio.templates.default = "psy300"

colours = ["#4c72b0", "#dd8452", "#55a868"]
evr = pca.explained_variance_ratio_

fig = go.Figure()
for k, name in enumerate(iris.target_names):
    mask = y_iris == k
    fig.add_trace(go.Scatter3d(
        x=X_pca[mask, 0], y=X_pca[mask, 1], z=X_pca[mask, 2],
        mode="markers", name=name,
        marker=dict(size=4, color=colours[k], opacity=0.8),
    ))

fig.update_layout(
    scene=dict(
        xaxis_title=f"PC1 ({evr[0]:.1%})",
        yaxis_title=f"PC2 ({evr[1]:.1%})",
        zaxis_title=f"PC3 ({evr[2]:.1%})",
    ),
    margin=dict(l=0, r=0, t=30, b=0), height=520,
    title="Iris projected onto its first three principal components",
)
fig
```

Note that PCA never saw the species labels. It found a direction that separates them anyway, because here the direction of largest variance happens to be the one that differs between species. As we will see next, this is not guaranteed.

## Principal Component Regression

Ridge and Lasso deal with correlated or many predictors by shrinking the regression coefficients. Principal Component Regression (PCR) takes a different approach and changes the predictors themselves:

1. Run a PCA on the predictors $X$ and keep the first $M$ components.
2. Fit a linear regression of $y$ on the new variables $Z_1, \dots, Z_M$ instead of the original predictors $X_1, \dots, X_p$:

$$
y = \theta_0 + \theta_1 Z_1 + \dots + \theta_M Z_M + \epsilon
$$

We write $\theta$ instead of $\beta$ for the coefficients, because they belong to the components, not to the original predictors.

The components are uncorrelated with each other, so the problem of correlated predictors disappears, and with $M < p$ we have fewer coefficients to estimate. The number of components $M$ controls the bias-variance trade-off, just like λ in Ridge and Lasso. With $M = p$, PCR is identical to ordinary linear regression. With fewer components, the model has less variance but more bias. We choose $M$ with cross-validation.

PCR works well when the directions of high variance in $X$ are also the ones that predict $y$. But since PCA ignores $y$, there is no guarantee for this. To see what can go wrong, let's have another look at the simulated data from above:

```{code-cell} ipython3
:tags: [hide-input]

fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], alpha=0.3, label='Data')
ax.arrow(*mean, *(pcs[0] * scales[0] * 3), head_width=0.2, head_length=0.3, color='r', linewidth=2, label='PC1')
ax.arrow(*mean, *(pcs[1] * scales[1] * 3), head_width=0.2, head_length=0.3, color='b', linewidth=2, label='PC2')
ax.set(xlabel="Feature 1", ylabel="Feature 2", title="PCA for 2D data")
ax.axis('equal')
ax.legend();
```

For this example, we construct the target so that it depends on the **low-variance** direction, PC2:

```{code-cell} ipython3
y = X.dot(pca2d.components_[1]) + rng.normal(size=500) / 2

fig, ax = plt.subplots(1, 2, figsize=(10, 3))
ax[0].scatter(X.dot(pca2d.components_[0]), y, alpha=0.3)
ax[0].set(xlabel="Projection on PC1", ylabel="y", title="High-variance direction")
ax[1].scatter(X.dot(pca2d.components_[1]), y, alpha=0.3)
ax[1].set(xlabel="Projection on PC2", ylabel="y", title="Low-variance direction")
plt.tight_layout();
```

We now fit a PCR with a single component. A pipeline lets us chain the scaling, the PCA, and the regression into one model:

```{code-cell} ipython3
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng)

pcr = make_pipeline(StandardScaler(), PCA(n_components=1), LinearRegression())
pcr.fit(X_train, y_train)

# Scores of the test data on the first component (scaling + PCA, without the regression)
X_test_pc = pcr[:-1].transform(X_test)

fig, ax = plt.subplots()
ax.scatter(X_test_pc, y_test, alpha=0.3, label="Ground truth")
ax.scatter(X_test_pc, pcr.predict(X_test), alpha=0.3, label="PCR predictions")
ax.set(xlabel="First PCA component", ylabel="y", title="PCR (1 component)")
plt.legend()
plt.show()

print(f"PCR R²: {pcr.score(X_test, y_test):.3f}")
```

:::{tip} Pipelines
`pcr[:-1]` selects all steps of the pipeline except the last one (the regression). The test data are therefore scaled with the training mean and standard deviation before they are projected. When you call `pcr.fit()` or `pcr.predict()`, the pipeline takes care of this automatically. This also matters for cross-validation: if you pass the whole pipeline to `cross_val_score()`, the scaler and the PCA are fitted on the training folds only.
:::

PCA keeps the direction of highest variance (PC1), but PC2 contains the predictive signal. With one component, PCR therefore does not predict anything: the R² is around 0. If we add the second component, PCR captures the predictive direction and performs much better:

```{code-cell} ipython3
pcr_2 = make_pipeline(StandardScaler(), PCA(n_components=2), LinearRegression())
pcr_2.fit(X_train, y_train)
print(f"PCR (2 components) R²: {pcr_2.score(X_test, y_test):.3f}")
```

Of course, with two components out of two features we are back at ordinary linear regression and have not reduced anything. The example is extreme on purpose, but the same thing happens in real data: a component that explains little variance in $X$ can still be important for predicting $y$. You will see an example of this in the exercise.

## Partial Least Squares Regression

Partial Least Squares (PLS) regression also builds new components from the predictors, but unlike PCA, it uses the target $y$ to do so. PLS is therefore a *supervised* method.

The first PLS component is a weighted sum of the predictors, just like a principal component. The difference lies in how the weights are chosen. PCA picks the direction with the largest variance in $X$. PLS picks the direction with the largest covariance with $y$: each (standardized) predictor gets a weight proportional to its covariance with $y$. The next components are found in the same way on what is left of $X$ and $y$ after removing the previous components. Because of this, PLS can find a predictive direction even if it only explains a small part of the variance in $X$.

Fitting the model is straightforward. `PLSRegression` standardizes the predictors by default (`scale=True`), so we do not need a scaler:

```{code-cell} ipython3
from sklearn.cross_decomposition import PLSRegression

pls = PLSRegression(n_components=1)
pls.fit(X_train, y_train)

fig, ax = plt.subplots()
ax.scatter(pls.transform(X_test), y_test, alpha=0.3, label="Ground truth")
ax.scatter(pls.transform(X_test), pls.predict(X_test), alpha=0.3, label="PLS predictions")
ax.set(xlabel="First PLS component", ylabel="y", title="PLS (1 component)")
plt.legend()
plt.show()

print(f"PLS R²: {pls.score(X_test, y_test):.3f}")
```

Even with a single component, PLS finds the predictive direction, because it uses the target to build its components. It reaches almost the same R² as PCR with both components.

This does not mean that PLS is always better. Since it uses $y$, PLS can adapt more closely to the training data, which means more variance. In practice, PCR and PLS often perform similarly, and in both cases we choose the number of components with cross-validation.

## Summary

| | PCA | PCR | PLS |
|---|---|---|---|
| Type | Unsupervised | Supervised (regression) | Supervised (regression) |
| Components | Directions of maximal variance in $X$ | Principal components of $X$ | Directions of maximal covariance with $y$ |
| Uses $y$ to build components? | No | No | Yes |
| Tuning parameter | Number of components | Number of components | Number of components |

* All three methods replace the $p$ original predictors with a smaller number of components. Each component is a weighted sum of **all** predictors, so unlike Lasso or subset selection, none of these methods removes predictors from the model.
* Standardize the predictors first, otherwise the predictors with the largest variance dominate the components.
* With all $p$ components, PCR and PLS are identical to ordinary linear regression.

```{code-cell} ipython3
:tags: [remove-input]
import sys; sys.path.append("..")
from quizpager import display_quiz_paged
display_quiz_paged('quiz/PCA_PCR.json')
```

**Now it's your turn:** head to the [exercise section](Exercises.ipynb) for Exercise 3, in which you will apply PCA, PCR, and PLS to a medical dataset.
