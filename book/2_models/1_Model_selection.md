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

# <i class="fa-solid fa-square-check"></i> Model Selection

As data scientists, you often have to work with huge amounts of data. For example, smart phones produce thousands of measurements a day. Having more information can, in theory, help us make better predictions, but it also brings risks: too many variables can make analyses infeasible, lead to false discoveries, or cause models to learn noise instead of real effects. 


## The large p issue

**Big data** refers to large data sets with many predictors which often cannot be processed or analyzed using traditional data processing techniques. For our prediction models, this brings some issues:

- While linear models can, in theory, still be applied to such data, the ordinary least squares (OLS) fit becomes infeasible, especially when p > n. In this case, the design matrix is not full rank and becomes singular. This means it cannot be inverted, which results in infinitely many possible solutions.
- The large number of features also reduces interpretability, making it more difficult for you as a scientist to understand which predictors truly drive the model’s behaviour.


This is where techniques like **linear model selection** becomes essential, offering techniques to refine our models and extract meaningful insights from high-dimensional data.


## Today's data: Hitters

For practical demonstration, we will use the `Hitters` dataset. This dataset provides Major League Baseball Data from the 1986 and 1987 seasons. It contains 322 observations of major league players on 20 variables (so it's not big data, but we can pretend it is). The Research aim is to predict a baseball player's salary on the basis of various predictors associated with the performance in the previous year. You can check its contents here: https://islp.readthedocs.io/en/latest/datasets/Hitters.html  

```{code-cell} ipython3
import statsmodels.api as sm 

# Get the data
hitters = sm.datasets.get_rdataset("Hitters", "ISLR").data
```

For computational reasons, we will not include all predictors but only a smaller subset:

```{code-cell} ipython3
# Keep a total of 10 variables - the target ´Salary´ and 9  features.
hitters_subset = hitters[["Salary", "CHits", "CAtBat", "CRuns", "CWalks", "Assists", "Hits", "HmRun", "Years", "Errors"]].copy()

# Remove rows with missing vlaues
hitters_subset.dropna(inplace=True)

hitters_subset.head()
```

Let’s also take a look at the correlation matrix to check for potential multicollinearity, which can affect the stability of linear regression models.

```{code-cell} ipython3
import seaborn as sns

sns.heatmap(hitters_subset.corr(), annot=True, cmap="coolwarm", fmt=".2f");
```

The heatmap reveals strong correlations between several predictors:

- `CHits` and `CAtBat` show a correlation of 1,
- `CHits` and `CRuns` have a very strong correlation of 0.98,

We thus remove two of the correlated features:

```{code-cell} ipython3
features_drop = ["CRuns", "CAtBat"]
hitters_subset = hitters_subset.drop(columns=features_drop)
```


## Handling big data in linear models

```{admonition} Handling big data
:class: hint

To handle large datasets efficiently in linear modeling, three methods will be introduced in this course:

- **Subset Selection**
- **Dimensionality Reduction**
- **Regularization / Shrinkage**

Today, we will focus on subset selection.
```

## Subset Selection
In subset selection we identify a subset of *p* predictos that are truly related to the outcome. The model get fitted using least squares on the reduces set of variables.

How do we determine which variables are relevant?! 

###  Best Subset Selection

We will start with performing Best Subset Selection (also called exhaustive search) as implemented in the `mlxtend` package. It has great documentation, e.g. for the [exhaustive search](https://rasbt.github.io/mlxtend/user_guide/feature_selection/ExhaustiveFeatureSelector/). In short, this approach is a brute-force evaluation of feature subsets. A specific performance metric (e.g. MSE, R², or accuracy) is optimized given an arbitrary regressor or classifier. For example, if we have 4 features, the algorithm will evaluate all 15 possible combinations of features.

```{code-cell} ipython3
:tags: ["remove-input"]
from jupyterquiz import display_quiz
display_quiz("quiz/BestSubsetSelection.json", shuffle_answers=True)
```

Before we start the subset selection, we first define the target and the features:

```{code-cell} ipython3
import numpy as np

y = hitters_subset["Salary"]
X = hitters_subset.drop("Salary", axis=1)
```

We first split our data into training and test dataset. Although the selection function uses cross-validation to identify the best subset of predictors (Step 3), this evaluation is done during the selection process and can still overfit to the data. To fairly assess how the final model performs on new data, we split off a test set and use it only after feature selection is complete.

```{code-cell} ipython3
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

|Purpose                        	   | What is it for?                                   | When?                                               |
|------------------------------------- |---------------------------------------------------|-----------------------------------------------------|
|Cross Validation in selection function|Helps choose the best subset of features           |During selection                                     |
|Test Set Evaluation                   |Checks how well the final model performs           |After selection                                      |


We then run Best Subset Selection on the training data:

```{code-cell} ipython3
from sklearn.linear_model import LinearRegression
from mlxtend.feature_selection import ExhaustiveFeatureSelector

efs = ExhaustiveFeatureSelector(
        estimator=LinearRegression(),
        min_features=1,
        max_features=X_train.shape[1],
        scoring='r2',
        cv=5,
        print_progress=False)

efs.fit(X_train, y_train)

print('Best R²: %.2f' % efs.best_score_)
print('Best subset (indices):', efs.best_idx_)
print('Best subset (corresponding names):', efs.best_feature_names_)
```

If you are interested in the details, they are stored in the metric dictionary:

```{code-cell} ipython3
import pandas as pd

df = pd.DataFrame.from_dict(efs.get_metric_dict()).T
df.sort_values('avg_score', inplace=True, ascending=False)
df
```

### Forward Stepwise Selection

Forward Stepwise Selection is a greedy feature‐selection method that starts with no features and, at each step, adds the single feature whose inclusion most improves the model’s performance. This “add the best remaining feature” is repeated until a desired number of features is reached, at which point the algorithm stops and returns the best subset.

An important parameter is `k_features`, which determines the number of features to select. We can, pass an integer (must be less than the total number of available features), a tuple (the algorithm will then evaluate all subset sizes between the min and max value), or one of two string options (`"best"`, or `"parsimonious"`). Please refer to the [documentation](https://rasbt.github.io/mlxtend/api_subpackages/mlxtend.feature_selection/) for further details.

```{code-cell} ipython3
from mlxtend.feature_selection import SequentialFeatureSelector

sfs_forward = SequentialFeatureSelector(
    estimator=LinearRegression(),
    k_features=(1, X_train.shape[1]),
    forward=True,
    scoring='r2',
    cv=5,
    verbose=0)
    
sfs_forward.fit(X_train, y_train)

print(f">> Forward SFS:")
print(f"   Best CV R²      : {sfs_forward.k_score_:.3f}")
print(f"   Optimal # feats : {len(sfs_forward.k_feature_idx_)}")
print(f"   Feature indices : {sfs_forward.k_feature_idx_}")
print(f"   Feature names   : {sfs_forward.k_feature_names_}")
```

You can see we ended up with the same three predictors as in best subset selection: `CWalks`, `Hits`, `HmRun`. However, this is not necessarily always the case — best subset and stepwise selection can, and often do, lead to different results. In our case we only had a small number of predictors, which makes it more likely to end up with the same subset.


### Backward Stepwise Selection

Backward Stepwise Selection begins with the full feature set and, at each step, removes the single feature whose exclusion most improves (or least harms) model performance. We keep repeating this “remove the worst feature” step until only the desired number of features remains, and then the algorithm returns that reduced subset:

```{code-cell} ipython3
sfs_backward = SequentialFeatureSelector(
    estimator=LinearRegression(),
    k_features=(1, X_train.shape[1]),
    forward=False,
    floating=False,
    scoring='r2',
    cv=5,
    verbose=0)

sfs_backward.fit(X_train, y_train)

print(f"<< Backward SFS:")
print(f"   Best CV R²      : {sfs_backward.k_score_:.3f}")
print(f"   Optimal # feats : {len(sfs_backward.k_feature_idx_)}")
print(f"   Feature indices : {sfs_backward.k_feature_idx_}")
print(f"   Feature names   : {sfs_backward.k_feature_names_}")
```

#### Summary

| Selection Method  | Finds best model? | Works for large p?        | Works for p>n?          | Computational cost         |
| ------------------| ------------------| --------------------------|-------------------------| ---------------------------|
| Best Subset       | **+** Yes         | **-** No                  | **-** No                | **-** Very high            |
| Forward Stepwise  | **-** No          | **+** Moderate/large p    | **o** If model size < n | **+** Efficient            |
| Backward Stepwise | **-** No          | **+** Yes (only if p < n) | **-** No                | **o** Relatively Efficient |


```{code-cell} ipython3
:tags: [remove-input]
from jupyterquiz import display_quiz
display_quiz('quiz/SubsetSelection.json')
```


#### What next?

Once we have identified the features that are relevant for predicting the outcome, let`s evaluate the model performance and estimate true test error with the thee predictors identified by Best Subset Selection and Forward Stepwise Seletion.

```{code-cell} ipython3
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

selected_features = list(sfs_backward.k_feature_names_)

# Subset the data
X_train_subset = X_train[selected_features]
X_test_subset = X_test[selected_features]

# Fit the model
model = LinearRegression()
model.fit(X_train_subset, y_train)

# Get predictions anf performance
y_pred = model.predict(X_test_subset)

mse_test = mean_squared_error(y_test, y_pred)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_pred)

print(f"Test MSE:  {mse_test:.2f}")
print(f"Test RMSE: {rmse_test:.2f}")
print(f"Test R²:   {r2_test:.4f}")
```

So in sum:

- On average, our predictions deviate from the actual salary by about 426 thousand dollars.
- Our model explains ~25% of the variance in salary.


### Regularization and Dimensionality Reduction

As mentioned before, regularization and dimensionality reduction are two other measures of dealing with large numbers of predictors. Regularization techniques will be introduced in the [next session](2_Regularization), and dimensionality reduction will be introduced in the [Principal Component Analysis](3_PCA_PCR) session.