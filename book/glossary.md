---
short_title: Glossary
---

# 📖 Glossary & Cheat Sheet

A place to look things up when a symbol or a scikit-learn argument stops making sense mid-chapter.

## Core vocabulary

Feature / predictor / independent variable
: An input column of the design matrix $X$. The three names are used interchangeably in this book, depending on whether the sentence is written from a machine-learning or a statistics perspective.

Target / response / label / dependent variable
: The quantity $y$ we want to predict. *Label* is normally reserved for classification.

Training set
: The data used to estimate the model parameters.

Test set
: Data held back and touched only once, at the very end, to obtain an unbiased estimate of generalisation performance.

Validation set
: Data used *during* model development to compare models or tune hyperparameters. Because you look at it repeatedly, it stops being an unbiased estimate of the test error.

Parameter
: A quantity learned from the data, e.g. the coefficients $\beta$ of a regression.

Hyperparameter
: A quantity fixed by you before training, e.g. the regularisation strength $\lambda$, the number of folds $k$, or a tree's `max_depth`.

Bias
: Error caused by a model being too rigid to represent the true relationship. See [](1_basics/2_bias_variance.md).

Variance
: Sensitivity of the fitted model to the particular training sample that was drawn.

Irreducible error
: Noise in the data itself. No model can go below it.

Overfitting
: Fitting structure that is specific to the training sample and does not generalise. Low training error, high test error.

Underfitting
: The model is not flexible enough to capture the real structure. High training *and* test error.

Discriminative model
: Models $P(Y \mid X)$ directly (logistic regression, SVM, trees).

Generative model
: Models $P(X \mid Y)$ and $P(Y)$, then applies Bayes' theorem (LDA, QDA, Naïve Bayes).

Kernel
: A function that computes inner products in an implicitly higher-dimensional space, letting a linear method draw non-linear boundaries.

## Symbols used throughout

| Symbol | Meaning |
|---|---|
| $n$ | number of observations |
| $p$ | number of predictors |
| $k$ | number of folds in cross-validation (also number of classes in some chapters) |
| $X$ | design matrix, shape $n \times p$ |
| $y$ | target vector, length $n$ |
| $\beta$ | regression coefficients |
| $\hat{f}$ | the estimated function |
| $\hat{y}$ | predicted values |
| $\varepsilon$ | error / noise term |
| $\sigma^2$ | noise variance (the irreducible error) |
| $\lambda$ | regularisation strength (called `alpha` in scikit-learn) |
| $\alpha$ | elastic net mixing parameter (called `l1_ratio` in scikit-learn) |
| $\pi_k$ | prior probability of class $k$ |
| $\mu_k$ | mean vector of class $k$ |
| $\Sigma$ | covariance matrix |
| $\delta_k$ | discriminant function for class $k$ |
| $\eta$ | learning rate in boosting |

## Metrics

| Metric | Formula | Used for |
|---|---|---|
| MSE | $\frac{1}{n}\sum (y_i - \hat{y}_i)^2$ | regression |
| RMSE | $\sqrt{\text{MSE}}$ | regression, in the units of $y$ |
| R² | $1 - \frac{\sum(y_i-\hat y_i)^2}{\sum(y_i-\bar y)^2}$ | regression |
| Accuracy | $\frac{TP+TN}{n}$ | classification (balanced classes) |
| Precision | $\frac{TP}{TP+FP}$ | "when I say positive, how often am I right?" |
| Recall | $\frac{TP}{TP+FN}$ | "of all real positives, how many did I catch?" |
| F1 | $2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}$ | classification with imbalanced classes |

## Naming traps in scikit-learn

```{warning} Where the names do not match the lecture
- **`alpha` is $\lambda$.** In `Ridge`, `Lasso` and `ElasticNet`, `alpha` is the regularisation *strength*. The elastic net *mixing* parameter, which the lecture calls $\alpha$, is `l1_ratio`.
- **`C` is inverted.** In `SVC`, a *small* `C` gives a *wide* margin and more violations. This is the opposite direction from the budget parameter in the lecture.
- **`KFold` does not shuffle by default.** If your rows are sorted by group or class, you must pass `shuffle=True` (see [](1_basics/3_resampling.md)).
- **`cross_val_score(model, X, y, cv=5)` silently stratifies** when `model` is a classifier, but `cv=KFold(5)` does not.
- **`fit_transform` on test data is a bug.** Scalers and PCA are fitted on the training data only, then `transform` is applied to the test data.
- **`predict_proba` returns one column per class.** For binary problems, the probability of class 1 is column `[:, 1]`.
```

## The standard workflow

Almost every chapter is a variation on the same five steps:

1. **Split** — hold out a test set before doing anything else.
2. **Preprocess** — fit scalers/encoders on the training data only, ideally inside a `Pipeline`.
3. **Select** — compare models and hyperparameters with cross-validation on the training data.
4. **Fit** — refit the chosen model on the full training data.
5. **Evaluate** — score once on the held-out test set and report that number.

```{tip} Use a Pipeline
Wrapping the preprocessing and the model into `make_pipeline(StandardScaler(), Ridge())` makes steps 2 and 3 leak-proof by construction: the scaler is refitted inside every CV fold automatically.
```
