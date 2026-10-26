---
short_title: Resampling
kernelspec:
  name: python3
  display_name: Python 3
---

# 🎲 Resampling Strategies

As future data scientists, you probably know how expensive data collection is. Time, money, and experimental design all limit how much data we get, so we cannot afford to set aside a large part of it just for testing. At the same time, we need to know how well a model will do on new data.

Resampling methods solve this by splitting the same data many times into training and test sets and combining the results. You already saw why this is necessary in the bias-variance exercise: depending on the random train/test split, the test MSE of the very same model varied between 0.20 and 0.40.

```{hint} Resampling strategies
The two most common resampling methods are:

- *Cross-validation*: split the data into folds and use each fold once as the test set
- *Bootstrapping*: draw samples with replacement, so the same observation can appear several times
```

## The data

We will use the famous [Iris](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html) dataset, which contains 150 samples from three species of the iris plant (iris setosa, iris virginica and iris versicolor). The data contains four features: the length and the width of the sepals and petals (in centimeters).

```{figure} https://miro.medium.com/v2/resize:fit:1400/format:webp/0*oUoXifiKu3tT5REt.png
:name: fig-iris
:alt: The three iris species setosa, versicolor and virginica, with petals and sepals labelled
:align: center

The three iris species in the dataset. Image source: [@naidubhavya06 on Medium](https://medium.com/@naidubhavya06/detailed-explanation-of-exploratory-data-analysis-using-iris-dataset-fa8082a4ce0f).
```

```{code-cell} ipython3
import seaborn as sns
import pandas as pd
from sklearn import datasets

# Get data
iris = datasets.load_iris(as_frame=True)
df = iris.frame
df['class'] = pd.Categorical.from_codes(iris.target, iris.target_names)

df.describe()
```

```{code-cell} ipython3
sns.scatterplot(data=df, x='sepal length (cm)', y='sepal width (cm)', hue="class");
```

The goal of our model is to predict the species of a flower from the two features shown in the plot (sepal length and width). What kind of problem is this?

```{code-cell} ipython3
:tags: [remove-input]
import sys; sys.path.append("..")
from quizpager import display_quiz_paged
display_quiz_paged('quiz/iris.json')
```

```{important} The rows are sorted
The rows of `iris` are **sorted by species**: the first 50 rows are setosa, the next 50 versicolor, and the last 50 virginica. Keep this in mind, it will cause problems later in this chapter.
```

## Validation Sets

:::{margin}
Hyperparameters are parameters that are not learned from the data but set by the researcher before the training process.
:::

The simplest resampling strategy is to split the dataset into two parts:

- *Training set*: used to fit the model
- *Validation set*: used to evaluate the model, e.g. to compare different models or hyperparameters

```{figure} figures/ValidationSet.drawio.png
:name: fig-validation-set
:alt: Validation set approach
:align: center

The validation set approach splits the dataset into a training and a validation set.
```

```{note} Validation set or test set?
The two terms are often mixed up, and scikit-learn's `train_test_split()` calls both of them a test set. Strictly speaking:

- a *validation set* is used to choose between models or hyperparameters, and can be used many times
- a *test set* is used only once at the very end, to report the performance of the final model

If you pick the model with the best test set score, the test set has turned into a validation set, and the reported score will be too optimistic. We will come back to this in the model selection session.
```

The training and validation set neither need to be of equal size nor do they need to be contiguous blocks in the data. `train_test_split()` shuffles the rows before splitting, so the sorted species are not a problem here. Let's try the validation set approach on the `Iris` data:

1. Define features and target data

```{code-cell} ipython3
# Features: sepal length and width; target: type of flower
X = df[["sepal length (cm)", "sepal width (cm)"]]
y = df["target"]
```

2. Split the data into training and validation samples

```{code-cell} ipython3
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
```

3. Fit the model (we use a support vector classifier, which you will learn about later in the seminar)

```{code-cell} ipython3
from sklearn import svm

model = svm.SVC(kernel='linear')
fit = model.fit(X_train, y_train)
```

4. Evaluate model performance

```{code-cell} ipython3
fit.score(X_test, y_test)
```

The `score()` method returns the accuracy of our predictions. In this case, our algorithm correctly predicted the species of the flower in 85% of cases.

```{code-cell} ipython3
:tags: [remove-input]

from jupytercards import display_flashcards
display_flashcards('quiz/validation_set.json');
```

### How much does that 85% actually mean?

The number above came out of exactly one split, chosen by `random_state=42`. There is nothing special about 42, so what happens if we repeat the same procedure with 500 different random splits?

```{code-cell} ipython3
:tags: [hide-input]

import numpy as np
import matplotlib.pyplot as plt

split_scores = []
for seed in range(500):
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.4, random_state=seed)
    split_scores.append(svm.SVC(kernel='linear').fit(X_tr, y_tr).score(X_te, y_te))

split_scores = np.array(split_scores)

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(split_scores, bins=25, color="#4c72b0", alpha=0.8)
ax.axvline(split_scores.mean(), color="k", ls="--",
           label=f"mean = {split_scores.mean():.3f}")
ax.axvline(split_scores[42], color="crimson", lw=2,
           label=f"random_state=42 → {split_scores[42]:.3f}")
ax.set(xlabel="Validation accuracy", ylabel="Number of splits",
       title="Accuracy of 500 different validation-set splits")
ax.legend()
plt.tight_layout()

print(f"mean  {split_scores.mean():.3f}")
print(f"std   {split_scores.std():.3f}")
print(f"range {split_scores.min():.3f} – {split_scores.max():.3f}")
```

The spread is large. Depending only on the random seed, the same model reaches anywhere between 63% and 92% accuracy. Our 85% was a lucky split, and a single split should never be reported as *the* performance of a model.

**Try it yourself:** the split *ratio* matters too. Is it better to train on 80% of the data and validate on 20%, or the other way round? Think about what you expect, then complete the code below with the test sizes `0.2`, `0.5` and `0.8` and run it. To keep luck out of the comparison, the code averages over 200 random splits for each ratio:

```{code-block} python
import numpy as np
from sklearn import datasets, svm
from sklearn.model_selection import train_test_split

# Load the data: sepal length and width as features, species as target
df = datasets.load_iris(as_frame=True).frame
X = df[["sepal length (cm)", "sepal width (cm)"]]
y = df["target"]

test_sizes = ...  # TODO: fill in the test sizes as a list

for test_size in test_sizes:
    accuracies = []
    for seed in range(200):
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=seed)
        accuracies.append(svm.SVC(kernel='linear').fit(X_tr, y_tr).score(X_te, y_te))

    print(f"train on {1 - test_size:.0%} / validate on {test_size:.0%}   ->   mean accuracy = {np.mean(accuracies):.3f}, std = {np.std(accuracies):.3f}")
```

:::{dropdown} Show answer
With more training data, the model is slightly better on average. With only 20% of the data for validation, however, the accuracy varies most between splits, because each estimate is based on just 30 flowers. The validation set approach cannot escape this tradeoff: every observation we use for validation is missing from training, and the other way round.
:::

```{hint} Summary
The validation set approach is quick and easy. Its weakness is that the result depends on a single split, which can make a good model look bad or a weak model look good.
```

## Cross-Validation (CV)

### K-fold CV

To get a more reliable estimate, we split the data several times and use every observation for validation exactly once.

In k-fold CV, the dataset is divided into $k$ equally sized **folds**. In each round, one fold is the validation set and the remaining $k-1$ folds form the training set. This is repeated $k$ times, so that every fold is used for validation once. The average accuracy across the $k$ folds is our estimate of the model's performance.

```{figure} figures/CV.drawio.png
:name: fig-cv
:alt: Cross validation
:align: center

K-fold cross-validation splits the dataset into $k$ equally sized folds. Each fold is used once as the validation set, while the model is trained on the remaining $k-1$ folds.
```

Let's try it on our data:

```{code-cell} ipython3
from sklearn.model_selection import KFold, cross_val_score

k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
model = svm.SVC(kernel='linear')

scores = cross_val_score(model, X, y, cv=k_fold)

print(f"Average accuracy:      {scores.mean():.3f}")
print(f"Individual accuracies: {np.round(scores, 3)}")
```

:::{warning} Shuffle your data
`KFold` splits the rows **in the order they appear**, unless you set `shuffle=True`. Since the iris rows are sorted by species, each validation fold would then contain only one or two species, and these species would be underrepresented in the training folds.
:::

Let's see what happens without shuffling:

```{code-cell} ipython3
scores_unshuffled = cross_val_score(model, X, y, cv=KFold(n_splits=5))
print(f"Without shuffling: {np.round(scores_unshuffled, 3)} → mean {scores_unshuffled.mean():.3f}")
```

The third fold contains only versicolor and the fifth fold only virginica flowers, and their accuracy drops to 0.3 and 0.2. The mean of 0.61 is caused by the row order alone. Whenever the row order of your data has some structure (sorted by group, recorded by session, ordered in time), shuffling or a stratified splitter matters more than the choice of $k$.

For classification, it is usually even better to use `StratifiedKFold`, which keeps the class proportions the same in every fold. If you pass a plain integer to `cross_val_score`, scikit-learn does this automatically:

```{code-cell} ipython3
# cv=5 with a classifier → StratifiedKFold under the hood
scores_stratified = cross_val_score(svm.SVC(kernel='linear'), X, y, cv=5)
print(f"Average accuracy: {scores_stratified.mean():.3f}")
```

scikit-learn offers many more splitters for different kinds of data. The [scikit-learn user guide on cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) gives a good overview, with a visualization of each splitter.

If we are interested in the individual models, we can also run the training and evaluation explicitly, which allows us to save them:

```{code-cell} ipython3
from sklearn.base import clone

base_model = svm.SVC(kernel='linear')
score_list = []
model_list = []

for train_index, test_index in k_fold.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]  # iloc because X is a DataFrame
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]  # iloc because y is a Series

    model = clone(base_model)  # create a new copy of the model for every iteration
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    score_list.append(score)
    model_list.append(model)

print(f"Accuracy per fold: {np.round(score_list, 3)}")
```

:::{warning} Which model do we use in the end?
In the loop above, we fitted the same type of model five times, once per fold, so `model_list` contains five slightly different models. It is tempting to keep the one from the first fold, since it has the highest accuracy. However, its higher score only means that its validation fold happened to contain easier flowers.

Cross-validation tells us how well this type of model performs on new data. In practice, we usually run it for several candidate models (e.g. different algorithms or hyperparameters) and pick the one with the best average score. The chosen model is then fitted once more on the complete dataset. You will do this in the model selection session.
:::

```{note} Validation set vs. k-fold
Both approaches give similar results here (0.85 vs. 0.81). The single validation split is one draw from the wide distribution we plotted above, while the k-fold estimate averages over five folds and therefore depends much less on luck. Both estimate the same quantity, so a small difference between them does not mean that one of them is wrong.
```

```{note} Choosing k
A higher $k$ trains each model on more data, so the estimate is less pessimistic. On the other hand, it needs more model fits, and each validation fold gets smaller. In practice, $k=5$ or $k=10$ are the usual choices.
```

**Try it yourself:** complete the code below with $k$ = `2`, `5`, `10`, `20` and `50` folds and run it. What do you think is a good tradeoff?

```{code-block} python
from sklearn import datasets, svm
from sklearn.model_selection import KFold, cross_val_score

# Load the data: sepal length and width as features, species as target
df = datasets.load_iris(as_frame=True).frame
X = df[["sepal length (cm)", "sepal width (cm)"]]
y = df["target"]

ks = ...  # TODO: fill in the numbers of folds as a list

for k in ks:
    cv = KFold(n_splits=k, shuffle=True, random_state=42)
    s = cross_val_score(svm.SVC(kernel='linear'), X, y, cv=cv)
    print(f"k = {k:>2}   mean accuracy = {s.mean():.3f}   "
          f"std across folds = {s.std():.3f}   ({k} model fits)")
```

:::{dropdown} Show answer
The mean barely changes once $k \ge 5$, while the standard deviation across folds keeps growing. With more folds, each validation fold is smaller, so the individual fold scores get noisier, even though their average stays stable. Beyond $k = 5$ or $10$, the extra model fits buy you very little.
:::

### Leave-one-out CV (LOOCV)

LOOCV is a special case of k-fold cross-validation, where $k$ equals the number of observations. The model is trained on all but one observation and validated on the remaining one. This is repeated for every observation, so each one is used for validation exactly once.

LOOCV gives an almost unbiased estimate, but it needs one model fit per observation. Also, each fold score can only be 0 or 1, because there is just a single observation to classify. In code, we only need to change the splitter from `KFold()` to `LeaveOneOut()`:

```{code-cell} ipython3
from sklearn.model_selection import LeaveOneOut

model = svm.SVC(kernel='linear')
loocv = LeaveOneOut()

scores = cross_val_score(model, X, y, cv=loocv)

print(f"Average accuracy:      {scores.mean():.3f}")
print(f"Individual accuracies: {np.unique(scores)}  (only 0 or 1 possible)")
print(f"Number of fits:        {len(scores)}")
```

## Bootstrapping

Bootstrapping estimates how much a result would vary if we had collected a different dataset. Instead of a single training set, we create many new datasets by sampling with replacement from the original data. Each bootstrap sample has the same size as the original dataset, but some observations appear more than once, while others do not appear at all.

For each bootstrap iteration:

1. A new sample (the bootstrap sample) is drawn from the data.
2. The model is trained on this bootstrap sample.
3. The observations that were not drawn, called out-of-bag (OOB) samples, are used to test the model.

Cross-validation is usually the better choice to estimate predictive performance. Bootstrapping is mostly used to quantify uncertainty, for example of model parameters or of a performance estimate.

Let's run 200 bootstrap iterations and look at the distribution of the OOB accuracies:

```{code-cell} ipython3
from sklearn.utils import resample

n = len(X)
oob_scores = []

for i in range(200):
    # Draw n row indices with replacement
    boot_idx = resample(np.arange(n), replace=True, n_samples=n, random_state=i)

    # Out-of-bag samples: all rows that were not drawn
    oob_idx = np.setdiff1d(np.arange(n), boot_idx)

    model = svm.SVC(kernel='linear')
    model.fit(X.iloc[boot_idx], y.iloc[boot_idx])
    oob_scores.append(model.score(X.iloc[oob_idx], y.iloc[oob_idx]))

lower, upper = np.percentile(oob_scores, [2.5, 97.5])
print(f"Mean OOB accuracy:   {np.mean(oob_scores):.3f}")
print(f"Middle 95% of runs:  {lower:.3f} – {upper:.3f}")

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(oob_scores, bins=20, color="#4c72b0", alpha=0.8)
ax.axvline(np.mean(oob_scores), color="k", ls="--", label="mean")
ax.set(xlabel="OOB accuracy", ylabel="Number of bootstrap samples")
ax.legend()
plt.tight_layout()
```

The mean OOB accuracy is very close to the 5-fold estimate (0.81), because both are computed on data the model did not see during training. The histogram adds what a single number cannot show: depending on which flowers end up in the training data, the accuracy varies roughly between 69% and 88%.

```{note} Is bootstrapping too optimistic?
You will sometimes read that bootstrapping is optimistic because training and test data overlap. This does not apply to the OOB estimate, since an out-of-bag observation is by definition not part of the bootstrap sample the model was trained on.

If anything, the OOB estimate is slightly **pessimistic**. A bootstrap sample of size $n$ contains only about $1 - (1 - 1/n)^n \approx 63\%$ distinct observations, so each model is trained on less data than in 5-fold CV (80%).
```

```{code-cell} ipython3
:tags: [hide-input]

# How many distinct observations does a bootstrap sample actually contain?
unique_fractions = [
    len(np.unique(resample(np.arange(n), replace=True, n_samples=n, random_state=i))) / n
    for i in range(200)
]
print(f"Mean fraction of distinct observations: {np.mean(unique_fractions):.3f}")
print(f"Theoretical value 1 - (1 - 1/n)^n:      {1 - (1 - 1/n)**n:.3f}")
```

```{tip} Summary
- A single train/test split gives a noisy estimate, so don't report it as *the* performance of your model.
- k-fold CV uses every observation for validation once and is the default choice for estimating performance and comparing models.
- Shuffle or stratify when the row order of your data has structure.
- Bootstrapping shows how much a result varies across resampled datasets and is mostly used to quantify uncertainty.

To see how the different scikit-learn splitters divide the data, have a look at the [visualization of cross-validation behavior](https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html) in the scikit-learn documentation.
```

Test your understanding of this session:

```{code-cell} ipython3
:tags: [remove-input]
display_quiz_paged('quiz/resampling.json')
```
