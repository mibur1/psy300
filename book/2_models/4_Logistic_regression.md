---
short_title: Logistic regression
kernelspec:
  name: python3
  display_name: Python 3
---
# 📊 Logistic Regression

So far, we have predicted continuous outcomes. Today, we turn to *classification*, where the outcome is a category. Our example is a dataset of 70 children between 24 and 83 months old. For each child, we know the age in months, whether the child has a theory of mind (`TOM`, 0/1), and whether the child understands display rules (`display`, 0/1). Understanding display rules means knowing that you do not always have to show the emotion you actually feel. Our research question is whether this understanding can be predicted from age.

## Why Not Linear Regression?

Before we use logistic regression, let's see what happens if we simply fit a linear regression to the data:

```{code-cell} ipython3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("data.dat", delimiter='\t')

fig, ax = plt.subplots()
ax.set_xlim(0, 100)
sns.regplot(x="age", y="display", data=df, ax=ax, truncate=False) # truncate=False extends the line to the x-axis limits
ax.set(xlabel="Age [month]", ylabel="Understanding display rules")

plt.show()
```

We could read the fitted line as a predicted probability. Can you think of problems with this approach for a binary outcome (0/1)?

:::{dropdown} Problems with linear regression for a binary outcome
- The model can produce predicted values below 0 or above 1, which are impossible probabilities. In our data, the line already predicts 1.02 for the oldest child, and it would fall below 0 for children younger than about 6 months.
- The variance of a 0/1 outcome is $p(1-p)$, so it changes whenever the probability $p$ changes with age. This violates the homoscedasticity assuption (constant variance of errors) of linear regression.
:::


## Logistic Regression with Python

In this example, we will use the `LogisticRegression()` class from `scikit-learn` for modeling.

```{code-cell} ipython3
import numpy as np
from sklearn.linear_model import LogisticRegression

# Convert 'age' into a NumPy array and reshape it to a 2D array (sklearn requirement)
# .reshape(-1, 1): Creates one column with as many rows as needed (-1 infers the row count)
X = np.asarray(df['age']).reshape(-1, 1)
print(X.shape)

# Convert 'display' to a NumPy array for the binary outcome
y = np.asarray(df['display']) # binary outcome

model = LogisticRegression()
model.fit(X, y)

print(f"Intercept: {model.intercept_}")
print(f"Coefficients: {model.coef_}")
```

```{note}
`LogisticRegression()` applies a Ridge penalty by default, just like the one you know from the regularization session. As before, the penalty depends on the scale of the predictors, so you should standardize them if they have different units. With a single predictor and the default `C=1`, the effect is negligible here.
```

### Interpreting the Model Outputs: Logits

The logistic regression still gives us an intercept and a slope, but they operate on the log-odds scale, not the probability scale:

- **Intercept:** The expected logit (log-odds) of the outcome (understanding display rules) when age = 0.
- **Slope/coefficient:** The change in the logit of understanding display rules for each one-month increase in age.

**But what even are logits?**

A linear model $\beta_0 + \beta_1X_1 + \dots + \beta_kX_k$ can produce any number, from minus infinity to plus infinity. A probability $P$, however, has to stay between 0 and 1. Logistic regression solves this by transforming the probability in two steps, so that it can take any value as well:

1. **Odds** $= \frac{P}{1-P}$: how many times more likely the event is to occur than not. A probability of 0.75 gives odds of 3. Odds have no upper limit, but they cannot be negative.
2. **Logit** $= \ln(\text{odds})$: the logarithm turns odds below 1 into negative numbers, so the logit has no lower limit either.

:::{div}
:class: centered-table
| Probability | $\to 0$ | 0.01 | 0.25 | 0.5 | 0.75 | 0.99 | $\to 1$ |
|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Odds | $\to 0$ | 0.01 | 0.33 | 1 | 3 | 99 | $\to \infty$ |
| Logit | $\to -\infty$ | −4.6 | −1.1 | 0 | 1.1 | 4.6 | $\to \infty$ |
:::

Now the logit and the linear model have the same range, so we can set them equal:

$$\text{Logit}(P) = \ln\left(\frac{P}{1-P}\right) = \beta_0 + \beta_1X_1 + \beta_2X_2 + \dots + \beta_kX_k$$

Unlike in linear regression, the coefficients are not found by least squares. They are estimated with *maximum likelihood*: we choose the values of $\beta$ that make the observed outcomes as probable as possible.

If we plot the equation, we get a straight line on the logit scale:

```{code-cell} ipython3
ages = np.linspace(0, 100, 100)
logits = model.intercept_[0] + model.coef_[0, 0] * ages

fig, ax = plt.subplots()
ax.plot(ages, logits, color="red", label="Model")
ax.axhline(0, color="gray", linestyle="--", label="Logit = 0 (probability = 0.5)")
ax.set(xlabel="Age [month]", ylabel="Logit (Log-Odds)", title="The model on the logit scale")

plt.legend()
plt.show()
```

A logit is hard to interpret directly, so we often use the *odds ratio* $e^{\beta}$ instead. Here, $e^{0.066} \approx 1.07$: with every additional month of age, the odds of understanding display rules are multiplied by 1.07.

### From Logits to Probabilities

We can transform the logits back into probabilities (more specifically the conditional probability of an observation belonging to class 1 given the predictors):

$$P(Y=1 \mid X) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 X)}}$$

To see what this looks like, we create an evenly spaced array of values for the age and use `model.predict_proba()` to get the predicted probability for each value:

```{code-cell} ipython3
# Create an evenly spaced array of values for the range 
x_range = np.linspace(1, 100, 100).reshape(-1, 1) 

# Predict probability of class 1 for each value in the range
y_prob = model.predict_proba(x_range)
y_prob = y_prob[:,1] # only get the second column

# Plot the results
fig, ax = plt.subplots()
ax.scatter(X, y, color='blue', label='data points', alpha=0.5)
ax.plot(x_range, y_prob, color='red', label='model predictions')
ax.axhline(0.5, color='gray', linestyle='--', label='probability = 0.5')
ax.set(xlabel='age', ylabel='display', title='Logistic Regression Model')

plt.legend()
```

The predictions now stay between 0 and 1. The effect of age on the probability is not constant: one month makes a larger difference in the middle of the curve than at its ends. The predicted probability is 0.39 at 36 months and 0.76 at 60 months.

To turn a probability into a class, `model.predict()` uses a threshold of 0.5. A probability of 0.5 corresponds to a logit of 0, which our model reaches at about 43 months. Older children are predicted to understand display rules, younger children are not.


## Model Evaluation

To evaluate our model, we can, for example, examine how many values of $y$ were predicted correctly:

```{code-cell} ipython3
predictions = model.predict(X)
accuracy = model.score(X, y)

print("Model predictions:", predictions)
print("\nAccuracy:", accuracy) 
```

The model correctly classifies about 77% of the children (54 of 70). To judge this number, we need a point of comparison: 39 of the 70 children understand display rules, so always predicting this majority class would already reach 56%. For a more detailed look, we can use a confusion matrix:

```{code-cell} ipython3
from sklearn.metrics import confusion_matrix

confusion_matrix(y, model.predict(X))
```

The output of the confusion matrix provides the following values:

|                     | Predicted Negative  | Predicted Positive  |
|---------------------|-------------------- |---------------------|
| **Actual Negative** | True Negative (TN)  | False Positive (FP) |
| **Actual Positive** | False Negative (FN) | True Positive (TP)  |

The model therefore correctly identifies 24 children without and 30 children with an understanding of display rules. It wrongly assigns 7 children to class 1 and 9 children to class 0. For more detail, we can print the classification report:

```{code-cell} ipython3
from sklearn.metrics import classification_report

report = classification_report(y, model.predict(X))
print(report)
```

The output can be interpreted as follows:

**Precision**: Proportion of correct predictions among all predictions for that class.

$$\text{Precision} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Positives (FP)}}$$

- *Class 0: when the model predicts that a child does not understand the display rules, it is correct 73% of the time.*
- *Class 1: when the model predicts that a child does understand the display rules, it is correct 81% of the time.*


**Recall**: Proportion of actual samples of a class that the model correctly identifies.

$$\text{Recall} = \frac{\text{True Positives (TP)}}{\text{True Positives (TP)} + \text{False Negatives (FN)}}$$

- *Class 0: 77% of the children who do not understand the display rules are correctly identified by the model.*
- *Class 1: 77% of the children who do understand the display rules are correctly identified by the model.* 


**F1-Score**: harmonic mean of precision and recall, which summarizes both in a single number.

$$F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$

- *It is 0.75 for class 0 and 0.79 for class 1, so the model performs similarly for both classes.*

**Support**: the actual number of observations of each class in the dataset

**Accuracy**: The overall proportion of correctly predicted observations.

$$\text{Accuracy} = \frac{TP + TN}{\text{Total number of observations}}$$

```{tip} Changing the threshold
The threshold of 0.5 is only a default. If one type of error is worse than the other, for example missing a disease, we can use `predict_proba()` and classify a case as positive already at a lower probability. This increases the recall for that class, but usually lowers its precision.
```

So far, we have evaluated the model on the same data we used to fit it, which is too optimistic. As in the previous sessions, we can get a better estimate with cross-validation. The rows of our data are roughly ordered by age, so we shuffle them, and we use stratified folds to keep the proportion of the two classes similar in each fold:

```{code-cell} ipython3
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(LogisticRegression(), X, y, cv=cv)

print(f"Cross-validated accuracy: {scores.mean():.3f}")
```

The cross-validated accuracy of 76% is close to the training accuracy of 77%. With a single predictor, the model has very little room to overfit.


## Multiple Logistic Regression

You may want to use two or more variables as inputs for the regression. In our example, we will use `age` and `TOM` as predictors for `display` by simply adding them to $X$.

```{code-cell} ipython3
X_multi = df[['age', 'TOM']]

model_multi = LogisticRegression()
model_multi.fit(X_multi, y)

print("Coefficients:", model_multi.coef_)
print("Odds ratios:", np.exp(model_multi.coef_))
print(classification_report(y, model_multi.predict(X_multi)))

scores_multi = cross_val_score(LogisticRegression(), X_multi, y, cv=cv)
print(f"Cross-validated accuracy: {scores_multi.mean():.3f}")
```

As in multiple linear regression, each coefficient describes the effect of a predictor while the other predictors are held constant. Children of the same age have about 4.4 times higher odds of understanding display rules if they have a theory of mind. The effect of age is now smaller (an odds ratio of 1.04 instead of 1.07), because age and theory of mind are correlated (r = 0.61).

The cross-validated accuracy increases from 76% to 79%. With 70 children, this is a difference of only about two children, so we should not read too much into it.

## Summary

* Logistic regression models the log-odds of an outcome as a linear function of the predictors, which keeps the predicted probabilities between 0 and 1.
* The coefficient $\beta$ is the change in the log-odds per unit of the predictor, and $e^{\beta}$ is the factor by which the odds change.
* Accuracy alone can be misleading. Compare it to the majority class, and look at the confusion matrix, precision, and recall.

```{code-cell} ipython3
:tags: [remove-input]
import sys; sys.path.append("..")
from quizpager import display_quiz_paged
display_quiz_paged('quiz/Logistic.json')
```

**Now it's your turn:** head to the [exercise section](Exercises.ipynb) for Exercise 4, in which you will classify breast tumors as benign or malignant.
