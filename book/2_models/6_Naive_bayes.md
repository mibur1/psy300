---
short_title: Naïve Bayes
kernelspec:
  name: python3
  display_name: Python 3
---
# 💡 Naïve Bayes

Naïve Bayes classifiers, like LDA and QDA, are generative models. They aim to model how the data was generated for each class and use this knowledge to make predictions. The foundation of Naïve Bayes is Bayes’ Theorem.

## Bayes’ Theorem

Imagine you're trying to assess whether someone is likely to be experiencing anxiety based on a behavioural cue like nail biting. What you're really after is:

> What’s the probability that a person has anxiety, given that they bite their nails?

In probability notation, we can write this as:

$$
P(\text{Anxiety} \mid \text{Nail biting})
$$

This is called a *conditional probability*: it expresses how likely one event is (anxiety) given that another has occurred (nail biting).

Now, here’s the tricky part: we often cannot estimate this probability directly. Data are usually collected the other way round: we start from people with and without a diagnosis and record their behaviour. From such data, we know other things:

- What percentage of people in general are anxious: $P(\text{Anxiety})$
- Among those with anxiety, how common is nail biting: $P(\text{Nail biting} \mid \text{Anxiety})$
- How common is nail biting in the population overall: $P(\text{Nail biting})$

Bayes’ Theorem allows us to flip the conditional and compute the probability we care about:

$$
P(\text{Anxiety} \mid \text{Nail biting}) = \frac{P(\text{Nail biting} \mid \text{Anxiety}) \cdot P(\text{Anxiety})}{P(\text{Nail biting})}
$$

Since how common anxiety is differs between populations (e.g. a clinic vs. the general population), we can also plug in the base rate $P(\text{Anxiety})$ that fits our situation.

What we just did with anxiety and nail biting is exactly what a Naïve Bayes classifier does, just with more features and more classes. In general machine learning notation, we replace:

- “Anxiety” with a class label $Y = k$
- “Nail biting” (or more features) with a full observation $X = x$

This gives us the general form of Bayes’ Theorem used in classification:

$$
P(Y = k \mid X = x) = \frac{P(X = x \mid Y = k) \cdot P(Y = k)}{P(X = x)}
$$

Where:

- $P(Y = k \mid X = x)$: **Posterior** – the probability of class $k$ given the features $x$
- $P(X = x \mid Y = k)$: **Likelihood** – the probability of seeing those features if the class is $k$
- $P(Y = k)$: **Prior** – how frequent the class is in general
- $P(X = x)$: **Evidence** – the overall probability of seeing $x$ (same across all classes)

This is already everything we need to build classifiers that estimate the most likely class $Y$ based on input features $X$.


## The Naïve Assumption

To compute the likelihood term $P(X \mid Y = k)$, that is, how likely we are to observe a particular combination of features for a given class, we usually need a complex model that captures how all the features interact.

Naïve Bayes makes a simplifying assumption:

**All features are conditionally independent given the class.**

Let’s return to our example. Suppose you're trying to predict whether a person is anxious based on multiple behavioural features:

- Nail biting (NB)
- Fidgeting (FI)
- Avoiding eye contact (EC)

In reality, these features are probably not independent. For example, people who are fidgety might also tend to avoid eye contact. However, modelling all these interactions can become very complex, so Naïve Bayes says: 

> “Let’s assume that once we know whether someone is anxious or not, these behaviours don’t influence each other anymore.”

Mathematically, this means:

$$
P(\text{NB, FI, EC} \mid \text{Anxiety}) = P(\text{NB} \mid \text{Anxiety}) \cdot P(\text{FI} \mid \text{Anxiety}) \cdot P(\text{EC} \mid \text{Anxiety})
$$

This assumption is clearly naïve (we know behaviours are interrelated) but it simplifies things a lot, especially when we have many features. And surprisingly, this assumption often works well enough in practice to make useful predictions! 

So in general terms, for a feature vector $X = (X_1, X_2, \dots, X_p)$, the likelihood simplifies to:

$$
P(X = x \mid Y = k) = \prod_{j=1}^{p} P(X_j = x_j \mid Y = k)
$$

---

## The Algorithm

To understand how Naïve Bayes works in practice, we will walk through a simple version of the algorithm using categorical (discrete) features: Discrete Naïve Bayes. Here, likelihoods are calculated from frequency tables rather than continuous distributions.

### 1. Estimate priors

Priors are the class probabilities in the training data:

$$P(Y = k) = \text{Proportion of samples in class } k$$

If we assume our training data has 60% anxious (A) and 40% not anxious ($\neg$A) people, we have:

$$P(\text{A}) = 0.6$$
$$P(\neg\text{A}) = 0.4$$

### 2. Estimate class-conditional likelihoods

For each feature $X_j$, estimate the likelihood of observing $x_j$ given class $k$

$$P(X_j = x_j \mid Y = k)$$  

For our example, let us assume the following likelihoods in the training data:

| Feature              | $P(\cdot \mid \text{Anxious})$ | $P(\cdot \mid \text{Not Anxious})$ |
|----------------------|--------------------------------|------------------------------------|
| Nail Biting (NB=yes) | 0.8                            | 0.3                                |
| Fidgeting (FI=yes)   | 0.7                            | 0.2                                |
| Eye Contact (EC=no)  | 0.6                            | 0.4                                |

*Note: We use **EC = no** because we’re modelling *avoiding* eye contact.*

### 3. Compute the posterior

Using Bayes’ Theorem under the Naïve Bayes assumption:

$$
P(Y = k \mid X = x) \propto P(Y = k)\prod_{j=1}^{p} P(X_j = x_j \mid Y = k)
$$

Here our observed features are:

$$
x = (\text{NB=yes},\; \text{FI=yes},\; \text{EC=no})
$$

corresponding to *nail biting*, *fidgeting*, and *no eye contact*.

**Anxious (A):**

$$
P(Y=A \mid X=x) \propto 
P(A)\, P(NB=yes \mid A)\, P(FI=yes \mid A)\, P(EC=no \mid A)
$$

$$
= 0.6 \cdot 0.8 \cdot 0.7 \cdot 0.6
= 0.2016
$$

**Not Anxious ($\neg$A):**

$$
P(Y=\neg A \mid X=x) \propto 
P(\neg A)\, P(NB=yes \mid \neg A)\, P(FI=yes \mid \neg A)\, P(EC=no \mid \neg A)
$$

$$
= 0.4 \cdot 0.3 \cdot 0.2 \cdot 0.4
= 0.0096
$$

These two numbers are not yet probabilities, because we dropped the evidence $P(X = x)$. We call them the *scores* $S_A$ and $S_{\neg A}$.

### 4. Make a prediction

For the prediction, we can simply choose the class with the highest posterior probability:

$$\hat{Y} = \arg \max_k P(Y = k \mid X = x)$$

If we compare the posterior scores for our example, we have:

- Anxious: 0.2016  
- Not Anxious: 0.0096

Since $0.2016 > 0.0096$, the model predicts:

$$
\hat{Y} = \text{Anxious}
$$

For the classification, this is already enough. If we want actual probabilities, we divide each score by the sum of both scores, so that they add up to 1:

$$
P(\text{A} \mid X) = \frac{S_A}{S_A + S_{\neg A}} = \frac{0.2016}{0.2016 + 0.0096} = \frac{0.2016}{0.2112} \approx 0.955
$$

$$
P(\neg\text{A} \mid X) = \frac{S_{\neg A}}{S_A + S_{\neg A}} = \frac{0.0096}{0.2016 + 0.0096} = \frac{0.0096}{0.2112} \approx 0.045
$$


### Quiz

```{code-cell} ipython3
:tags: [remove-input]
from jupyterquiz import display_quiz
display_quiz("quiz/NaiveBayes.json", shuffle_answers=True)
```

```{dropdown} Solution

To solve this, we use the same priors and likelihoods as before, but change the feature values:

- NB = **no** → use $P(\text{NB=no} \mid Y) = 1 - P(\text{NB=yes} \mid Y)$
- FI = **yes**
- EC = **yes** → so $P(\text{EC=yes} \mid Y) = 1 - P(\text{EC=no} \mid Y)$

---

**For Anxious (A):**

$P(\text{A}) \cdot P(\text{NB=no} \mid \text{A}) \cdot P(\text{FI=yes} \mid \text{A}) \cdot P(\text{EC=yes} \mid \text{A}) =$
$0.6 \cdot (1 - 0.8) \cdot 0.7 \cdot (1 - 0.6) = 0.6 \cdot 0.2 \cdot 0.7 \cdot 0.4 = 0.0336$

**For Not Anxious ($\neg$A):**

$0.4 \cdot (1 - 0.3) \cdot 0.2 \cdot (1 - 0.4) = 0.4 \cdot 0.7 \cdot 0.2 \cdot 0.6 = 0.0336$

Both scores are equal, so the correct answer is **Tie**.

With continuous features, exact ties practically never happen. If they do, the result depends on the implementation: `sklearn` picks the class that comes first in `classes_`.
```

---

## Gaussian Naïve Bayes

In Gaussian Naïve Bayes, we assume that the features are **continuous** and that their values follow a **normal distribution** for each class.

This means that for each class $Y = k$ and each feature $X_j$:

$$
P(X_j = x_j \mid Y = k) = \frac{1}{\sqrt{2\pi\sigma_{jk}^2}} \exp\left( -\frac{(x_j - \mu_{jk})^2}{2\sigma_{jk}^2} \right)
$$

So the steps of Naïve Bayes for continuous data become:

1. Estimate class priors: $P(Y = k)$ from class proportions
2. Estimate means and variances $\mu_{jk}$ and $\sigma_{jk}^2$ for each feature and class
3. Plug into the Gaussian formula to compute likelihoods
4. Multiply likelihoods and prior, and choose the class with the highest posterior

This is essentially what `GaussianNB()` in scikit-learn does under the hood.

Compared to QDA, Gaussian Naïve Bayes only estimates one variance per feature and class, and no covariances between the features. With 100 features, that is 100 parameters per class instead of the 5050 of a full covariance matrix. This is why Naïve Bayes copes well with many features, even with few observations.

---

## Gaussian Naïve Bayes in Python

For a quick illustration, we use the same simulated data and the same split as in the LDA/QDA session:

```{code-cell} ipython3
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

# Generate synthetic data
X, y = make_classification(n_samples=400, n_features=2, n_informative=2, 
                           n_redundant=0, n_classes=2, n_clusters_per_class=1, 
                           random_state=5)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

We can then fit the model and evaluate it on the test set:

```{code-cell} ipython3
nb = GaussianNB()
nb.fit(X_train, y_train)

print(classification_report(y_test, nb.predict(X_test)))
```

With a test accuracy of 90%, Naïve Bayes lies between LDA (88%) and QDA (93%). Finally, we can plot the distributions the model learned for each class:

```{code-cell} ipython3
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sns.set_theme(style="darkgrid")
fig, ax = plt.subplots()

# Grid for likelihood computation
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xg = np.linspace(x_min, x_max, 300)
yg = np.linspace(y_min, y_max, 200)
xx, yy = np.meshgrid(xg, yg)
Xgrid = np.vstack([xx.ravel(), yy.ravel()]).T

# Plot the class densities learned by the model (theta_: means, var_: variances)
for label, color in enumerate(['blue', 'red']):
    mu, std = nb.theta_[label], np.sqrt(nb.var_[label])
    P = np.exp(-0.5 * (Xgrid - mu) ** 2 / std ** 2).prod(1)
    Pm = np.ma.masked_array(P, P < 0.03)
    ax.pcolormesh(xx, yy, Pm.reshape(xx.shape), shading='auto', alpha=0.4, cmap=color.title() + 's')
    ax.contour(xx, yy, P.reshape(xx.shape), levels=[0.01, 0.1, 0.5, 0.9], colors=color, alpha=0.2);

# Plot decision boundary
Z = nb.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax.contour(xx, yy, Z, levels=[0.5], linewidths=2, colors='black')

# Scatter plot
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=50, cmap='bwr')

# Legend
ax.set(xlabel="Feature 1", ylabel="Feature 2", title="Naïve Bayes Class Distributions")
legend_elements = [
    Line2D([], [], marker='o', linestyle='None', markerfacecolor='blue',
           markeredgewidth=0, label='Class 0', markersize=8),
    Line2D([], [], marker='o', linestyle='None', markerfacecolor='red',
           markeredgewidth=0, label='Class 1', markersize=8),
    Line2D([], [], color='black', linestyle='-', linewidth=2, label='Decision boundary')
]
ax.legend(handles=legend_elements, loc="upper left");  # explicit loc: "best" is slow on a dense mesh
```

This plot visualises how the Gaussian Naïve Bayes model estimates the class distributions:

- Each class is modelled as a Gaussian distribution with independent features
- Because there are no covariances, the ellipses are always aligned with the axes. Naïve Bayes cannot represent the tilt of the red class, which QDA captured
- The coloured contours represent density levels of the Gaussian likelihoods $P(X \mid Y = k)$
- The shaded background shows regions of higher probability under each class
- The decision boundary between the two classes lies where the posterior probabilities are equal

---

## Summary

LDA, QDA, and Naïve Bayes all use Bayes' theorem in the same way. They only differ in how they model the distribution of the features within each class, $P(X \mid Y = k)$:

| | LDA | QDA | Gaussian Naïve Bayes |
|---|---|---|---|
| Distribution within each class | Multivariate Gaussian | Multivariate Gaussian | Independent 1D Gaussians |
| Covariances | One matrix shared by all classes | One matrix per class | None, only one variance per feature and class |
| Parameters for 10 features and 2 classes | 2 × 10 means + 1 × 55 (co)variances = 75 | 2 × 10 means + 2 × 55 (co)variances = 130 | 2 × 10 means + 2 × 10 variances = 40 |
| Decision boundary | Linear | Quadratic | Quadratic, like QDA, but cannot be tilted |
| Works well when | The classes have a similar spread | There are many observations per class | There are many features (e.g. text data) |

Gaussian Naïve Bayes is therefore QDA with all covariances between the features set to zero. The name refers to this "naïve" independence assumption, not to the use of Bayes' theorem, which LDA and QDA rely on just as much. The assumption has two advantages:

1. **Fewer parameters:** Without covariances, the number of parameters grows linearly with the number of features instead of quadratically. With many features and few observations, QDA cannot estimate its covariance matrices reliably, while Naïve Bayes still works. As usual, we trade more bias for less variance.
2. **Any type of feature:** Because each feature is modelled on its own, it does not have to be Gaussian. Yes/no features (as in the anxiety example), counts (e.g. word frequencies in a text), and continuous measures can in principle even be combined in one model. scikit-learn offers separate variants for the different types, such as `CategoricalNB` for categories and `MultinomialNB` for counts.

That's it! You can now head to [Exercise 5](Exercises.ipynb) to apply LDA, QDA, and Naïve Bayes yourself.
