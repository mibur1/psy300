---
short_title: LDA & QDA
kernelspec:
  name: python3
  display_name: Python 3
---
# ➗ LDA & QDA

Logistic regression directly models how the probability of a class changes with the features, $P(Y \mid X)$. It is a **discriminative model**. The decision boundary (for example where both classes are equally likely) follows directly from these probabilities.

**Generative models** such as Linear Discriminant Analysis (LDA), Quadratic Discriminant Analysis (QDA), and Naïve Bayes (which will be introduced in the next session) take a different approach. They first model how the features are *distributed* within each class. To classify a new observation, they then use Bayes' theorem to check which class makes it most likely.

```{tip} Generative models

- Learn the distribution of features for each class, not just how to separate them
- Use this information to calculate the probability that a new observation belongs to each class
- Can generate new samples by sampling from the learned distributions
```

## Linear Discriminant Analysis (LDA)

LDA assumes that:

- Within each class, the features follow a multivariate Gaussian distribution
- All classes share the same covariance matrix

As a visual intuition in a 2D case (2 predictors), this means the class distributions look like ellipses with the same shape and orientation (but centered at different locations if the classes differ). In detail, LDA requires 4 steps to make a decision:

- **Step 1**: Model the distribution of the predictors $X$ separately for each class in $Y$
- **Step 2**: Use Bayes' theorem to calculate estimates for the posterior probability
- **Step 3**: Derive the discriminant function for each class
- **Step 4**: Apply a decision rule to classify the observation

---

**Step 1: Estimate Class Distributions**

We assume that each class $k$ generates its data points from a multivariate normal distribution:

$$
P(X | Y = k) = \frac{1}{(2\pi)^{p/2} |\Sigma|^{1/2}} \exp\left(-\frac{1}{2} (X - \mu_k)^T \Sigma^{-1} (X - \mu_k)\right)
$$

where:

- $X$ is the feature vector
- $p$ is the number of features
- $\mu_k$ is the mean vector of class $k$
- $\Sigma$ is the covariance matrix shared by all classes

In practice, we do not know $\mu_k$ and $\Sigma$, so we estimate them from the training data: $\mu_k$ is the mean of the observations in class $k$, and $\Sigma$ is the average of the covariance matrices within the classes (the *pooled* covariance matrix).

---

**Step 2: Apply Bayes' Theorem**

We want to know the posterior probability. This is the probability of a class $k$ given a new observation $X$:

$$
P(Y = k | X) = \frac{P(X | Y = k) P(Y = k)}{P(X)}
$$

where:

- $P(X | Y = k)$ is the likelihood (the Gaussian density from Step 1)
- $P(Y = k)$ is the prior probability of class $k$, which we also write as $\pi_k$
- $P(X)$ is the evidence (the overall probability of observing $X$)

The prior $P(Y = k)$ is typically estimated as the relative frequency of class $k$ in the training data, unless we want to impose different priors.

---

To classify a new observation, we only need to know which class has the largest posterior probability. Taking the logarithm preserves the order of the probabilities and turns multiplication into addition. The evidence $P(X)$ is the same for all classes, so it does not affect the ranking and becomes a constant:

$$
\log P(Y = k | X) = \log P(X | Y = k) + \log P(Y = k) + \text{const}
$$

---

**Step 3: Derive the Discriminant Function**

If we substitute the Gaussian density from Step 1 into this expression and expand it, we can remove all terms that do not depend on the class $k$ (see [James et al. (ISLR) Chapter 4.4](https://www.statlearning.com/) if you are interested in the details). This includes the quadratic term $X^T \Sigma^{-1} X$, which is the same for all classes because they share $\Sigma$. What remains is the discriminant function $\delta_k(X)$:

$$
\delta_k(X) = X^T \Sigma^{-1} \mu_k - \frac{1}{2} \mu_k^T \Sigma^{-1} \mu_k + \log(\pi_k)
$$

where:

- $X^T \Sigma^{-1} \mu_k$ measures how well $X$ matches the mean of class $k$, taking the covariance into account
- $-\frac{1}{2} \mu_k^T \Sigma^{-1} \mu_k$ is a constant for each class, which depends on the class mean
- $\log(\pi_k)$ favors classes that are more common (prior probability)

$\delta_k(X)$ is linear in $X$ (there are no squared terms), which is why LDA produces linear decision boundaries.

---

**Step 4: Decision Rule**

The final decision is made by comparing the discriminant functions, and we classify the new observation into the class with the highest discriminant value:

$$
\hat{Y} = \arg \max_k \delta_k(X)
$$


### Connection to Logistic Regression

In the two-class case, LDA and logistic regression lead to the same type of model: the log-odds are a linear function of $X$. The difference is how the coefficients are estimated. Logistic regression estimates them directly and makes no assumption about the distribution of the features. This requires an iterative optimization, which is why you sometimes see convergence warnings. LDA derives them from the class proportions, the class means, and the shared covariance matrix. These are simple summary statistics of the training data, so fitting LDA requires no iterative optimization at all. If the features are approximately Gaussian within each class, LDA tends to be more stable and can perform better. If they are not, logistic regression is usually the safer choice.

---

## Quadratic Discriminant Analysis (QDA)

QDA is a more flexible version of LDA. It also assumes a Gaussian distribution of the features within each class, but it allows each class to have its own covariance matrix $\Sigma_k$.

The discriminant function for QDA is:

$$
\delta_k(X) = -\frac{1}{2} (X - \mu_k)^T \Sigma_k^{-1} (X - \mu_k) - \frac{1}{2} \log |\Sigma_k| + \log(\pi_k)
$$

where:
- $\Sigma_k$ is the covariance matrix specific to class $k$
- The term $-\frac{1}{2} \log |\Sigma_k|$ is present because the spread of the data differs between the classes

Since $\Sigma_k$ now differs between the classes, the quadratic term of $X$ can no longer be removed. This leads to quadratic decision boundaries.

```{tip} Choosing Between LDA and QDA

**LDA**
- Is a good choice when the classes have a similar spread in the feature space
- Tends to work better when the sample size is small or the number of features is high

**QDA**
- Is more appropriate when the spread differs clearly between the classes
- Needs a large sample size *per class* to estimate the separate covariance matrices

This is another bias-variance trade-off. A covariance matrix for $p$ features has $p(p+1)/2$ parameters. LDA estimates one such matrix, while QDA estimates one per class, so QDA has less bias but more variance.
```

The following plot shows the difference between the two assumptions. Under the LDA assumption, the two classes have the same shape and orientation. Under the QDA assumption, each class has its own:

```{code-cell} ipython3
:tags: [remove-input]

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from matplotlib import gridspec

np.random.seed(3)

# Generate Bivariate Normal Data
def bivariate_norm(m1, m2, sigma1, sigma2, n1, n2):
    d1 = np.random.multivariate_normal(m1, sigma1, n1)
    d2 = np.random.multivariate_normal(m2, sigma2, n2)

    X = np.vstack([d1, d2])
    groups = np.array(["1"] * n1 + ["2"] * n2)

    return X, groups

# Parameters
m_1 = np.array([0.5, -0.5])
m_2 = np.array([-1.5, 0.7])
sigma_1 = np.array([[1, 0.5],
                    [0.5, 1]])
sigma_2 = np.array([[0.8, -0.7],
                    [-0.7, 0.8]])

# Generate datasets
X_lda, g_lda = bivariate_norm(m_1, m_2, sigma_1, sigma_1, 2000, 2000)
X_qda, g_qda = bivariate_norm(m_1, m_2, sigma_1, sigma_2, 2000, 2000)

# Helper: Contour Plot Function
def density_contour(ax, X, groups, title):
    colors = {"1": "red", "2": "blue"}

    for group in ["1", "2"]:
        pts = X[groups == group].T
        kde = gaussian_kde(pts)

        # Grid for evaluation
        xmin, ymin = X.min(axis=0) - 1
        xmax, ymax = X.max(axis=0) + 1
        xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
        grid = np.vstack([xx.ravel(), yy.ravel()])

        zz = kde(grid).reshape(xx.shape)
        ax.contour(xx, yy, zz, colors=colors[group], linewidths=1)

    ax.set_title(title)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")

# Plot
fig, ax = plt.subplots(1,2)
density_contour(ax[0], X_lda, g_lda, "Under LDA assumption")
density_contour(ax[1], X_qda, g_qda, "Under QDA assumption")
plt.tight_layout();
```

## LDA and QDA in Python

In this example, we use simulated data with 2 features and 2 classes. Note that the two classes clearly have different shapes:

```{code-cell} ipython3
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Generate synthetic data
X, y = make_classification(n_samples=400, n_features=2, n_informative=2, 
                           n_redundant=0, n_classes=2, n_clusters_per_class=1, 
                           random_state=5)

fig, ax = plt.subplots()
ax.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr', alpha=0.5)
ax.set(title="Simulated Data", xlabel="x1", ylabel="x2");
```

Fitting the model is straightforward. However, please have a look at the [documentation](https://scikit-learn.org/stable/api/sklearn.discriminant_analysis.html) for additional options.

```{code-cell} ipython3
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA, \
                                          QuadraticDiscriminantAnalysis as QDA


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

lda = LDA(store_covariance=True) # store the covariance matrix, so that we can plot it later
lda.fit(X_train, y_train)

qda = QDA(store_covariance=True)
qda.fit(X_train, y_train);
```

We can then print the classification report:

```{code-cell} ipython3
from sklearn.metrics import classification_report

# Print classification report
print('LDA Classification Report:')
print(classification_report(y_test, lda.predict(X_test)))

print('QDA Classification Report:')
print(classification_report(y_test, qda.predict(X_test)))
```

QDA performs better on the test set (93% vs. 88% accuracy), which is what we would expect given the different shapes of the classes. To see what the models do, we can plot the Gaussian distribution that each model learned for each class, together with the resulting decision boundary:

```{code-cell} ipython3
import numpy as np
import seaborn as sns
from scipy.stats import multivariate_normal
from matplotlib.lines import Line2D

sns.set_theme(style="darkgrid")

def plot_distributions(model, X, y, ax, title):
    # Grid over the feature space
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
    Xgrid = np.c_[xx.ravel(), yy.ravel()]

    # Plot the Gaussian distribution learned for each class
    for k, color in enumerate(['blue', 'red']):
        mu = model.means_[k]
        # LDA shares one covariance matrix, QDA stores one per class
        cov = model.covariance_[k] if isinstance(model.covariance_, list) else model.covariance_
        P = multivariate_normal(mean=mu, cov=cov).pdf(Xgrid)
        P = (P / P.max()).reshape(xx.shape)  # normalise so the peak is 1
        Pm = np.ma.masked_array(P, P < 0.03)
        ax.pcolormesh(xx, yy, Pm, shading='auto', alpha=0.4, cmap=color.title() + 's')
        ax.contour(xx, yy, P, levels=[0.01, 0.1, 0.5, 0.9], colors=color, alpha=0.3)

    # Decision boundary
    Z = model.predict(Xgrid).reshape(xx.shape)
    ax.contour(xx, yy, Z, levels=[0.5], linewidths=2, colors='black')

    # Data points
    ax.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='bwr')
    ax.set(title=title, xlabel="Feature 1", ylabel="Feature 2")

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
plot_distributions(lda, X_train, y_train, ax[0], "LDA Class Distributions")
plot_distributions(qda, X_train, y_train, ax[1], "QDA Class Distributions")

legend_elements = [
    Line2D([], [], marker='o', linestyle='None', markerfacecolor='blue',
           markeredgewidth=0, label='Class 0', markersize=8),
    Line2D([], [], marker='o', linestyle='None', markerfacecolor='red',
           markeredgewidth=0, label='Class 1', markersize=8),
    Line2D([], [], color='black', linestyle='-', linewidth=2, label='Decision boundary')
]
ax[0].legend(handles=legend_elements, loc="upper left")  # explicit loc: "best" is slow on a dense mesh
plt.show()
```

LDA gives both classes a distribution with the same shape and orientation, even though the red class is narrower and tilted. As a result, its decision boundary is a straight line. QDA estimates a separate covariance matrix for each class, so its distributions match the shapes of the classes much better, and the boundary is curved. The curved boundary also encloses regions far away from most of the data, such as the area in the lower left that QDA assigns to the red class.

## Summary

| | Logistic regression | LDA | QDA |
|---|---|---|---|
| What it models | $P(Y \mid X)$ directly | $P(X \mid Y)$ and $P(Y)$ | $P(X \mid Y)$ and $P(Y)$ |
| Assumption about the features | None | Gaussian within each class | Gaussian within each class |
| Covariance matrices | Not modeled | One shared by all classes | One per class |
| Decision boundary | Linear | Linear | Quadratic |

```{code-cell} ipython3
:tags: [remove-input]
import sys; sys.path.append("..")
from quizpager import display_quiz_paged
display_quiz_paged('quiz/LDA_QDA.json')
```

You will apply LDA and QDA in Exercise 5, together with the Naïve Bayes classifier from the next session.
