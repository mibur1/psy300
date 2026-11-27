---
short_title: KNN
kernelspec:
  name: python3
  display_name: Python 3
---

# 🎯 K-Nearest Neighbors

LDA, QDA, and Naïve Bayes all assume a certain distribution of the features within each class. The k-nearest neighbors (kNN) algorithm makes no such assumption. It is a *non-parametric* method: it does not estimate any parameters, but simply stores the training data. To classify a new observation, it finds the $k$ closest points in the training data (its *neighbors*) and assigns the most common class among them. For regression, the prediction is the average of the neighbors' values instead.

The following plot illustrates the idea with two classes. The point in the center is a new observation, which we want to classify based on the labeled points of Class A and Class B. To do this, we calculate its distance to every point. The dashed circle marks the distance to the fifth-nearest neighbor, so all points inside it are the 5 nearest neighbors:

```{code-cell} ipython3
:tags: [remove-input]
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

np.random.seed(123)
palette = sns.color_palette("Set2", 3)

class_a = np.random.randn(10, 2) + [-2, 2]
class_b = np.random.randn(10, 2) + [2, -2]

new_example = np.array([[0, 0]])
distances = np.vstack([class_a, class_b])

# Distances
euclidean_distances = np.sqrt(((distances - new_example) ** 2).sum(axis=1))
sorted_indices = np.argsort(euclidean_distances)
nearest_indices = sorted_indices[:5]
radius = euclidean_distances[nearest_indices[-1]]

# Data points
sns.scatterplot(x=class_a[:, 0], y=class_a[:, 1], color=palette[0], s=200, label='Class A')
sns.scatterplot(x=class_b[:, 0], y=class_b[:, 1], color=palette[1], s=200, label='Class B')
sns.scatterplot(x=new_example[:, 0], y=new_example[:, 1], color=palette[2], s=300)

# Annotate
plt.annotate("Data to\nclassify", (0.1, 0.1), textcoords="offset points", xytext=(80, 40), ha='center',
             arrowprops=dict(arrowstyle='->', color='grey'))

# Lines
for i, point in enumerate(distances):
    color = 'black' if i in nearest_indices else 'grey'
    alpha = 1 if i in nearest_indices else 0.5
    plt.plot([new_example[0, 0], point[0]], [new_example[0, 1], point[1]], linestyle='--', color=color, alpha=alpha, zorder=-1)

# Circle around the 5 nearest neighbors
circle = plt.Circle((0, 0), radius, color='grey', linestyle='--', fill=False, alpha=0.5)
plt.gca().add_artist(circle)
plt.gca().set_aspect('equal', adjustable='box')

# Custom legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', label='Class A', markerfacecolor=palette[0], markersize=10),
    Line2D([0], [0], marker='o', color='w', label='Class B', markerfacecolor=palette[1], markersize=10),
    Line2D([0], [0], linestyle='--', color='grey', label='Distances')
]

plt.xlim(-6,6)
plt.ylim(-5.5,5.5)
plt.xticks([], [])
plt.yticks([], [])
plt.legend(handles=legend_elements, loc="lower left", frameon=False)
plt.title("5-Nearest Neighbors")
plt.show()
```

In an algorithmic description, this includes the following steps:

**Step 1: Neighbor Identification**  
Given a positive integer $k$ and an observation $x_0$, the kNN classifier first identifies the $k$ points in the training data that are closest to $x_0$, represented by the set $N_0$.

**Step 2: Conditional Probability Estimation**  
The classifier then estimates the conditional probability for class $j$ as the fraction of points in $N_0$ whose response values equal $j$:

$$P(Y = j \mid X = x_0) = \frac{1}{k} \sum_{i \in N_0} I(y_i = j)$$

where:
- $I(y_i = j)$ is an indicator function that equals 1 if the label of the neighbor $y_i$ is class $j$, and 0 otherwise.
- $k$ is the number of neighbors considered.
- $N_0$ is the set of the $k$ nearest neighbors to $x_0$.

**Step 3: Classification Decision**  
Finally, the test observation $x_0$ is classified into the class with the largest estimated probability:

$$\hat{y} = \underset{j}{\operatorname{argmax}} \; P(Y = j \mid X = x_0)$$

In the plot above, 3 of the 5 nearest neighbors belong to Class A, so the estimated probability for Class A is 3/5 = 0.6, and the new observation is classified as Class A.

## Choosing k

The number of neighbors $k$ is a hyperparameter, and it controls the bias-variance trade-off:

- A **small $k$** (e.g. $k = 1$) gives a very flexible model. The decision boundary follows every single training point, which makes it sensitive to noise and outliers (low bias, high variance).
- A **large $k$** averages over many neighbors and gives a smooth decision boundary, but it can miss finer structures in the data (high bias, low variance). In the extreme case where $k$ equals the number of training points, the model always predicts the most common class.

There is no formula to determine the best $k$ in advance, so we choose it with cross-validation.

```{important} Scale your features
kNN is based on distances. If one feature is measured in large units (e.g. an area in mm²) and another in small units (e.g. a ratio between 0 and 1), the first one dominates the distances, and the second one is practically ignored. You should therefore standardize the features before fitting a kNN model.
```

## Today's Data: The Iris Dataset

You already know the [Iris dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html). It contains measurements for three iris species: setosa, versicolor, and virginica. To keep things visual, we only use the sepal length and width:

```{code-cell} ipython3
import pandas as pd
from sklearn import datasets

# Get data
iris = datasets.load_iris(as_frame=True)

df = iris.frame
df['class'] = pd.Categorical.from_codes(iris.target, iris.target_names)

sns.scatterplot(data=df, x="sepal length (cm)", y="sepal width (cm)", hue="class", palette="Set2");
```

Setosa stands out clearly, as it tends to have shorter and wider sepals. Versicolor and virginica overlap a lot, which makes them harder to separate with these two features. Given a **new flower** with certain sepal measurements, how can we decide which species it belongs to? This is the kind of problem a classifier like kNN is designed to solve.

```{code-cell} ipython3
X = df[["sepal length (cm)", "sepal width (cm)"]]
y = df["target"]
```

## kNN Classifier Implementation

We fit the kNN model for a range of $k$ values and evaluate each of them with a stratified and shuffled 5-fold cross-validation. As in the PCA session, we use a pipeline, so that the scaler is fitted on the training folds only:

```{code-cell} ipython3
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
k_range = range(1, 60)
accuracies = []

# Loop over all k values and save the mean accuracy
for k in k_range:
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    scores = cross_val_score(knn, X, y, cv=cv)
    accuracies.append(scores.mean())

best_k = k_range[np.argmax(accuracies)]
print(f"Best k: {best_k}")
print(f"Accuracy: {max(accuracies):.2f}")

# Plot
fig, ax = plt.subplots()
sns.lineplot(x=k_range, y=accuracies, marker='o', ax=ax)
ax.axvline(best_k, color="gray", linestyle="--")
ax.set(xlabel="k", ylabel="Cross-validated accuracy");
```

With $k = 1$, the model reaches only about 71% accuracy. The accuracy increases with $k$ and levels off at around 78-79% for $k$ values above roughly 20. The highest value is reached at $k = 32$, but the differences between the larger values of $k$ are small and partly due to chance.

To see what $k$ does, we can compare the decision boundaries of a very flexible model ($k = 1$), the model with the best $k$, and a very smooth model ($k = 100$):

```{code-cell} ipython3
from matplotlib.colors import ListedColormap

# Grid for predicting and plotting the decision boundary (in the original units)
x_min, x_max = X.iloc[:, 0].min() - 0.5, X.iloc[:, 0].max() + 0.5
y_min, y_max = X.iloc[:, 1].min() - 0.5, X.iloc[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid = pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=X.columns)

cmap = ListedColormap(sns.color_palette("Set2", 3))  # same 3 colours as the points
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

for ax, k in zip(axes, [1, best_k, 100]):
    knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
    knn.fit(X, y)
    Z = knn.predict(grid).reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap)
    sns.scatterplot(data=df, x="sepal length (cm)", y="sepal width (cm)", hue="class", palette="Set2", ax=ax, legend=(k == 1))
    ax.set(title=f"k = {k}")

plt.tight_layout()
```

With $k = 1$, the decision boundary is fragmented and creates small islands around single flowers. With $k = 100$, the boundary between versicolor and virginica is very smooth, and the model starts to ignore the structure of the data. The best $k$ lies in between.

```{note} kNN in higher dimensions
kNN works well with a small number of features and many observations. With many features, all points become far away from each other, so even the "nearest" neighbors are not really close anymore. This is known as the *curse of dimensionality*. In such cases, models with more structure (e.g. logistic regression or LDA) often perform better.
```

## Summary

* kNN classifies an observation by the majority class of its $k$ nearest training points. For regression, it averages their values (`KNeighborsRegressor`).
* It makes no assumptions about the distribution of the data, but it needs scaled features, because it is based on distances.
* $k$ controls the bias-variance trade-off: a small $k$ gives a flexible but noisy model, a large $k$ a smooth but possibly too simple one. We choose it with cross-validation.

```{code-cell} ipython3
:tags: [remove-input]
import sys; sys.path.append("..")
from quizpager import display_quiz_paged
display_quiz_paged('quiz/KNN.json')
```

**Now it's your turn:** head to the [exercise section](Exercises.ipynb) for Exercise 6, in which you will use kNN to classify breast tumors.
