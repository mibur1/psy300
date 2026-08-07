---
short_title: Regression recap
kernelspec:
  name: python3
  display_name: Python 3
---

# 🔁 Recap: Regression Models

```{code-cell} ipython3
:tags: [remove-cell]
import plotly.io as pio

# A neutral Plotly look that stays legible in both the light and the dark
# version of the site: transparent paper, faint plot background, grey text.
psy300 = pio.templates["plotly_white"]
psy300.layout.paper_bgcolor = "rgba(0,0,0,0)"
psy300.layout.plot_bgcolor = "rgba(128,128,128,0.08)"
psy300.layout.font.color = "#888888"
pio.templates["psy300"] = psy300
pio.templates.default = "psy300"
```

One of the most important concepts in any multivariate statistics seminar such as [psy111](https://mibur1.github.io/psy111) are (linear) regression models. Let's quickly recap this concept and how to implement it in Python.

Have a look at the following code, which creates some simulated data. Can you deduce from the code what the underlying pattern is?

```python
import numpy as np
import pandas as pd

x = np.linspace(-5, 5, 30)
y = (x**3 + np.random.normal(0, 15, size=x.shape)) / 50

df = pd.DataFrame({'x': x, 'y': y})
```

:::{dropdown} Click to reveal the plot
Here you can see the data in a scatterplot, with a linear regression model fitted to the data. Do you think the linear model fits the data well?

```{code-cell} ipython3
:tags: [remove-input]
import warnings
warnings.filterwarnings("ignore", message=".*Polyfit may be poorly conditioned.*")

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Generate sample data
np.random.seed(42)
x = np.linspace(-5, 5, 30)
y = (x**3 + np.random.normal(0, 15, size=x.shape)) / 50
df = pd.DataFrame({'x': x, 'y': y})

# Scatter trace for the raw data
scatter = go.Scatter(
    x=df['x'], y=df['y'], mode='markers', name='Data',
    marker=dict(size=10, color='lightgrey', line=dict(color='gray', width=2)),
)

# Fit a linear regression model (polynomial order 1)
coeffs = np.polyfit(df['x'], df['y'], 1)
x_fit = np.linspace(-5, 5, 400)
y_fit = np.polyval(coeffs, x_fit)

regression = go.Scatter(x=x_fit, y=y_fit, mode='lines', name='Model',
                        line=dict(width=3, color='#4c72b0'))

layout = go.Layout(
    xaxis=dict(title="x", range=[-5.5, 5.5], fixedrange=True),
    yaxis=dict(title="y", range=[-3, 3], fixedrange=True),
    margin=dict(l=10, r=10, t=30, b=20),
)
go.Figure(data=[scatter, regression], layout=layout)
```
:::

Let's take a closer look at the model. As introduced last semester, we can e.g. use the `statsmodels.formula.api` library to specify and fit regression models with a formula notation similar to R. We use the `ols()` class to fit the model specified as `y ~ x`, which translates to "y predicted by x":

```{code-cell} ipython3
import statsmodels.formula.api as smf

model = smf.ols("y ~ x", data=df).fit()
print(model.summary())
```

In this output, the most important information are the model parameters displayed under `coef` and the performance statistics such as the `R-squared`.
You can see that our model has an R-squared of 0.753, which means that the model explains 75% of the variance in the data. That's pretty good! But I'm sure we can do better. After all, life is more complicated than just a straight line, no? <sub>(And we also know that the underlying data was simulated according to a 3rd order polynomial.)</sub>

**Drag the slider** below to increase the order of the polynomial and watch the R² climb:

```{code-cell} ipython3
:tags: [remove-input]

# Scatter trace for the raw data
scatter = go.Scatter(x=df['x'], y=df['y'], mode='markers', name='Data',
                     marker=dict(size=10, color='lightgrey', line=dict(color='gray', width=2)))

# Generate regression curves for polynomial orders 1 through 30
regression_traces = []
r2_list = []
x_fit = np.linspace(-5, 5, 400)
for order in range(1, 31):
    coeffs = np.polyfit(df['x'], df['y'], order)
    y_fit = np.polyval(coeffs, x_fit)
    regression_traces.append(
        go.Scatter(x=x_fit, y=y_fit, mode='lines', name='Model', visible=False,
                   line=dict(width=3, color='#4c72b0'))
    )

    y_pred = np.polyval(coeffs, df['x'])
    r2 = 1 - np.sum((df['y'] - y_pred) ** 2) / np.sum((df['y'] - df['y'].mean()) ** 2)
    r2_list.append(r2)

regression_traces[0]['visible'] = True

# Create slider steps
steps = []
for i in range(30):
    vis = [True] + [False] * 30
    vis[i + 1] = True
    steps.append(dict(
        method="update",
        args=[{"visible": vis},
              {"annotations": [dict(x=0.02, y=0.98, xref="paper", yref="paper",
                                    text=f"Model R² = {r2_list[i]:.3f}",
                                    showarrow=False, font=dict(size=18, color="gray"), align="left")]}],
        label=str(i + 1),
    ))

sliders = [dict(active=0, currentvalue={"prefix": "Order of the polynomial regression model: "},
                pad={"t": 30}, steps=steps)]

layout = go.Layout(
    annotations=[dict(x=0.02, y=0.98, xref="paper", yref="paper",
                      text=f"Model R² = {r2_list[0]:.3f}",
                      showarrow=False, font=dict(size=18, color="gray"), align="left")],
    sliders=sliders,
    xaxis=dict(title="x", range=[-5.5, 5.5], fixedrange=True),
    yaxis=dict(title="y", range=[-3, 3], fixedrange=True),
    margin=dict(l=10, r=10, t=30, b=20),
)

go.Figure(data=[scatter] + regression_traces, layout=layout)
```

As you probably expected, the R² increases as you increase the order of the polynomial in the regression model. However, this doesn't stop after the 3rd order polynomial (which is the true function that generated the data). The R² continues to increase until it hits 1 for a model that includes a 29th-order polynomial. You can see that the model now goes through every single one of the data points. This did not happen by chance! A polynomial of degree 29 can perfectly interpolate the present data, which consists of 30 data points. This is because a polynomial of degree $n-1$ has $n$ coefficients, which can be uniquely determined to pass through $n$ distinct points (given that all the x-values are distinct).

But what should you do with this information? Well, as the topic of this seminar is *statistical and machine learning*, we are usually concerned with making predictions for new, unseen data. Until now, we have always fit (trained) and evaluated (tested) our model on the same data, aiming to make statistical inferences about the coefficients of relatively small models. We can call this the *training data*[^training]. However, we can also generate new data with the same underlying function and test the model on this repeatedly generated data that reflects the same underlying true association between y and x. We call this the *testing data*[^testing]:

[^training]: Training data refers to the data which was used for model fitting.
[^testing]: Testing data refers to data which was used to evaluate the performance of a model. This is new, unseen data, meaning that it was not used for training.

```{code-cell} ipython3
:tags: [remove-input]

# Compute polynomial coefficients fitted on the existing (training) data
training_coeffs = [np.polyfit(df['x'], df['y'], order) for order in range(1, 31)]

# Generate new test data (randomly drawn so the points are not equally spaced)
np.random.seed(69)
x_all_test = np.linspace(-5, 5, 100)
indices_test = np.sort(np.random.choice(np.arange(100), size=30, replace=False))
x_test = x_all_test[indices_test]
y_test = (x_test**3 + np.random.normal(0, 15, size=x_test.shape)) / 50
df_test = pd.DataFrame({'x': x_test, 'y': y_test})

test_scatter = go.Scatter(
    x=df_test['x'], y=df_test['y'], mode='markers', name='Test Data',
    marker=dict(size=10, color='lightgrey', line=dict(color='gray', width=2)))

# Apply the *training* models to the *test* data
regression_traces_test = []
r2_test_list = []
for order in range(1, 31):
    coeffs = training_coeffs[order - 1]

    y_fit = np.polyval(coeffs, x_fit)
    regression_traces_test.append(
        go.Scatter(x=x_fit, y=y_fit, mode='lines', name='Model', visible=False,
                   line=dict(width=3, color='#4c72b0')))

    y_pred_test = np.polyval(coeffs, df_test['x'])
    r2_test = 1 - np.sum((df_test['y'] - y_pred_test)**2) / np.sum((df_test['y'] - df_test['y'].mean())**2)
    r2_test_list.append(r2_test)

regression_traces_test[0]['visible'] = True

steps_test = []
for i in range(30):
    vis = [True] + [False] * 30
    vis[i + 1] = True
    steps_test.append(dict(
        method="update",
        args=[{"visible": vis},
              {"annotations": [dict(x=0.02, y=0.98, xref="paper", yref="paper",
                                    text=f"Test R² = {r2_test_list[i]:.3f}",
                                    showarrow=False, font=dict(size=18, color="gray"), align="left")]}],
        label=str(i + 1)))

layout_test = go.Layout(
    annotations=[dict(x=0.02, y=0.98, xref="paper", yref="paper",
                      text=f"Test R² = {r2_test_list[0]:.3f}",
                      showarrow=False, font=dict(size=18, color="gray"), align="left")],
    sliders=[dict(active=0, currentvalue={"prefix": "Order of polynomial: "},
                  pad={"t": 30}, steps=steps_test)],
    xaxis=dict(title="x", range=[-5.5, 5.5], fixedrange=True),
    yaxis=dict(title="y", range=[-3, 3], fixedrange=True),
    margin=dict(l=10, r=10, t=30, b=20),
)

go.Figure(data=[test_scatter] + regression_traces_test, layout=layout_test)
```

:::{margin}
<a href="https://commons.wikimedia.org/wiki/File:William_of_Ockham.png" target="_blank">
<img src="https://upload.wikimedia.org/wikipedia/commons/7/70/William_of_Ockham.png" alt="William of Occam" style="width:100%;">
</a>

William of Occam. Image by [Moscarlop](https://commons.wikimedia.org/wiki/File:William_of_Ockham.png), used under [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).
:::

You can see that the R² now has a peak around a 3rd order polynomial regression model and then drastically decreases for higher order polynomials. This means that our previously trained models do not really fit our new data anymore. Why is this the case? Basically, these higher-order models became too flexible and *overfit*[^overfit] to the training data. Once we apply the models to new testing data, they will produce a much worse performance, as they are too specialized (they basically just memorized the training data).

[^overfit]: Overfitting refers to fitting patterns in the training data which do not generalize to the testing data.

So how can we then find the best model that avoids overfitting? From your classes in basic (psychological) methods, you might be familiar with *Occam's razor*, which states that:

> Entia non sunt multiplicanda praeter necessitatem.

This essentially translates to *"Before you try a complicated hypothesis, you should make quite sure that no simplification of it will explain the facts equally well"*. Our model should thus be as simple as possible, but as complex as necessary. In machine learning, this concept is often referred to as the [bias-variance tradeoff](2_bias_variance.md), which we will explore next week.

## Different Ways of Implementing Regression Models

There are many different Python packages that allow you to implement regression models, with popular choices being `statsmodels`, `numpy`, and `sklearn`. The specific choice ultimately depends on your preferences and goals.

At the top of this section, you have already seen the high-level *formula API* solution from `statsmodels`. For a cubic model, this would look like this:

```{code-cell} ipython3
import statsmodels.formula.api as smf

# Fit model
model = smf.ols("y ~ x + I(x**2) + I(x**3)", data=df).fit()

# Predictions
x_new = pd.DataFrame({"x": [-4, 1, 3]})
print(model.predict(x_new))
```

Alternatively, you can use the lower-level `OLS()` approach, which requires you to manually specify the design matrix (make sure to add a column for the intercept/bias):

```{code-cell} ipython3
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures

# Create design matrix
X = x.reshape(-1, 1)
poly = PolynomialFeatures(degree=3, include_bias=True)
X_poly = poly.fit_transform(X)

# Fit model
model = sm.OLS(y, X_poly).fit()
```

A cool thing about statsmodels is that you can get the model results in various forms:

```{code-cell} ipython3
summary = model.summary()
print(summary)
```

```{code-cell} ipython3
print(summary.tables[1])
```

```{code-cell} ipython3
print(model.params)  # [beta0, beta1, beta2, beta3]
```

```{code-cell} ipython3
# Prediction
x_new = np.array([[-4], [1], [3]])
X_new_poly = poly.transform(x_new)
y_hat = model.predict(X_new_poly)
print(y_hat)
```

With the polynomial features, we can also directly use `sklearn`:

```{code-cell} ipython3
from sklearn.linear_model import LinearRegression

# Fit model
model = LinearRegression(fit_intercept=False).fit(X_poly, y)  # the intercept is already in X_poly
print(model.coef_)  # [beta0, beta1, beta2, beta3]

# Predict
y_hat = model.predict(X_new_poly)
print(y_hat)
```

And last but not least, one could also use `numpy`:

```{code-cell} ipython3
import numpy as np
from matplotlib import pyplot as plt

coeffs = np.polyfit(x, y, deg=3)
print(coeffs)
p = np.poly1d(coeffs)

# Predict
x_new = np.array([-4, 1, 3])
y_hat = p(x_new)
print(y_hat)

# Plot
fig, ax = plt.subplots()
ax.scatter(x, y, color="gray", label="Data")
x_fit = np.linspace(x.min(), x.max(), 200)
ax.plot(x_fit, p(x_fit), label="Model")
plt.legend();
```

In practice, `statsmodels` is ideal for statistical analysis and reporting (it provides a lot of information for inference), `scikit-learn` for machine learning pipelines and prediction, and `numpy` for quick or lightweight fits.

```{tip} Summary
- Regression models can be used as prediction models in the context of machine learning.
- The performance of a prediction model should always be assessed on new, unseen data.
- It is often useful to look for the simplest possible model that still provides sufficiently accurate answers.
- There are many Python packages that allow you to implement regression models. Choose whichever suits your goals best.
```
