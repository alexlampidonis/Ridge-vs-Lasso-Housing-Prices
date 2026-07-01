A comparison of two regularized linear regression techniques — **Ridge** (L2 penalty) and **Lasso** (L1 penalty) — for predicting median house values in California, with hyperparameter selection via cross-validation and an analysis of how each regularization method shrinks feature coefficients differently.

---

## Table of Contents

- [Problem Overview](#problem-overview)
- [Dataset](#dataset)
- [Approach](#approach)
  - [1. Data Preparation](#1-data-preparation)
  - [2. Ridge Regression](#2-ridge-regression)
  - [3. Lasso Regression](#3-lasso-regression)
  - [4. Coefficient Comparison](#4-coefficient-comparison)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [How to Run](#how-to-run)
- [Key Takeaways](#key-takeaways)

---

## Problem Overview

The goal is to predict the **median house value** (in units of $100,000) for California districts based on features like median income, house age, average rooms/bedrooms, population, and location. Two regularized linear models — Ridge and Lasso regression — are trained and compared, both in terms of predictive performance (mean squared error) and how they handle feature selection through coefficient shrinkage.

## Dataset

The **California Housing dataset**, loaded directly via `sklearn.datasets.fetch_california_housing()`. It contains district-level features (median income, house age, average rooms, average bedrooms, population, average occupancy, latitude, longitude) with no missing values, and a continuous target: median house value.

## Approach

### 1. Data Preparation

- Features (`X`) and target (`y`) were extracted directly from the dataset.
- The data was split into **75% training / 25% test**, using a fixed random seed for reproducibility.
- Features were standardized with `StandardScaler` (fit on the training set only, then applied to both splits), which is important for regularized linear models since the penalty term is sensitive to feature scale — without standardization, features with larger numeric ranges would be penalized unfairly compared to smaller-scale features.

### 2. Ridge Regression

A `RidgeCV` model was fit using 5-fold cross-validation to automatically select the best regularization strength (**alpha**, the L2 penalty weight) from a logarithmically spaced grid (`np.logspace(-6, 6, 13)`). The selected model was evaluated by computing mean squared error (MSE) on both the training and test sets.

### 3. Lasso Regression

A `LassoCV` model was similarly fit using 5-fold cross-validation over a logarithmically spaced alpha grid. The search grid was refined during experimentation — starting from a coarser grid (`np.logspace(-6, 6, 13)`, which selected α ≈ 0.001) and moving to a finer, wider grid (`np.logspace(-8, 8, 50)`), which selected a stronger regularization strength (α ≈ 0.00356) and produced a smaller test MSE than the initial pass.

### 4. Coefficient Comparison

The fitted coefficients from both models were placed side-by-side in a table (one row per feature) and visualized as a grouped bar chart, to directly compare how each regularization method affects individual feature weights.

## Results

- Ridge and Lasso achieved **very similar training and test MSE**, with Lasso showing a **slightly lower test MSE**, suggesting marginally better generalization to unseen data in this case.
- The clearest qualitative difference was in the coefficients themselves: **Lasso shrank the `Population` coefficient to exactly zero**, effectively removing it from the model — a hallmark of L1 regularization's built-in feature selection behavior, which Ridge (L2) does not replicate, since Ridge shrinks coefficients toward zero but rarely sets them to exactly zero.
- Most other coefficients were generally smaller under Lasso than under Ridge, with `HouseAge` being a minor exception (slightly smaller under Ridge).

## Repository Structure

```
.
└── Regularization_Regressions.ipynb   # Full notebook: data prep, Ridge/Lasso fitting, coefficient comparison
```

## How to Run

1. Clone the repository and open the notebook in Jupyter, Google Colab, or your preferred environment.
2. Install dependencies:
   ```bash
   pip install numpy pandas scikit-learn matplotlib
   ```
3. Run the single notebook cell top to bottom. No external dataset download is required — the California Housing dataset is fetched directly via `scikit-learn`.

## Key Takeaways

- **Ridge and Lasso solve the same overfitting problem differently**: both add a penalty to the loss function to discourage large coefficients, but Ridge's L2 penalty shrinks coefficients smoothly toward zero, while Lasso's L1 penalty can drive some coefficients to *exactly* zero — making Lasso useful as an implicit feature selection tool.
- **Cross-validated alpha selection removes guesswork**: rather than manually picking a regularization strength, `RidgeCV`/`LassoCV` search a grid of candidate values and pick the one that performs best under cross-validation, balancing bias and variance automatically.
- **The choice of search grid matters**: refining the Lasso alpha search from a coarse grid to a wider, finer one changed the selected regularization strength meaningfully and improved test performance, a reminder that hyperparameter search granularity is itself a modeling choice worth iterating on.
- **Standardization is essential before regularization**: since penalty terms act directly on coefficient magnitudes, features must be on comparable scales for the penalty to treat them fairly.
