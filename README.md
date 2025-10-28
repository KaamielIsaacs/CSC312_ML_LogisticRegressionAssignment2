# 📉 CSC312 – Machine Learning – Logistic Regression

This repository contains the solution for **Assignment 2** of the CSC312 Machine Learning course, focusing on the implementation of **Logistic Regression** for binary classification. The project is divided into two main parts, addressing both **non-regularized** and **regularized** forms of the model.

-----

## ✨ Project Overview

The primary objective of this assignment is to implement the core components of the Logistic Regression model using a fully **vectorized approach** in NumPy, thereby avoiding explicit loops over training examples.

The assignment utilizes two different datasets to demonstrate Logistic Regression's capabilities in handling:

1.  **Linearly Separable Data** (Non-regularized).
2.  **Non-linearly Separable Data** (Regularized, using Polynomial Features).

-----

## 🛠️ Key Implementation Components

The solution notebook (`CSC312_Assignment_2_Logistic_Regression_ex.ipynb`) contains the implementation of the following vectorized functions:

### 1\. Model Core

  * **Sigmoid Function ($\mathbf{g(z)}$):** The core activation function used for hypothesis calculation:
    $$h_{\theta}(x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$
  * **Hypothesis Function ($\mathbf{h(\theta)}$):** Calculates the probability of the positive class for a given set of features $X$ and parameters $\theta$.
  * **Prediction Function:** Converts the probability output of the hypothesis function into binary class labels (0 or 1).

### 2\. Learning Algorithm

  * **Cost Function ($\mathbf{J(\theta)}$):** Implementation of the penalized log-loss function:
      * **Non-Regularized** (Part 1): $\quad J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \left[ y^{(i)}\log(h_\theta(x^{(i)})) + (1-y^{(i)})\log(1-h_\theta(x^{(i)})) \right]$
      * **Regularized** (Part 2): $\quad J(\theta)_{reg} = J(\theta) + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2$
  * **Gradient Descent (Vectorized):** Implementation of a function to iteratively update the parameters $\theta$ using the gradient of the cost function. This includes the logic for applying **regularization ($\lambda$)** to the gradient descent step (excluding $\theta_0$).

### 3\. Feature Engineering (Part 2)

  * **Polynomial Feature Mapping:** Implementation of a function to transform the original two features ($x_1, x_2$) into a higher-dimensional space ($1, x_1, x_2, x_1^2, x_1 x_2, x_2^2, \dots$) up to a specified polynomial degree. This is crucial for fitting a non-linear decision boundary to the data.

-----

## 📂 Data Sets

The assignment uses two distinct datasets provided within the notebook:

| Part | Dataset Type | Description | Model Focus |
| :--- | :--- | :--- | :--- |
| **Part 1** | Linearly Separable | Simple data for basic logistic regression implementation. | Non-Regularized Cost & Gradient |
| **Part 2** | Non-Linearly Separable | Data with a circular or complex boundary, requiring feature expansion. | Regularized Cost & Polynomial Features |

-----

## 🚀 Setup and Execution

### Prerequisites

  * Python 3.x
  * The following libraries:
      * `numpy` (for vectorized computations)
      * `pandas`
      * `scikit-learn`
      * `matplotlib` (for decision boundary plotting)

### How to Run

1.  **Open the Notebook:** Load the `CSC312_Assignment_2_Logistic_Regression_ex.ipynb` file in a Jupyter environment.
2.  **Complete Code Sections:** Implement the required functions within the designated code blocks:
    ```python
    #FILL IN BELOW

    #STOP FILLING IN HERE
    ```
3.  **Execute:** Run the cells sequentially to initialize parameters, perform gradient descent, and visualize the final decision boundary for both the non-regularized and regularized models.
