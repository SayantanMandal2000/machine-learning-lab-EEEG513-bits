# Logistic Regression using the Newton-Raphson Method

This repository contains the code and results for **Lab Assignment 4** of the  
**Machine Learning for Electronics Engineers Lab**.  

The project focuses on implementing **logistic regression for binary classification** *from scratch* using the **Newton-Raphson optimization method**.

---

## 📌 Project Overview
The main goal of this lab is to understand the inner workings of logistic regression by manually performing parameter updates (beta coefficients) and verifying results with a Python implementation.  

The **Newton-Raphson method** is used here to iteratively compute the **maximum likelihood estimates** of the logistic regression model parameters.

The notebook covers **two binary classification problems** with different datasets.

---

## 🔹 Task 1: Simple Binary Classification

**Objective:**  
Build a logistic regression model to classify data points based on a single predictor variable.

**Dataset:**

| xᵢ | yᵢ |
|----|----|
| 2  | 0  |
| 4  | 0  |
| 6  | 1  |

- **Algorithm:** Logistic Regression with Newton-Raphson  
- **Initial Beta Values:** β₀ = 0.5, β₁ = -0.5  
- **Iterations:** 2  

### ✅ Results
- Iteration 1: β = [-8.269, 2.423]  
- Iteration 2: β = [-6.544, 0.759]  

Final model equation:  

$$
P(x) = \frac{1}{1 + e^{-(-9.58 + 2.74x)}}
$$

**Prediction (x = 5):**
- Predicted Probability: **0.0602**  
- Classified Label: **0**  

---

## 🔹 Task 2: Extended Binary Classification

**Objective:**  
Perform logistic regression on a larger dataset and classify new samples.

**Dataset:**  
- x = [1, 2, …, 10]  
- y = [0, 0, 0, 1, …, 1]  

- **Algorithm:** Logistic Regression with Newton-Raphson  
- **Initial Beta Values:** β₀ = 1.0, β₁ = 0.5  
- **Iterations:** 3  

### ✅ Results
- Iteration 1: β = [-8.195, 1.794]  
- Iteration 2: β = [-6.092, 1.754]  
- Iteration 3: β = [-9.585, 2.742]  

Final model equation:  
$$
P(x) = \frac{1}{1 + e^{-(-9.58 + 2.74x)}}
$$

**Predictions:**
- Input x = 2.5 → Probability = **0.0613**, Classified Label = **0**  
- Input x = 3.5 → Probability = **0.5033**, Classified Label = **1**  

---

## ⚙️ Dependencies
This project requires the following Python library:

- `numpy`

Install it with:
```bash
pip install numpy

