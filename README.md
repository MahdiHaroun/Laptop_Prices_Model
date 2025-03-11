# Laptop Prices Regression Model

### Overview of the Project 🚀🌍🌐

Advanced Regression Module to Predict Laptop Prices Based on Features Such As **CPU**, **GPU**, **RAM**, and More. The project utilizes advanced regression models, machine learning, exploratory data analysis (EDA), and train-test splitting methods to ensure the best score using the R^2 function.

### Features Found in Dataset:

1. **Brand**  
2. **Processor**  
3. **GPU**  
4. **RAM (GB)**  
5. **Storage**  
6. **Screen Size (inch)**  
7. **Resolution**  
8. **Battery Life (hours)**  
9. **Weight (kg)**  
10. **Operating System**  
11. **Price ($)**

## Data Preprocessing Phase 📈🔧🌍

The dataset does not contain null values or duplicated entries.

## Exploratory Data Analysis (EDA) Phase 1 🎨🔢🔍

- Started by checking for features with a high correlation to the target variable, **Price ($)**.
- Checked for normal distribution for the **RAM (GB)** feature and the target feature using plots and skewness methods. Applied log transformation for normalization, enhancing model performance.
- Began feature engineering using the `get_dummies()` function from NumPy for quick and efficient data encoding.

## Machine Learning Methodology 📊🌐🚀

- Splitting the data with **30% for testing**.

## Results and Discussion

- Using `LinearRegression()` and `StandardScaler()`, the score achieved was **0.9703**. Switching to `MinMaxScaler()` with the `Pipeline()` method gave the same score.
- Adding polynomial regression of degree 2 with `StandardScaler()` increased the score to **0.9867**. Given that `StandardScaler` and `MinMaxScaler` gave similar results, only `StandardScaler` was used in further tests.
- Using Ridge Regression with `StandardScaler` gave the same results as linear regression. Similarly, Ridge Regression with polynomial degree 2 and `GridSearchCV` (5-fold cross-validation) also matched the result of **0.9876**.
- For Lasso Regression starting with `alpha=0.1` (without scaling), the score was **0.9703**. To find the optimal alpha, a function was used to compare coefficient weights and identify the best value. The worst alpha led to an R^2 score of **-0.0008**, indicating overfitting.
- Using Lasso with `GridSearchCV` and `Pipeline()` (including `StandardScaler`) achieved a score of **0.9875**, showing significant improvement.
- Elastic Net Regression with an initial alpha of **0.1**, `l1_ratio=0.1`, `StandardScaler`, and polynomial degree 2 via pipeline achieved a score of **0.9759**. Using `GridSearchCV` improved this to **0.9876**. The best score was achieved using Elastic Net with `GridSearchCV` and `StandardScaler` through the pipeline.

## Exploratory Data Analysis (EDA) Phase 2 🎨🔢🔍

- Transitioned from using `get_dummies()` to `OneHotEncoder` and `LabelEncoder` for feature encoding.

## Results and Discussion

- After feature encoding and keeping **30% for testing**, Linear Regression with `StandardScaler` gave a score of **0.8807**, worse than using `get_dummies` in Phase 1. Adding polynomial features (degree 2) through pipeline improved the score to **0.9883**. Degree 3 slightly reduced it to **0.9856**.
- Ridge Regression with polynomial degree 2 and `StandardScaler` matched the linear model's score of **0.9883**.
- Lasso Regression with polynomial degree 2 and `StandardScaler` gave a score of **0.9350**. Increasing the degree to 3 resulted in **0.9941**, the highest so far.
- Using the pipeline named **8_1** and `GridSearchCV` for cross-validation, the best parameters identified were:

  - **Lasso Regression Module**
  - **Alpha = 1**
  - **Polynomial Degree = 3**
  - Achieved Score: **0.99417**

## References 🔍📑🌐

Kaggle Dataset: [Laptop Prices](https://www.kaggle.com/datasets/asinow/laptop-price-dataset/data)

