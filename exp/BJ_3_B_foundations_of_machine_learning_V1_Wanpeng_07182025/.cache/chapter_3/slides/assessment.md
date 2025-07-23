# Assessment: Slides Generation - Week 3: Supervised Learning - Linear Regression

## Section 1: Introduction to Linear Regression

### Learning Objectives
- Understand concepts from Introduction to Linear Regression

### Activities
- Practice exercise for Introduction to Linear Regression

### Discussion Questions
- Discuss the implications of Introduction to Linear Regression

---

## Section 2: Key Concepts in Linear Regression

### Learning Objectives
- Understand and define the roles of dependent and independent variables in linear regression.
- Compute and interpret the slope and intercept of a regression line.
- Demonstrate the ability to graphically represent a linear regression model and analyze its parameters.

### Assessment Questions

**Question 1:** What is the dependent variable in a linear regression model?

  A) The variable that is being predicted
  B) The variable that is manipulated
  C) The variable that causes change in another
  D) None of the above

**Correct Answer:** A
**Explanation:** The dependent variable is the variable that is being predicted or explained, which depends on the independent variable.

**Question 2:** In the regression equation y = mx + b, what does 'm' represent?

  A) The dependent variable
  B) The slope of the regression line
  C) The y-intercept
  D) The independent variable

**Correct Answer:** B
**Explanation:** 'm' represents the slope of the regression line, indicating how much the dependent variable y changes for a one-unit increase in the independent variable x.

**Question 3:** What can be inferred if the slope (m) of the regression line is negative?

  A) There is no relationship between variables
  B) The dependent variable increases as the independent variable increases
  C) The dependent variable decreases as the independent variable increases
  D) The independent variable has no effect on the dependent variable

**Correct Answer:** C
**Explanation:** A negative slope indicates an inverse relationship, meaning as the independent variable increases, the dependent variable decreases.

**Question 4:** What does the intercept (b) represent in the linear regression equation?

  A) The value of y when x is zero
  B) The slope of the regression line
  C) The independent variable
  D) The change in y for a one-unit change in x

**Correct Answer:** A
**Explanation:** The intercept (b) represents the predicted value of the dependent variable y when the independent variable x is zero.

### Activities
- 1. Given a dataset of house prices including square footage, number of bedrooms, and age, create a scatter plot and fit a linear regression line. Analyze the slope and intercept.
- 2. Using an online linear regression tool or software, input your own data to perform a simple linear regression analysis, and present your findings on the correlation between your chosen independent variable and the dependent variable.

### Discussion Questions
- What challenges do you think arise when choosing independent variables for a linear regression model?
- How would you explain the importance of the regression line to someone unfamiliar with data analysis?
- Can you think of a real-world situation where linear regression could be applied? Discuss potential independent and dependent variables.

---

## Section 3: Mathematical Foundations

### Learning Objectives
- Understand concepts from Mathematical Foundations

### Activities
- Practice exercise for Mathematical Foundations

### Discussion Questions
- Discuss the implications of Mathematical Foundations

---

## Section 4: Simple vs Multiple Linear Regression

### Learning Objectives
- Understand concepts from Simple vs Multiple Linear Regression

### Activities
- Practice exercise for Simple vs Multiple Linear Regression

### Discussion Questions
- Discuss the implications of Simple vs Multiple Linear Regression

---

## Section 5: Assumptions of Linear Regression

### Learning Objectives
- Understand the key assumptions of linear regression and their relevance to model validity.
- Be able to assess whether these assumptions hold true in practical applications of linear regression.

### Assessment Questions

**Question 1:** What assumption states that the relationship between the independent and dependent variables must be linear?

  A) Homoscedasticity
  B) Independence
  C) Normality of Errors
  D) Linearity

**Correct Answer:** D
**Explanation:** Linearity is the assumption that the relationship between the independent variable(s) and the dependent variable is linear, meaning that changes in the independent variable produce proportional changes in the dependent variable.

**Question 2:** Which assumption requires that the residuals be equally spread across all levels of the independent variable?

  A) Normality of Errors
  B) Linearity
  C) Independence
  D) Homoscedasticity

**Correct Answer:** D
**Explanation:** Homoscedasticity means that the variance of the residuals should remain constant across all predicted values. This ensures that the prediction errors do not exhibit some biased patterns.

**Question 3:** What statistical test is commonly used to check the independence of residuals?

  A) T-test
  B) Wilcoxon test
  C) Durbin-Watson statistic
  D) Chi-square test

**Correct Answer:** C
**Explanation:** The Durbin-Watson statistic is a test used to detect the presence of autocorrelation in the residuals of a regression model, which assesses their independence.

**Question 4:** What is a common method for visually assessing the normality of residuals?

  A) Boxplot
  B) Histogram
  C) Pie chart
  D) Line chart

**Correct Answer:** B
**Explanation:** A histogram allows for a visual inspection of the distribution of residuals, and when the residuals are normally distributed, the histogram will approximate a bell-shaped curve.

### Activities
- Create a scatter plot using a provided dataset of study hours vs. exam scores. Fit a linear regression line and assess the assumptions of linearity, homoscedasticity, and normality of errors by examining the resulting plots (residual plot and histogram).
- Using statistical software (e.g., Python with statsmodels or R), run a linear regression analysis on a sample dataset. Report whether the assumptions are met and describe any visualizations used to check them.

### Discussion Questions
- Why is it important to check the assumptions of linear regression before interpreting results?
- How would you approach a situation where one of the assumptions is violated, for example, if the residuals are not normally distributed?

---

## Section 6: Implementation of Linear Regression

### Learning Objectives
- Understand the step-by-step process of implementing linear regression using Scikit-learn.
- Be able to interpret model coefficients and their implications on feature relevance.

### Assessment Questions

**Question 1:** What is the purpose of the train-test split in the implementation of linear regression?

  A) To combine the features and target variable
  B) To validate the model's performance on unseen data
  C) To increase the number of observations in the dataset
  D) To make predictions on the training dataset

**Correct Answer:** B
**Explanation:** The train-test split allows us to evaluate how well our model generalizes to new, unseen data by keeping a separate set of observations for testing.

**Question 2:** Which of the following libraries is used for linear regression in Python as per this slide?

  A) TensorFlow
  B) NumPy
  C) Scikit-learn
  D) Matplotlib

**Correct Answer:** C
**Explanation:** Scikit-learn is the library utilized for implementing linear regression as highlighted in the slide.

**Question 3:** In a linear regression model, the coefficients obtained after fitting the model represent?

  A) The predicted target variable values
  B) The loss function of the model
  C) The sensitivity of the target variable to changes in each feature
  D) The shape of the regression line

**Correct Answer:** C
**Explanation:** Each coefficient indicates how much the target variable is expected to change with a one-unit change in the respective feature, holding other features constant.

**Question 4:** What can be inferred if the model coefficients are close to zero?

  A) The model fits data perfectly
  B) The features are highly relevant to predicting the target variable
  C) The features may not have a significant impact on the target variable
  D) The model is overfitting the training data

**Correct Answer:** C
**Explanation:** Coefficients that are close to zero suggest that the corresponding features do not contribute significantly to the prediction of the target variable.

### Activities
- Using a different dataset, implement a linear regression model following the steps outlined. Report the model coefficients and visualize the predictions.
- Try to change the size of the train-test split and observe how it affects the model performance. Document your findings.

### Discussion Questions
- Why is it important to understand the data preparation process before fitting a linear regression model?
- How might the assumptions of linear regression affect your model's performance?

---

## Section 7: Evaluating Model Performance

### Learning Objectives
- Understand concepts from Evaluating Model Performance

### Activities
- Practice exercise for Evaluating Model Performance

### Discussion Questions
- Discuss the implications of Evaluating Model Performance

---

## Section 8: Interpreting Model Results

### Learning Objectives
- Understand concepts from Interpreting Model Results

### Activities
- Practice exercise for Interpreting Model Results

### Discussion Questions
- Discuss the implications of Interpreting Model Results

---

## Section 9: Common Issues in Linear Regression

### Learning Objectives
- Understand the concept and implications of multicollinearity in linear regression.
- Identify methods to diagnose and resolve multicollinearity issues.
- Recognize the phenomenon of overfitting and its impacts on model performance.
- Explore regularization techniques and their roles in managing model complexity.

### Assessment Questions

**Question 1:** What is multicollinearity in a linear regression model?

  A) When the dependent variable is correlated with the independent variables
  B) When two or more independent variables are highly correlated
  C) The model is too simplistic
  D) When the model accurately predicts new data

**Correct Answer:** B
**Explanation:** Multicollinearity occurs when two or more independent variables are highly correlated, leading to instability in coefficient estimates.

**Question 2:** What is a common method to diagnose multicollinearity?

  A) Cross-validation
  B) Variance Inflation Factor (VIF)
  C) R-squared calculation
  D) T-test for coefficients

**Correct Answer:** B
**Explanation:** Variance Inflation Factor (VIF) is used to diagnose multicollinearity; a VIF greater than 10 suggests high multicollinearity.

**Question 3:** Which of the following indicates overfitting?

  A) High accuracy on both training and validation datasets
  B) Low bias and high variance
  C) Model used is overly simplified
  D) High accuracy on training data but low on validation data

**Correct Answer:** D
**Explanation:** Overfitting typically shows high accuracy on training data but significantly lower accuracy when tested on validation or new data.

**Question 4:** Which regularization technique adds a penalty equal to the absolute value of the magnitude of coefficients?

  A) Ridge Regression
  B) Lasso Regression
  C) Elastic Net
  D) None of the Above

**Correct Answer:** B
**Explanation:** Lasso Regression (L1 regularization) adds a penalty based on the absolute value of coefficient magnitudes, promoting sparsity.

### Activities
- Identify a dataset and visually inspect the correlation between independent variables using a correlation matrix. Discuss your findings related to multicollinearity.
- Using a sample dataset, fit both a basic linear regression model and a complex polynomial model. Evaluate and compare their performance using training and validation datasets to illustrate overfitting.

### Discussion Questions
- Why is it crucial to diagnose multicollinearity before interpreting a regression model?
- How might the inclusion of irrelevant predictors affect the performance of a linear regression model?
- Can you provide an example of a real-world situation where overfitting might occur?

---

## Section 10: Applications of Linear Regression

### Learning Objectives
- Understand the applications of linear regression in various fields such as economics, healthcare, and social sciences.
- Analyze real-world datasets using linear regression and interpret the results.

### Assessment Questions

**Question 1:** Which field commonly uses linear regression to predict economic growth?

  A) Healthcare
  B) Education
  C) Economics
  D) Marketing

**Correct Answer:** C
**Explanation:** Economists use linear regression to model relationships between GDP growth and various economic factors such as interest rates.

**Question 2:** What is a key advantage of using linear regression in healthcare?

  A) It eliminates the need for data
  B) It guides public health interventions
  C) It requires no statistical software
  D) It offers subjective results

**Correct Answer:** B
**Explanation:** Linear regression helps assess how different levels of physical activity affect health metrics, guiding public health interventions.

**Question 3:** In social sciences, linear regression models the impact of education on what?

  A) Public transport
  B) Income
  C) Climate change
  D) Technology advancements

**Correct Answer:** B
**Explanation:** Researchers use linear regression to evaluate how years of education influence an individual's income.

**Question 4:** What relationship do businesses analyze using linear regression in marketing?

  A) Customer satisfaction and product features
  B) Sales revenue and advertising spend
  C) Employee performance and job satisfaction
  D) Website traffic and bounce rate

**Correct Answer:** B
**Explanation:** Companies utilize linear regression to quantify how changes in advertising expenditures affect sales revenue.

### Activities
- Use a dataset (provided in class) to perform linear regression analysis on a chosen variable. Interpret the coefficients and discuss the implications of your findings.
- Conduct a small survey to collect data on physical activity levels and health metrics (like blood pressure). Use linear regression to analyze the results and present your conclusions.

### Discussion Questions
- Discuss how the application of linear regression can vary across disciplines. What challenges might arise in certain fields?
- How might linear regression influence decision-making in public policy? Provide examples based on the healthcare or social sciences context.

---

