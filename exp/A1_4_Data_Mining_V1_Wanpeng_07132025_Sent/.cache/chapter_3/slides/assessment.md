# Assessment: Slides Generation - Week 4: Logistic Regression

## Section 1: Introduction to Logistic Regression

### Learning Objectives
- Understand concepts from Introduction to Logistic Regression

### Activities
- Practice exercise for Introduction to Logistic Regression

### Discussion Questions
- Discuss the implications of Introduction to Logistic Regression

---

## Section 2: Motivation for Logistic Regression

### Learning Objectives
- Understand concepts from Motivation for Logistic Regression

### Activities
- Practice exercise for Motivation for Logistic Regression

### Discussion Questions
- Discuss the implications of Motivation for Logistic Regression

---

## Section 3: Understanding Binary Classification

### Learning Objectives
- Define binary classification and its key characteristics.
- Explain the role of logistic regression in binary classification.
- Understand the usage of the logistic function in mapping input features to outcomes.

### Assessment Questions

**Question 1:** What is binary classification?

  A) Categorizing data into more than two classes.
  B) Predicting a continuous outcome.
  C) Classifying data into two distinct categories.
  D) Analyzing time-based data.

**Correct Answer:** C
**Explanation:** Binary classification involves categorizing data into two distinct classes.

**Question 2:** Which function does logistic regression utilize to map input features to probabilities?

  A) Linear function
  B) Quadratic function
  C) Logistic function (sigmoid function)
  D) Exponential function

**Correct Answer:** C
**Explanation:** Logistic regression uses the logistic function (sigmoid function) to convert linear combinations into probabilities ranging from 0 to 1.

**Question 3:** In logistic regression, what does a predicted probability of more than 0.5 usually indicate?

  A) The instance belongs to class 0.
  B) The instance belongs to class 1.
  C) The model is invalid.
  D) The probability threshold must be lowered.

**Correct Answer:** B
**Explanation:** A predicted probability greater than 0.5 typically means that the instance is predicted to belong to class 1.

**Question 4:** What is a common application of binary classification?

  A) Image recognition
  B) Stock price prediction
  C) Email spam detection
  D) Natural language processing

**Correct Answer:** C
**Explanation:** Email spam detection is a classic example of binary classification as it involves determining if an email is spam or not.

### Activities
- Create a simple dataset that can be used to apply logistic regression. Define two features and two classes, then discuss how you would approach building a logistic regression model.

### Discussion Questions
- Can you think of other practical examples where binary classification is applied in daily life?
- What challenges might arise when using logistic regression for binary classification?
- Discuss how the choice of decision threshold can affect the performance of a binary classification model.

---

## Section 4: Logistic Function

### Learning Objectives
- Understand the logistic function and its formula.
- Interpret the logistic function in the context of probability.
- Recognize the significance of logistic regression in binary classification.

### Assessment Questions

**Question 1:** What is the range of the logistic function?

  A) [0, 1]
  B) (-∞, ∞)
  C) [0, ∞)
  D) (-1, 1)

**Correct Answer:** A
**Explanation:** The logistic function maps any real-valued number to the range [0, 1].

**Question 2:** Which of the following best describes the shape of the logistic function curve?

  A) Linear
  B) S-shaped (sigmoid)
  C) Parabolic
  D) Exponential

**Correct Answer:** B
**Explanation:** The logistic function produces an S-shaped curve known as the sigmoid curve.

**Question 3:** If f(x) = 0.8 for a given input x, what can we infer about the predicted class?

  A) Class 0
  B) Class 1
  C) Uncertain outcome
  D) Not enough information

**Correct Answer:** B
**Explanation:** Since f(x) > 0.5, we predict class 1 (e.g., 'yes', 'true', 'positive').

**Question 4:** In a logistic regression model, what does the output of the logistic function represent?

  A) A continuous value
  B) A probability of being in a certain class
  C) A categorical output
  D) None of the above

**Correct Answer:** B
**Explanation:** The output of the logistic function is interpreted as the probability of the input belonging to a specific class.

### Activities
- Graph the logistic function using a graphing tool or software. Highlight its range and the S-shaped nature of the curve.
- Using a dataset, apply logistic regression to predict a binary outcome and interpret the model using the logistic function.

### Discussion Questions
- In what real-world scenarios can the logistic function be applied? Provide examples.
- Discuss how the logistic function compares to other functions used in regression analysis.
- How does the interpretation of probabilities from the logistic function inform decision-making in business or healthcare?

---

## Section 5: Modeling with Logistic Regression

### Learning Objectives
- Understand concepts from Modeling with Logistic Regression

### Activities
- Practice exercise for Modeling with Logistic Regression

### Discussion Questions
- Discuss the implications of Modeling with Logistic Regression

---

## Section 6: Cost Function and Optimization

### Learning Objectives
- Explain the role of the cost function in logistic regression and its formulation.
- Discuss the gradient descent method and how it applies to optimize model parameters in logistic regression.

### Assessment Questions

**Question 1:** What is the purpose of the cost function in logistic regression?

  A) To measure model performance
  B) To penalize misclassifications
  C) To optimize parameter estimates
  D) All of the above

**Correct Answer:** D
**Explanation:** The cost function in logistic regression serves multiple purposes, including measuring performance, penalizing misclassifications, and optimizing parameters.

**Question 2:** Which optimization algorithm is commonly used to minimize the cost in logistic regression?

  A) Gradient Descent
  B) Newton's Method
  C) Genetic Algorithm
  D) None of the above

**Correct Answer:** A
**Explanation:** Gradient Descent is the standard optimization algorithm used in logistic regression to minimize the cost function.

**Question 3:** In the gradient descent update rule, what does the term 'alpha' represent?

  A) The cost function
  B) The learning rate
  C) The number of iterations
  D) The model parameters

**Correct Answer:** B
**Explanation:** 'Alpha' is the learning rate that determines the size of the steps taken towards minimizing the cost function.

**Question 4:** Why is the Binary Cross-Entropy Loss used in logistic regression?

  A) It is suitable for multiple classes
  B) It measures the dissimilarity between the true and predicted probabilities
  C) It simplifies the optimization process
  D) All of the above

**Correct Answer:** B
**Explanation:** Binary Cross-Entropy Loss is specifically designed for binary classification problems, measuring the dissimilarity between true labels and predicted probabilities effectively.

### Activities
- Implement gradient descent to minimize the cost function for a sample logistic regression problem using a programming language of your choice (e.g., Python, R).
- Visualize the cost function over iterations of gradient descent to understand how the cost decreases as parameters are updated.

### Discussion Questions
- What challenges might arise when using gradient descent in optimization?
- How can choosing different learning rates affect the performance of the gradient descent algorithm in logistic regression?
- What modifications can be made to improve the convergence of gradient descent?

---

## Section 7: Making Predictions

### Learning Objectives
- Understand how to compute predictions using the logistic regression model.
- Interpret the outputs, including predicted probabilities and odds, from a logistic regression analysis.
- Identify the role of coefficients and thresholds in making predictions.

### Assessment Questions

**Question 1:** What is the output of the logistic function?

  A) An integer indicating the class label
  B) A probability between 0 and 1
  C) A categorical variable
  D) The logit score

**Correct Answer:** B
**Explanation:** The output of the logistic function is a probability between 0 and 1 that indicates the likelihood of the event occurring.

**Question 2:** What does the term 'decision boundary' refer to in logistic regression?

  A) The line that divides the training data into two classes
  B) The threshold applied to the predicted probabilities
  C) The coefficients of the model
  D) None of the above

**Correct Answer:** B
**Explanation:** The 'decision boundary' refers to the threshold applied to the predicted probabilities to classify observations into different classes.

**Question 3:** If the probability predicted by logistic regression is 0.7 and the threshold is set at 0.5, what will be the predicted class?

  A) Class 0
  B) Class 1
  C) Unclassified
  D) Depends on additional data

**Correct Answer:** B
**Explanation:** Since the probability (0.7) is greater than the threshold (0.5), the predicted class will be Class 1.

**Question 4:** What is the interpretation of a coefficient in a logistic regression model?

  A) The change in predicted probability with a unit change in the predictor variable
  B) A direct count of occurrences of an event
  C) The fixed point in the model where predictions change
  D) None of the above

**Correct Answer:** A
**Explanation:** A coefficient in logistic regression indicates how much the log odds of the outcome changes with a one-unit increase in the predictor variable.

### Activities
- Use a dataset to implement logistic regression in a programming language of your choice. Report the predicted probabilities and class labels for several observations.
- Create a confusion matrix based on your predictions and assess the model's accuracy.

### Discussion Questions
- Why is it important to choose a threshold when using logistic regression? How can different thresholds affect the outcomes?
- How can you explain the predicted probabilities to a non-technical audience?

---

## Section 8: Performance Metrics

### Learning Objectives
- Identify evaluation metrics specific to logistic regression.
- Understand how to interpret accuracy, precision, recall, and F1-score.
- Recognize scenarios where each metric is relevant.

### Assessment Questions

**Question 1:** What does precision measure in the context of logistic regression?

  A) The proportion of true positive predictions among all predictions made
  B) The proportion of correct predictions overall
  C) The ability of the model to capture all positive cases
  D) The balance between false positives and true positives

**Correct Answer:** A
**Explanation:** Precision measures the accuracy of the positive predictions made by the model, specifically how many of the predicted positives were actual positives.

**Question 2:** If a model has a recall of 70% and a precision of 80%, what does this indicate?

  A) The model identifies 70% of all actual positives correctly.
  B) The model has a balanced performance across all metrics.
  C) The model has a high rate of false positives.
  D) The model is unreliable and should not be used.

**Correct Answer:** A
**Explanation:** A recall of 70% indicates that the model successfully identifies 70% of actual positive cases, while precision indicates that 80% of predicted positives are indeed positives.

**Question 3:** In what scenario might you prioritize recall over precision?

  A) Email spam detection
  B) Medical diagnosis for a life-threatening disease
  C) Image classification tasks
  D) Financial market predictions

**Correct Answer:** B
**Explanation:** In medical diagnosis for a life-threatening disease, identifying most actual positive cases (high recall) is prioritized to ensure patients receive necessary treatment.

**Question 4:** What is the F1-score used for?

  A) To measure the total correct predictions in a dataset.
  B) To balance precision and recall in evaluating a model.
  C) To analyze multi-class classification performance.
  D) To calculate the error rate of a regression model.

**Correct Answer:** B
**Explanation:** The F1-score is the harmonic mean of precision and recall, and is particularly useful when handling imbalanced datasets to balance the two metrics.

### Activities
- Using a given dataset, calculate the accuracy, precision, recall, and F1-score of your logistic regression model. Discuss your findings with a partner.

### Discussion Questions
- Why is it important to consider multiple performance metrics when evaluating a model?
- How might the domain of an application influence the choice of performance metrics?

---

## Section 9: ROC Curve and AUC

### Learning Objectives
- Describe the significance of the ROC curve in model evaluation.
- Calculate and interpret the Area Under the Curve (AUC).
- Explain the relationship between the ROC curve and the performance of binary classifiers.

### Assessment Questions

**Question 1:** What is plotted on the Y-axis of the ROC curve?

  A) False Positive Rate (FPR)
  B) True Positive Rate (TPR)
  C) Accuracy
  D) Precision

**Correct Answer:** B
**Explanation:** The Y-axis of the ROC curve represents the True Positive Rate (TPR), which measures the proportion of actual positives correctly identified.

**Question 2:** What does an AUC of 0.7 indicate?

  A) The model is perfect in classification.
  B) The model has poor discriminative ability.
  C) The model has acceptable discriminative ability.
  D) The model performs worse than random guessing.

**Correct Answer:** C
**Explanation:** An AUC of 0.7 suggests the model has acceptable discriminative ability, meaning it performs better than random chance, but is not perfect.

**Question 3:** What does a ROC curve closer to the top-left corner signify?

  A) Poor model performance
  B) Random guessing
  C) Better model performance
  D) Inadequate data

**Correct Answer:** C
**Explanation:** A ROC curve that is closer to the top-left corner indicates better model performance, with higher True Positive Rates and lower False Positive Rates.

**Question 4:** In the context of ROC curve analysis, what is a False Positive Rate (FPR)?

  A) The percentage of actual negatives that are incorrectly classified as positives.
  B) The percentage of identified positives that are false.
  C) The total number of false classifications divided by total classifications.
  D) The total number of true positives divided by total positives.

**Correct Answer:** A
**Explanation:** FPR is defined as the proportion of actual negatives that are incorrectly identified as positives.

### Activities
- Using a dataset of your choice, generate an ROC curve using statistical software (e.g., Python, R) and calculate the AUC. Discuss the implications of the AUC value you obtained.
- Create a simulated binary classification scenario where you manually define TPR and FPR at different thresholds, and plot the ROC curve based on your calculations.

### Discussion Questions
- How can ROC curves be used to compare different classification models?
- What challenges might arise when interpreting ROC curves for imbalanced datasets?
- Discuss how varying the threshold affects the TPR and FPR, and why it is crucial to understand this relationship.

---

## Section 10: Assumptions of Logistic Regression

### Learning Objectives
- Outline key assumptions associated with logistic regression models.
- Discuss implications of these assumptions for model quality.
- Identify methods to test for violations of logistic regression assumptions.

### Assessment Questions

**Question 1:** Which assumption is NOT associated with logistic regression?

  A) Linear relationship between independent variables and log odds
  B) Independence of observations
  C) Normality of predictors
  D) No multicollinearity among predictors

**Correct Answer:** C
**Explanation:** Logistic regression does not require the predictors to be normally distributed.

**Question 2:** What does it mean for independent variables to have no multicollinearity?

  A) They are perfectly correlated with one another.
  B) They have little or no correlation.
  C) They all have a linear relationship with the dependent variable.
  D) They include only categorical variables.

**Correct Answer:** B
**Explanation:** No multicollinearity means that independent variables have little or no correlation with each other.

**Question 3:** Why is a large sample size important for logistic regression?

  A) It ensures the model can include more predictors.
  B) It helps improve the accuracy of parameter estimates.
  C) It allows the use of causal inference.
  D) It guarantees normality among predictors.

**Correct Answer:** B
**Explanation:** A large sample size helps improve the accuracy of parameter estimates and ensures stability in the model.

**Question 4:** Which graph can you use to check the linearity in logit for your model?

  A) Histogram
  B) Box plot
  C) Residual plot
  D) Scatter plot of log odds vs. predictions

**Correct Answer:** D
**Explanation:** A scatter plot of log odds vs. predictions is used to visually assess the linearity assumption.

### Activities
- Conduct a simulation where students run a logistic regression analysis on a dataset, checking the assumptions beforehand and reporting any violations.
- Create a small report summarizing how each assumption of logistic regression was tested using a real or synthetic data set.

### Discussion Questions
- What are the potential consequences of violating the independence of observations assumption in logistic regression?
- How could multicollinearity affect the meanings of coefficients in your model?

---

## Section 11: Common Applications

### Learning Objectives
- Identify various real-world applications of logistic regression in different industries.
- Analyze how logistic regression serves as a decision-making tool in healthcare, finance, marketing, and social media.
- Discuss the significance of interpreting logistic regression outcomes correctly for practical applications.

### Assessment Questions

**Question 1:** Which industry commonly uses logistic regression for predicting customer churn?

  A) Manufacturing
  B) Retail
  C) Marketing
  D) Agriculture

**Correct Answer:** C
**Explanation:** Logistic regression is extensively used in marketing to analyze customer behavior and predict churn, which helps in devising retention strategies.

**Question 2:** What is a common application of logistic regression in healthcare?

  A) Predicting stock prices
  B) Predicting the likelihood of disease
  C) Classifying emails as spam or not
  D) Estimating delivery times

**Correct Answer:** B
**Explanation:** In healthcare, logistic regression is frequently used to predict the likelihood of various diseases based on patient data.

**Question 3:** In the context of finance, what does logistic regression help determine?

  A) Investment strategies
  B) Creditworthiness of loan applicants
  C) Stock market trends
  D) Customer preferences

**Correct Answer:** B
**Explanation:** Logistic regression is applied in finance for evaluating the creditworthiness of loan applicants by analyzing different financial variables.

**Question 4:** Which outcome variable is typically used in logistic regression to identify diseases?

  A) Age
  B) Disease status
  C) Salary
  D) Interest rates

**Correct Answer:** B
**Explanation:** The outcome variable in logistic regression for medical diagnosis is often disease status, indicating if a disease is present (1) or absent (0).

### Activities
- Research and report on an innovative application of logistic regression in a non-traditional field, such as environmental science or sports analytics. Prepare a presentation summarizing your findings.

### Discussion Questions
- What are some challenges that might arise when using logistic regression in real-world applications?
- In what ways can logistic regression models be improved for better accuracy in predictions?
- Can you think of an industry where logistic regression has not yet been applied but could be beneficial? Why?

---

## Section 12: Case Study: Logistic Regression in Action

### Learning Objectives
- Examine a real-world dataset using logistic regression to understand its relevance in predicting outcomes.
- Identify the implications of logistic regression results on decision-making in a business context.

### Assessment Questions

**Question 1:** What outcome does logistic regression predict?

  A) A continuous variable
  B) A categorical variable with more than two classes
  C) A binary categorical variable
  D) A series of linear equations

**Correct Answer:** C
**Explanation:** Logistic regression is specifically designed for binary classification problems, predicting outcomes that fall into one of two categories.

**Question 2:** Why is data preparation crucial in logistic regression?

  A) It randomizes the input data.
  B) It ensures the data is in a usable format and contains no inaccuracies.
  C) It increases the size of the dataset.
  D) It replaces all values with zeros.

**Correct Answer:** B
**Explanation:** Data preparation helps to ensure that the dataset is clean, accurate, and suitable for building a model, leading to more reliable predictions.

**Question 3:** What does the ROC curve illustrate?

  A) The relationship between two continuous variables
  B) The trade-off between sensitivity and specificity
  C) The accuracy of a regression equation
  D) The linear relationship between features

**Correct Answer:** B
**Explanation:** The ROC curve helps visualize the trade-off between true positive rate (sensitivity) and false positive rate (1-specificity) for a binary classifier.

**Question 4:** What role do coefficients play in a logistic regression model?

  A) They determine the model's complexity.
  B) They indicate the strength and direction of the relationship between features and the outcome.
  C) They represent categorical variables.
  D) They are used for data normalization.

**Correct Answer:** B
**Explanation:** Coefficients in logistic regression measure the change in the log-odds of the outcome for a one-unit change in the predictor variable.

### Activities
- 1. Recreate the data preparation steps using a different dataset. Focus on cleaning, encoding, and preparing your data for logistic regression analysis.
- 2. Fit a logistic regression model on a chosen dataset and interpret the coefficients to explain how each predictor influences the outcome.

### Discussion Questions
- Discuss how the features selected for a logistic regression model can impact the outcomes. Which features do you think would be most predictive?
- In what other contexts might logistic regression be useful beyond customer churn prediction?

---

## Section 13: Handling Multiclass Classification

### Learning Objectives
- Understand concepts from Handling Multiclass Classification

### Activities
- Practice exercise for Handling Multiclass Classification

### Discussion Questions
- Discuss the implications of Handling Multiclass Classification

---

## Section 14: Challenges and Limitations

### Learning Objectives
- Recognize common challenges associated with logistic regression.
- Explore solutions and best practices to mitigate these challenges.
- Identify how to properly interpret the results from logistic regression analyses.

### Assessment Questions

**Question 1:** What does logistic regression primarily assume about the relationship between independent variables and the dependent variable?

  A) It assumes a non-linear relationship.
  B) It assumes independence of all observations.
  C) It assumes a linear relationship with the log odds.
  D) It assumes that all predictors have equal influence.

**Correct Answer:** C
**Explanation:** Logistic regression assumes that the relationship between independent variables and the log odds of the dependent variable is linear.

**Question 2:** What is a sign that multicollinearity is likely present among predictors in a dataset?

  A) Low variance inflation factor (VIF) values.
  B) High correlation between independent variables.
  C) High sample size.
  D) Decreased model complexity.

**Correct Answer:** B
**Explanation:** Multicollinearity is evident when independent variables are highly correlated, which often translates to inflated VIF values.

**Question 3:** What could be a potential outcome of not addressing outliers in a logistic regression model?

  A) Improved predictions.
  B) Biased coefficient estimates.
  C) Increased interpretability.
  D) Reduction in model complexity.

**Correct Answer:** B
**Explanation:** Outliers can disproportionately influence the regression results, leading to biased coefficient estimates that distort the model's effectiveness.

**Question 4:** How can you address class imbalance in a logistic regression model?

  A) Expand the dataset by removing majority class instances.
  B) Use regularization methods.
  C) Utilize oversampling or undersampling techniques.
  D) Increase the number of predictors.

**Correct Answer:** C
**Explanation:** Addressing class imbalance can be effectively done through techniques like oversampling the minority class or undersampling the majority class to create a more balanced dataset.

### Activities
- Choose a dataset and run a logistic regression analysis. Identify at least three challenges you encountered and propose solutions for each. Present your findings in a short report.
- Analyze a given logistic regression model output to identify signs of multicollinearity or the influence of outliers. Suggest potential actions to mitigate these issues.

### Discussion Questions
- What are the implications of violating the independence assumption in logistic regression?
- How would you explain the effects of multicollinearity to someone not familiar with regression techniques?
- Discuss how the interpretability of a model decreases as complexity increases. What are the trade-offs involved?

---

## Section 15: Future of Logistic Regression and Trends

### Learning Objectives
- Discuss how logistic regression is evolving with new methodologies.
- Identify trends that are shaping the future of logistic regression.
- Evaluate the impact of advancements such as AutoML and interpretability tools on logistic regression.

### Assessment Questions

**Question 1:** Which emerging trend enhances logistic regression's interpretability?

  A) Use of larger datasets
  B) Incorporation of regularization techniques
  C) Utilization of SHAP and LIME techniques
  D) Integration with decision trees

**Correct Answer:** C
**Explanation:** SHAP and LIME are tools designed to enhance the interpretability of machine learning models, including logistic regression.

**Question 2:** How does automated machine learning (AutoML) impact logistic regression?

  A) It replaces logistic regression with deep learning entirely.
  B) It automates model selection and tuning for logistic regression models.
  C) It limits the use of logistic regression to small datasets.
  D) It focuses solely on feature engineering methods.

**Correct Answer:** B
**Explanation:** AutoML tools optimize the selection and tuning of logistic regression models, enabling efficient model building without extensive manual input.

**Question 3:** Which of the following is a challenge facing the future of logistic regression?

  A) Ensuring algorithm accuracy on small datasets.
  B) Handling bias and fairness in model predictions.
  C) Decreasing computational demands.
  D) Reducing reliance on external data sources.

**Correct Answer:** B
**Explanation:** As logistic regression models become more complex, ensuring fairness and mitigating bias in predictions is a significant challenge.

**Question 4:** What benefit does regularization bring to logistic regression?

  A) Increases model interpretability
  B) Reduces the risk of overfitting in high-dimensional data
  C) Simplifies model construction
  D) Maximizes predictive accuracy without constraints

**Correct Answer:** B
**Explanation:** Regularization techniques like Lasso and Ridge help manage overfitting in logistic regression models, particularly when dealing with many predictors.

### Activities
- Conduct a literature review on a current trend in logistic regression research such as the integration with deep learning. Prepare a 5-minute presentation to share your findings with the class.
- Experiment with an AutoML tool like H2O.ai or Google AutoML. Choose a dataset and evaluate how the tool selects and optimizes a logistic regression model.

### Discussion Questions
- In what scenarios do you think integrating external data would significantly improve a logistic regression model?
- How can we ensure the robustness of logistic regression models against bias and ethical concerns in AI?

---

## Section 16: Summary and Key Takeaways

### Learning Objectives
- Recap important concepts covered in the logistic regression module.
- Explain the implications of these concepts for data mining practices.

### Assessment Questions

**Question 1:** What is the primary purpose of logistic regression?

  A) To predict probabilities of binary outcomes.
  B) To make predictions only on large datasets.
  C) To classify data into multiple categories.
  D) To perform clustering on datasets.

**Correct Answer:** A
**Explanation:** The primary purpose of logistic regression is to predict probabilities that represent binary outcomes, not to classify data into multiple categories or cluster datasets.

**Question 2:** How does a positive coefficient in logistic regression affect the predicted outcome?

  A) It decreases the probability of the event occurring.
  B) It has no effect on the predicted outcome.
  C) It increases the likelihood of the event occurring.
  D) It produces a random effect.

**Correct Answer:** C
**Explanation:** A positive coefficient in logistic regression indicates that as the predictor variable increases, the likelihood of the event occurring also increases.

**Question 3:** What evaluation metric can be used to visualize a model's performance?

  A) ROC Curve
  B) Decision Tree
  C) Confusion Matrix
  D) Histogram

**Correct Answer:** C
**Explanation:** A confusion matrix is used to visualize the performance of a classification model, showing the counts of true positives, false positives, true negatives, and false negatives.

**Question 4:** Which of the following applications can logistic regression be used for?

  A) Predicting customer churn.
  B) Forecasting sales for multiple product categories.
  C) Calculating the median of a dataset.
  D) Clustering similar users into groups.

**Correct Answer:** A
**Explanation:** Logistic regression is well-suited for predicting binary outcomes, such as customer churn, making it applicable in business analytics.

### Activities
- Create a case study report detailing the application of logistic regression in a field of your choice (e.g., healthcare, marketing), including the dataset sourced, variables analyzed, and insights gained.

### Discussion Questions
- Discuss how logistic regression might change with the advent of new AI techniques and big data tools. How would you adapt its applications?
- In what ways do you think the interpretability of coefficients in logistic regression contributes to its utility in decision-making?

---

