# Assessment: Slides Generation - Weeks 4-8: Supervised Learning Techniques

## Section 1: Introduction to Supervised Learning

### Learning Objectives
- Understand the definition and principles of supervised learning.
- Recognize the importance of labeled data in training machine learning models.
- Identify real-world applications and use cases for supervised learning.

### Assessment Questions

**Question 1:** What is supervised learning?

  A) Learning from unlabeled data
  B) Learning with labeled data
  C) Learning without human feedback
  D) Learning through reinforcement

**Correct Answer:** B
**Explanation:** Supervised learning involves training a model on a labeled dataset.

**Question 2:** Which of the following accurately defines labeled data?

  A) Data with no specific output
  B) Input data paired with corresponding output labels
  C) Only numerical data without any labels
  D) Random data without discernible patterns

**Correct Answer:** B
**Explanation:** Labeled data consists of input features and their corresponding output labels – essential for supervised learning.

**Question 3:** What is a common metric used to evaluate supervised learning models?

  A) Randomness
  B) Time Complexity
  C) Mean Squared Error (MSE)
  D) Labeling Rate

**Correct Answer:** C
**Explanation:** Mean Squared Error (MSE) is a standard metric used to evaluate the performance of regression models in supervised learning.

**Question 4:** In supervised learning, what does the model aim to achieve?

  A) Generate new data points
  B) Map new inputs to correct outputs
  C) Minimize data representation
  D) Remove noise from the dataset

**Correct Answer:** B
**Explanation:** The main goal of supervised learning is to learn a function that maps new input data to the correct output based on the training dataset.

### Activities
- Analyze a dataset of your choice and identify a classification or regression problem. Prepare a summary of how you would approach the supervised learning task, including the choice of algorithm and evaluation metrics.

### Discussion Questions
- What challenges might arise when working with labeled datasets in supervised learning?
- How do you think the choice of evaluation metric can influence the development of a supervised learning model?

---

## Section 2: Key Concepts in Supervised Learning

### Learning Objectives
- Define labeled data and its significance in supervised learning.
- Explain the model training process and its steps in supervised learning.

### Assessment Questions

**Question 1:** What is the primary goal of supervised learning?

  A) To predict outcomes based on input data
  B) To cluster data points
  C) To reduce data dimensionality
  D) To identify anomalies

**Correct Answer:** A
**Explanation:** The primary goal of supervised learning is to predict outcomes based on input data using labeled examples.

**Question 2:** What constitutes labeled data?

  A) Data that is unordered and ungrouped
  B) Data points with only input features
  C) Data points with input features and corresponding output labels
  D) Data that has undergone dimensionality reduction

**Correct Answer:** C
**Explanation:** Labeled data consists of data points that include both the features (inputs) and their corresponding labels (outputs).

**Question 3:** Which step is essential in model training?

  A) Selecting a random algorithm
  B) Iteratively adjusting model parameters based on feedback
  C) Ignoring the output labels
  D) Focusing solely on data visualization

**Correct Answer:** B
**Explanation:** Model training involves iteratively adjusting model parameters based on feedback to improve prediction accuracy.

**Question 4:** Why is quality labeled data important?

  A) It speeds up the learning process
  B) It ensures more accurate predictions
  C) It reduces the computational cost
  D) It simplifies the model architecture

**Correct Answer:** B
**Explanation:** Quality labeled data is critical for the success of supervised learning models as it directly impacts the accuracy of predictions.

### Activities
- Create a simple labeled dataset containing examples of fruits, with features such as color, size, and type. Describe how this dataset could be used in a supervised learning model to predict fruit types.

### Discussion Questions
- How does the quality of labeled data influence the performance of a supervised learning model?
- Can you think of real-world applications where supervised learning is commonly used? Please provide examples.
- What challenges do you foresee in collecting labeled data for a supervised learning problem?

---

## Section 3: Linear Regression

### Learning Objectives
- Understand the basic assumptions of linear regression.
- Identify applications of linear regression in real-world scenarios.
- Interpret coefficients of a linear regression model and their implications.

### Assessment Questions

**Question 1:** Which of the following is an assumption of linear regression?

  A) Independence of observations
  B) Non-linearity of relationships
  C) Homoscedasticity
  D) Both A and C

**Correct Answer:** D
**Explanation:** Linear regression assumes that observations are independent and that there is constant variance (homoscedasticity) of the errors.

**Question 2:** What is the meaning of the term 'homoscedasticity' in the context of linear regression?

  A) The residuals are normally distributed.
  B) The variance of the residuals is constant across all levels of the independent variables.
  C) The independent variables are uncorrelated.
  D) The dependent variable is normally distributed.

**Correct Answer:** B
**Explanation:** Homoscedasticity refers to the property of having constant variance of the residuals at all levels of the independent variables.

**Question 3:** In linear regression, which term represents the error in prediction?

  A) β0
  B) Y
  C) β1
  D) ε

**Correct Answer:** D
**Explanation:** In the regression equation, ε represents the error term, showing the difference between the observed and predicted values.

**Question 4:** What does the slope coefficient (β1, β2, ..., βn) in the linear regression equation indicate?

  A) The value of the dependent variable when all independent variables are zero.
  B) The change in the dependent variable for a one-unit increase in the independent variable.
  C) The total number of independent variables in the model.
  D) The average value of the dependent variable.

**Correct Answer:** B
**Explanation:** The slope coefficients indicate how much the dependent variable is expected to change for each one-unit increase in the respective independent variable.

### Activities
- Implement a simple linear regression model on a small dataset using Python (e.g., using libraries like pandas and scikit-learn) and visualize the results using Matplotlib or Seaborn.
- Collect a dataset related to your interests (e.g., hours studied vs. exam scores, or housing prices based on various features) and conduct a linear regression analysis.

### Discussion Questions
- Discuss a real-world situation where linear regression could be applied and elaborate on the choice of independent variables.
- What challenges do you foresee in meeting the assumptions of linear regression in practical scenarios?

---

## Section 4: Mathematical Foundations of Linear Regression

### Learning Objectives
- Understand concepts from Mathematical Foundations of Linear Regression

### Activities
- Practice exercise for Mathematical Foundations of Linear Regression

### Discussion Questions
- Discuss the implications of Mathematical Foundations of Linear Regression

---

## Section 5: Implementing Linear Regression

### Learning Objectives
- Implement a linear regression model using Python.
- Interpret the output of a linear regression analysis.
- Understand the significance of model evaluation metrics in assessing regression models.

### Assessment Questions

**Question 1:** Which of the following is the dependent variable in the linear regression equation?

  A) β0
  B) y
  C) x1
  D) ε

**Correct Answer:** B
**Explanation:** In the linear regression equation, y represents the dependent variable that we aim to predict based on independent variables.

**Question 2:** What function do you call to split your dataset into training and testing sets?

  A) train_test_split()
  B) split_data()
  C) create_sets()
  D) correlation()

**Correct Answer:** A
**Explanation:** The train_test_split() function from Scikit-learn is utilized to split the dataset into training and testing sets.

**Question 3:** What metric is used to evaluate the accuracy of a regression model?

  A) Accuracy Score
  B) R-squared
  C) Precision
  D) F1 Score

**Correct Answer:** B
**Explanation:** R-squared is a common metric used to evaluate the fit of a regression model, indicating how much variation in the dependent variable is explained by the independent variables.

**Question 4:** Which of the following libraries is NOT typically used for implementing linear regression in Python?

  A) Scikit-learn
  B) TensorFlow
  C) Numpy
  D) Requests

**Correct Answer:** D
**Explanation:** Requests is a library for making HTTP requests in Python and is not used for implementing linear regression.

### Activities
- Complete a hands-on tutorial using a dataset of your choice to implement linear regression in Python. Evaluate the model's performance and plot the regression line on a graph.

### Discussion Questions
- What are the potential limitations of using linear regression for predictive modeling?
- In what scenarios would you choose linear regression over more complex algorithms?
- How do you handle outliers when performing linear regression, and why is it important?

---

## Section 6: Logistic Regression

### Learning Objectives
- Understand concepts from Logistic Regression

### Activities
- Practice exercise for Logistic Regression

### Discussion Questions
- Discuss the implications of Logistic Regression

---

## Section 7: Understanding the Logistic Function

### Learning Objectives
- Understand concepts from Understanding the Logistic Function

### Activities
- Practice exercise for Understanding the Logistic Function

### Discussion Questions
- Discuss the implications of Understanding the Logistic Function

---

## Section 8: Implementing Logistic Regression

### Learning Objectives
- Implement a logistic regression model using Scikit-learn.
- Evaluate the performance of a logistic regression model using accuracy and confusion matrix metrics.
- Understand the role of the logistic function and decision boundary in the context of binary classification.

### Assessment Questions

**Question 1:** Which function in Scikit-learn is used for logistic regression?

  A) LinearRegression()
  B) LogisticRegression()
  C) REGRESSION()
  D) SVC()

**Correct Answer:** B
**Explanation:** The LogisticRegression() function in Scikit-learn is specifically designed for logistic regression tasks.

**Question 2:** What is the range of the logistic (sigmoid) function?

  A) 0 to 1
  B) -1 to 1
  C) -∞ to +∞
  D) 0 to +∞

**Correct Answer:** A
**Explanation:** The logistic function outputs values between 0 and 1, making it suitable for modeling probabilities.

**Question 3:** What is the purpose of splitting the dataset into training and testing sets?

  A) To increase dataset size
  B) To evaluate the model's performance on unseen data
  C) To simplify data loading
  D) To ensure all features are used

**Correct Answer:** B
**Explanation:** Splitting the dataset allows us to test the model's performance on new, unseen data, which provides a better estimate of its generalization ability.

**Question 4:** What does the confusion matrix indicate?

  A) The speed of the model
  B) The model's accuracy compared to other models
  C) The counts of true positive, true negative, false positive, and false negative predictions
  D) The optimal decision boundary

**Correct Answer:** C
**Explanation:** A confusion matrix summarizes the results of predictions by showing true positives, true negatives, false positives, and false negatives.

### Activities
- Using a provided dataset, create a logistic regression model to predict a binary outcome. Preprocess your data as needed and evaluate the model's performance using a confusion matrix.

### Discussion Questions
- Why is logistic regression favored for binary classification tasks?
- What challenges might you face when using logistic regression with real-world data?
- How do you think feature selection impacts the accuracy of a logistic regression model?

---

## Section 9: Decision Trees

### Learning Objectives
- Understand the structure and function of decision trees.
- Identify the advantages and limitations of decision trees.
- Learn how decision trees partition data based on feature values.

### Assessment Questions

**Question 1:** What does the root node in a decision tree represent?

  A) The final classification or output
  B) A feature used for splitting the data
  C) The entire dataset before any splits
  D) An intermediate decision point

**Correct Answer:** C
**Explanation:** The root node represents the entire dataset before any splits occur in the decision tree.

**Question 2:** Which of the following is a method used to determine how to split the data in decision trees?

  A) Mean Squared Error
  B) Entropy
  C) Cross-validation
  D) Gradient Descent

**Correct Answer:** B
**Explanation:** Entropy is one of the criteria used to determine how to make splits in decision tree algorithms.

**Question 3:** What is the purpose of pruning in decision trees?

  A) To increase the size of the tree
  B) To improve interpretability and reduce overfitting
  C) To visualize the tree better
  D) To identify the root node

**Correct Answer:** B
**Explanation:** Pruning is used to remove branches that have little importance, thereby helping to simplify the model and improve generalization.

**Question 4:** Which of the following statements about decision trees is true?

  A) They can only be used for regression tasks.
  B) They can create complex models easily with many splits.
  C) They do not handle missing values well.
  D) They are not suitable for categorical data.

**Correct Answer:** B
**Explanation:** Decision trees can indeed create complex models with many splits, and they can handle both categorical and continuous data.

### Activities
- Using a given dataset, construct a simple decision tree to classify data points based on specified features. Document the decision process at each node.

### Discussion Questions
- Discuss how decision tree models can be interpreted by stakeholders compared to other machine learning models.
- In what scenarios would you prefer to use a decision tree over other models such as random forests or support vector machines?

---

## Section 10: Building Decision Trees

### Learning Objectives
- Explain the splitting criteria used in decision trees, including Gini impurity and entropy.
- Understand how tree pruning can improve model performance and reduce overfitting.

### Assessment Questions

**Question 1:** Which criterion is commonly used to evaluate splits in decision trees?

  A) Mean Absolute Error
  B) Gini impurity
  C) Standard deviation
  D) R-squared

**Correct Answer:** B
**Explanation:** Gini impurity is a common criterion used to measure the quality of a split in decision trees.

**Question 2:** What does a lower Gini impurity value indicate?

  A) Higher impurity in the dataset
  B) More class overlap
  C) Greater class separation
  D) Increased model complexity

**Correct Answer:** C
**Explanation:** A lower Gini impurity value indicates greater class separation, meaning that the nodes are more 'pure'.

**Question 3:** What is the main purpose of tree pruning?

  A) To increase the depth of the tree
  B) To remove nodes that contribute little to predictive accuracy
  C) To simplify the tree building process
  D) To enhance the feature selection process

**Correct Answer:** B
**Explanation:** The main purpose of tree pruning is to remove branches that have little importance and reduce overfitting.

**Question 4:** Which of the following statements is true regarding entropy?

  A) Higher values indicate higher purity
  B) It measures class distribution uncertainty
  C) It is computationally less expensive than Gini impurity
  D) It cannot be calculated for continuous data

**Correct Answer:** B
**Explanation:** Entropy measures the uncertainty in the dataset. A higher entropy indicates more uncertainty in class distribution.

### Activities
- Implement a decision tree classifier using Scikit-learn with a provided dataset. Visualize the decision tree and interpret the results.
- Conduct an experiment by modifying parameters in a decision tree model (e.g., max depth, splitting criteria) and observe how these changes affect the model's performance.

### Discussion Questions
- How would the choice between Gini impurity and entropy affect the construction of a decision tree?
- What are the trade-offs between pre-pruning and post-pruning techniques?

---

## Section 11: Ensemble Methods

### Learning Objectives
- Define ensemble methods and explain their contribution to improving model accuracy.
- Differentiate between bagging and boosting techniques and their respective approaches.

### Assessment Questions

**Question 1:** Which of the following is an ensemble method?

  A) Logistic Regression
  B) K-Nearest Neighbors
  C) Random Forests
  D) Single Decision Trees

**Correct Answer:** C
**Explanation:** Random Forests is an ensemble method that combines multiple decision trees.

**Question 2:** What is the primary goal of ensemble methods?

  A) To minimize the dataset size
  B) To combine multiple models for better predictions
  C) To use only the best performing model
  D) To maximize computational efficiency

**Correct Answer:** B
**Explanation:** The primary goal of ensemble methods is to combine predictions from multiple models to improve accuracy.

**Question 3:** What is a characteristic of boosting techniques?

  A) They train models independently and combine predictions
  B) They focus on correcting errors of previous models
  C) They always use the same model type
  D) They are only useful for classification tasks

**Correct Answer:** B
**Explanation:** Boosting techniques sequentially train models that focus on correcting the errors made by previous models.

**Question 4:** In the context of Random Forest, what does 'bagging' refer to?

  A) Using different algorithms for training
  B) Combining predictions from all models without replacement
  C) Sampling multiple subsets of data with replacement
  D) Sequentially adding models to improve predictions

**Correct Answer:** C
**Explanation:** Bagging, or Bootstrap Aggregating, involves creating multiple subsets of the dataset through sampling with replacement.

### Activities
- Experiment with Random Forests on a dataset and compare its performance with a single decision tree.
- Implement a Boosting algorithm, such as AdaBoost or Gradient Boosting, on a real-world dataset and evaluate its performance.
- Visualize the predictions from a Random Forest model versus a single Decision Tree model to observe the differences in accuracy and robustness.

### Discussion Questions
- How can ensemble methods be applied in real-world applications? Provide examples.
- Discuss the scenarios in which using an ensemble method would be more advantageous than a single model.
- What are the limitations of ensemble methods, and how can they be addressed?

---

## Section 12: Introduction to Neural Networks

### Learning Objectives
- Understand the architecture of neural networks.
- Identify key terms such as nodes, layers, and activation functions.
- Explain the learning process of neural networks, including forward and backward passes.

### Assessment Questions

**Question 1:** What is a key component of neural networks?

  A) Nodes
  B) Matrices
  C) Clusters
  D) All of the above

**Correct Answer:** A
**Explanation:** Nodes (or neurons) are the fundamental building blocks of neural networks.

**Question 2:** What function is commonly used to introduce non-linearity in a neural network?

  A) Linear function
  B) Activation function
  C) Loss function
  D) Weights

**Correct Answer:** B
**Explanation:** Activation functions, such as ReLU and Sigmoid, help introduce non-linearity, allowing neural networks to model complex relationships.

**Question 3:** What does the output layer in a neural network provide?

  A) Input features
  B) Intermediate computations
  C) Final predictions or classifications
  D) Bias adjustments

**Correct Answer:** C
**Explanation:** The output layer is responsible for returning the final prediction or classification based on the computations from the previous layers.

**Question 4:** During the backward pass of training, what is primarily calculated to update the network's weights?

  A) Inputs
  B) Predictions
  C) Gradients
  D) Outputs

**Correct Answer:** C
**Explanation:** Gradients are calculated during the backward pass to determine how the weights should be adjusted to minimize the loss function.

### Activities
- Design a simple neural network architecture for a classification problem, specifying the number of input, hidden, and output layers.
- Implement a basic neural network using a programming language of your choice (e.g., Python). Define the architecture and activation functions.

### Discussion Questions
- How do activation functions affect the performance of a neural network?
- Discuss the implications of choosing too many or too few hidden layers in a neural network architecture.
- What are some real-world problems that neural networks are particularly well-suited to solve?

---

## Section 13: Training Neural Networks

### Learning Objectives
- Explain the core components of backpropagation and its significance in neural network training.
- Identify and compare different activation functions and their roles in neural networks.
- Describe various optimization algorithms and evaluate their impact on model training and performance.

### Assessment Questions

**Question 1:** What is backpropagation used for in neural networks?

  A) Initializing weights
  B) Updating weights during training
  C) Forward pass computation
  D) None of the above

**Correct Answer:** B
**Explanation:** Backpropagation is used to update the weights of the model during training to minimize the loss.

**Question 2:** Which of the following activation functions outputs values in the range (0, 1)?

  A) Softmax
  B) ReLU
  C) Sigmoid
  D) Tanh

**Correct Answer:** C
**Explanation:** The Sigmoid function outputs values between 0 and 1, making it suitable for binary classification outputs.

**Question 3:** What is the primary purpose of optimization algorithms in training neural networks?

  A) To visualize data
  B) To adjust the structure of the network
  C) To update the weights based on loss gradients
  D) To calculate the loss function

**Correct Answer:** C
**Explanation:** Optimization algorithms are crucial for updating the weights of the network based on calculated loss gradients during training.

**Question 4:** Which optimization algorithm maintains an exponentially decaying moving average of past gradients?

  A) Stochastic Gradient Descent
  B) Adam
  C) Momentum
  D) Adagrad

**Correct Answer:** B
**Explanation:** Adam optimizer combines the benefits of both AdaGrad and RMSProp by keeping a moving average of the gradients and the squared gradients.

### Activities
- Implement the backpropagation algorithm from scratch for a basic neural network using a small dataset, and compare the results with those obtained using a built-in machine learning framework.
- Create a graph to visualize how different activation functions shape outputs in a simple neural network over a range of inputs.

### Discussion Questions
- How does the choice of activation function impact the learning process of a neural network?
- In what scenarios would you prefer one optimization algorithm over others?
- Can you think of real-world applications where the ability to train a neural network effectively is crucial?

---

## Section 14: Model Evaluation Metrics

### Learning Objectives
- Understand various evaluation metrics used in model assessment.
- Apply these metrics to evaluate a classification model.
- Differentiate between accuracy, precision, recall, F1 score, and ROC-AUC.

### Assessment Questions

**Question 1:** Which metric is commonly used to evaluate the quality of a classification model?

  A) Mean Squared Error
  B) Precision
  C) Root Mean Squared Error
  D) R-squared

**Correct Answer:** B
**Explanation:** Precision is a key metric to evaluate the performance of classification models.

**Question 2:** What does a high recall indicate in a classification model?

  A) The model has a high false positive rate.
  B) The model successfully captures most of the actual positive instances.
  C) The model achieves a high overall accuracy.
  D) The model has low precision.

**Correct Answer:** B
**Explanation:** High recall indicates that a model is successful at identifying as many relevant instances as possible.

**Question 3:** The F1 Score is designed to find a balance between which two metrics?

  A) Accuracy and Recall
  B) Precision and Recall
  C) Precision and Specificity
  D) Recall and Specificity

**Correct Answer:** B
**Explanation:** The F1 Score is the harmonic mean of precision and recall, balancing the two metrics.

**Question 4:** What does an ROC-AUC value of 0.85 signify?

  A) The model performs poorly and is almost random.
  B) The model has some capability to distinguish between classes.
  C) The model perfectly discriminates between positive and negative instances.
  D) The model has a moderate to high performance in distinguishing classes.

**Correct Answer:** D
**Explanation:** An AUC of 0.85 suggests that the model performs well at distinguishing between positive and negative instances.

### Activities
- Given the following confusion matrix: TP=50, TN=30, FP=10, FN=10, calculate the accuracy, precision, recall, and F1-score.

### Discussion Questions
- In what scenarios would you prioritize recall over precision?
- How would the choice of metric affect model selection in imbalanced datasets?
- Can you provide examples of real-world applications where F1 Score is more relevant than accuracy?

---

## Section 15: Practical Applications of Supervised Learning

### Learning Objectives
- Recognize various fields that are utilizing supervised learning techniques.
- Develop a conceptual project idea that employs supervised learning methods.

### Assessment Questions

**Question 1:** Which application of supervised learning is related to predicting financial risk?

  A) Disease Diagnosis
  B) Credit Scoring
  C) Churn Prediction
  D) Predictive Maintenance

**Correct Answer:** B
**Explanation:** Credit Scoring involves evaluating the likelihood of a customer defaulting on a loan, which directly relates to financial risk assessment.

**Question 2:** What type of supervised learning technique would be most appropriate for predicting customer churn?

  A) Regression
  B) Clustering
  C) Classification
  D) Dimensionality Reduction

**Correct Answer:** C
**Explanation:** Classification techniques are used for predicting categorical outcomes, such as whether a customer is likely to churn or not.

**Question 3:** In the context of predicting loan defaults, what does the logistic regression formula describe?

  A) The relationship between features and an input variable.
  B) The probability of a binary outcome.
  C) The clustering of customer data.
  D) The process of data preprocessing.

**Correct Answer:** B
**Explanation:** The logistic regression formula describes the probability that a binary outcome (like loan default) occurs based on input features.

**Question 4:** Which of the following techniques is NOT typically used in supervised learning?

  A) Random Forest
  B) Support Vector Machines
  C) K-means Clustering
  D) Decision Trees

**Correct Answer:** C
**Explanation:** K-means Clustering is an unsupervised learning technique, while the others are supervised learning algorithms.

### Activities
- Identify a project in a domain of your choice (e.g., healthcare, finance, marketing) where supervised learning could provide insights. Outline a potential approach, including the type of supervised learning model and the data features you would use.

### Discussion Questions
- How might the choice of supervised learning model impact the results in a business scenario?
- What ethical considerations should be taken into account when applying supervised learning in real-world applications?

---

## Section 16: Ethical Considerations in Supervised Learning

### Learning Objectives
- Identify ethical challenges in supervised learning.
- Understand the importance of addressing bias in machine learning models.
- Learn about tools for improving model explainability.
- Recognize the impact of regulations on data privacy.

### Assessment Questions

**Question 1:** What is a major ethical concern in deploying supervised learning models?

  A) Data privacy
  B) Model interpretability
  C) Possible bias in data
  D) All of the above

**Correct Answer:** D
**Explanation:** Ethical considerations in supervised learning include data privacy, model bias, and interpretability.

**Question 2:** Which tool can be used to enhance explainability in machine learning models?

  A) TensorFlow
  B) LIME
  C) PyTorch
  D) Keras

**Correct Answer:** B
**Explanation:** LIME (Local Interpretable Model-agnostic Explanations) is a tool that enhances model explainability.

**Question 3:** How can bias in supervised learning models be mitigated?

  A) By ignoring the training data
  B) By regularly auditing datasets
  C) By increasing model complexity
  D) By limiting data usage

**Correct Answer:** B
**Explanation:** Regularly auditing datasets helps identify and mitigate any inherent biases.

**Question 4:** Which of the following regulations is important for data privacy?

  A) HIPAA
  B) GDPR
  C) CCPA
  D) All of the above

**Correct Answer:** D
**Explanation:** All of these regulations are important for ensuring data privacy across different contexts.

### Activities
- Create a hypothetical scenario where a supervised learning model is applied, and identify potential ethical issues that may arise.
- Review a dataset used in a public machine learning model and identify any potential biases present in the data.

### Discussion Questions
- What steps can organizations take to ensure accountability for decisions made by supervised learning models?
- Can you think of an example where a lack of transparency in machine learning led to public backlash? What could have been done differently?

---

