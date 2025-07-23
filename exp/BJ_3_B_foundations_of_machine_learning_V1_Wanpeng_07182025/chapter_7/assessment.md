# Assessment: Slides Generation - Week 7: Feature Engineering and Selection

## Section 1: Introduction to Feature Engineering

### Learning Objectives
- Understand the role of features in machine learning.
- Recognize the significance of feature engineering in enhancing model performance.
- Identify different types of features and how they impact learning algorithms.

### Assessment Questions

**Question 1:** Why are features important in machine learning models?

  A) They help in data compression
  B) They directly impact model performance
  C) They are used for data visualization
  D) They do not play a significant role

**Correct Answer:** B
**Explanation:** Features are the inputs to machine learning models and can greatly affect their performance.

**Question 2:** What is a categorical feature?

  A) A feature that represents numerical values
  B) A feature that represents distinct groups or categories
  C) A feature that measures continuous data
  D) A feature that is always binary

**Correct Answer:** B
**Explanation:** Categorical features are qualitative variables that represent distinct groups or categories, such as color or type.

**Question 3:** What is the purpose of dimensionality reduction in feature engineering?

  A) To increase the number of features in the model
  B) To reduce the complexity of the model and improve performance
  C) To eliminate the need for data cleaning
  D) To improve data visualization

**Correct Answer:** B
**Explanation:** Dimensionality reduction helps simplify the model by focusing on the most relevant features, thus improving performance.

**Question 4:** Which of the following is an example of an engineered feature?

  A) The square root of a numerical feature
  B) The original feature itself
  C) A feature representing a unique ID
  D) Any raw data input

**Correct Answer:** A
**Explanation:** Creating the square root of a numerical feature is an example of engineering a new feature from existing data.

### Activities
- Pick a dataset and identify at least three potential features that could be engineered to improve model predictions. Describe how each feature might be constructed.

### Discussion Questions
- What challenges do you face when selecting features for a new machine learning project?
- Can you think of a real-world scenario where feature engineering could significantly impact the outcome of a prediction? Discuss.

---

## Section 2: Understanding Features

### Learning Objectives
- Define what features are and explain their significance in machine learning.
- Analyze how different features can impact the performance of a model.

### Assessment Questions

**Question 1:** What is a feature in the context of machine learning?

  A) An algorithm
  B) A parameter of the model
  C) A measurable property or characteristic
  D) A type of model

**Correct Answer:** C
**Explanation:** A feature is a measurable property or characteristic used in machine learning.

**Question 2:** Why are good features important in a machine learning model?

  A) They reduce the complexity of the model
  B) They help the model discern patterns effectively
  C) They determine the number of parameters in the model
  D) They increase the runtime of the model

**Correct Answer:** B
**Explanation:** Good features are important because they help the model identify relevant patterns in the data, leading to better predictions.

**Question 3:** What is feature engineering?

  A) The process of creating a model
  B) The process of selecting features for a model
  C) The process of transforming raw data into meaningful features
  D) The process of evaluating a model's performance

**Correct Answer:** C
**Explanation:** Feature engineering is the process of transforming raw data into meaningful features suitable for modeling.

**Question 4:** What could potentially happen if irrelevant or noisy features are included in a model?

  A) The model's performance will improve
  B) The model may experience overfitting
  C) The model's predictions will always be accurate
  D) The model will require less training data

**Correct Answer:** B
**Explanation:** Including irrelevant or noisy features may lead to overfitting, where the model learns noise from the training data instead of general patterns.

### Activities
- Select a public dataset (like housing prices or user engagement metrics). Identify and list at least 5 features that could be meaningful for a predictive model. Justify the choice of each feature.

### Discussion Questions
- How does the choice of features affect the interpretability of a machine learning model?
- What strategies can be employed to deal with irrelevant features in a dataset?

---

## Section 3: Feature Types

### Learning Objectives
- Identify different types of features in datasets.
- Understand when to use each feature type in machine learning tasks.
- Develop skills in feature engineering and selection based on feature types.

### Assessment Questions

**Question 1:** Which of the following is NOT a type of feature?

  A) Numerical
  B) Categorical
  C) Visual
  D) Textual

**Correct Answer:** C
**Explanation:** Visual is not a standard category for features; it is usually classified as image features.

**Question 2:** What type of feature is 'Gender' categorized as?

  A) Numerical
  B) Categorical
  C) Textual
  D) Image

**Correct Answer:** B
**Explanation:** Gender is a categorical feature as it represents groups with no inherent numeric value.

**Question 3:** Which technique is commonly used for encoding categorical features?

  A) Min-Max scaling
  B) One-hot encoding
  C) Normalization
  D) PCA

**Correct Answer:** B
**Explanation:** One-hot encoding is a technique used to convert categorical features into a numerical format suitable for machine learning algorithms.

**Question 4:** What processing technique is typically used for textual features?

  A) Principal Component Analysis (PCA)
  B) Term Frequency-Inverse Document Frequency (TF-IDF)
  C) K-means clustering
  D) Decision Trees

**Correct Answer:** B
**Explanation:** TF-IDF is a common technique used to transform text into a format that can be understood by machine learning algorithms.

**Question 5:** Which feature type requires convolutional neural networks for automatic feature extraction?

  A) Numerical
  B) Categorical
  C) Textual
  D) Image

**Correct Answer:** D
**Explanation:** Image features are often processed using convolutional neural networks (CNNs) to automatically learn and extract spatial hierarchies of features.

### Activities
- In groups, categorize the following features into numerical, categorical, textual, or image: 'Height', 'Review text', 'Income', 'Email address', 'Photo of a dog'.

### Discussion Questions
- What challenges might arise when converting textual data into numerical format?
- How does the type of feature influence the choice of machine learning model?
- Discuss scenarios where categorical features may have ordinal implications.

---

## Section 4: Feature Extraction Techniques

### Learning Objectives
- Understand various feature extraction techniques.
- Differentiate between supervised and unsupervised dimensionality reduction methods.
- Apply at least one feature extraction method practically using a data analysis tool.

### Assessment Questions

**Question 1:** What does PCA stand for?

  A) Principal Component Analysis
  B) Principal Cluster Analysis
  C) Primary Component Analysis
  D) Principal Collection Analysis

**Correct Answer:** A
**Explanation:** PCA stands for Principal Component Analysis, a technique used for reducing the dimensionality of data.

**Question 2:** Which method is used for supervised dimensionality reduction?

  A) PCA
  B) LDA
  C) TF-IDF
  D) Clustering

**Correct Answer:** B
**Explanation:** LDA, or Linear Discriminant Analysis, is a supervised dimensionality reduction technique that utilizes class labels to help separate different classes.

**Question 3:** What does TF-IDF emphasize when processing documents?

  A) Common words in all documents
  B) Unique words in each document
  C) Length of documents
  D) Number of documents

**Correct Answer:** B
**Explanation:** TF-IDF helps identify words that are more significant in distinguishing documents by emphasizing those that are unique to each document.

**Question 4:** Which step is NOT part of PCA?

  A) Standardization of data
  B) Computing class means
  C) Calculating covariance matrix
  D) Projecting data onto principal components

**Correct Answer:** B
**Explanation:** Computing class means is part of LDA, not PCA. PCA focuses on variance and dimensionality reduction.

### Activities
- Implement PCA on a sample dataset using Python and visualize the explained variance.
- Use LDA to classify a two-class problem with real-world data in a notebook environment, and assess the classification accuracy.
- Calculate the TF-IDF values for a small set of sample documents to identify key terms.

### Discussion Questions
- How does PCA help in enhancing model performance? Can you think of scenarios where PCA might not be effective?
- Discuss a case where LDA might perform better than PCA. What are the advantages of using LDA?
- How could TF-IDF be applied in search engines? In what ways can it influence search results?

---

## Section 5: Feature Selection

### Learning Objectives
- Define feature selection and its significance in machine learning.
- Identify techniques for selecting relevant features and explain their importance.

### Assessment Questions

**Question 1:** What is the primary goal of feature selection?

  A) Increase the number of features
  B) Reduce overfitting and improve performance
  C) Increase computation time
  D) Simplify models

**Correct Answer:** B
**Explanation:** Feature selection aims to reduce overfitting and improve model performance by selecting relevant features.

**Question 2:** Why is reducing the number of features important in machine learning?

  A) It always increases model accuracy
  B) It decreases the amount of data to process
  C) It can help with model interpretability
  D) Both B and C

**Correct Answer:** D
**Explanation:** Reducing features decreases the processing load and enhances model interpretability since simpler models are easier to understand.

**Question 3:** Choosing irrelevant features for a model can lead to which of the following?

  A) Improved model performance
  B) Overfitting
  C) More accurate predictions
  D) Faster computation

**Correct Answer:** B
**Explanation:** Including irrelevant features can cause a model to overfit by learning noise in the data rather than the relevant underlying patterns.

**Question 4:** Which of the following best describes redundancy in feature selection?

  A) A feature that is crucial for prediction
  B) A feature that adds little value as it is similar to another feature
  C) A feature that is always relevant
  D) A feature that improves model interpretability

**Correct Answer:** B
**Explanation:** Redundant features provide little additional value for prediction when other features are already included, potentially complicating the model.

### Activities
- Conduct a feature selection exercise using a given dataset. Use techniques such as correlation analysis to identify and justify the selection of relevant features.

### Discussion Questions
- How might the inclusion of irrelevant features impact the interpretability of a model?
- Can you think of scenarios in your field where feature selection may play a crucial role? Discuss.

---

## Section 6: Feature Selection Techniques

### Learning Objectives
- Differentiate between various feature selection techniques.
- Evaluate the effectiveness of each method in practical scenarios.
- Understand the advantages and limitations of feature selection methods.

### Assessment Questions

**Question 1:** Which of the following is a type of feature selection method?

  A) Filter Methods
  B) Wrapper Methods
  C) Embedded Methods
  D) All of the above

**Correct Answer:** D
**Explanation:** All options represent popular types of feature selection methods.

**Question 2:** What is a main advantage of Wrapper Methods?

  A) They are computationally inexpensive.
  B) They can be model-specific, often yielding better accuracy.
  C) They assess the relevance of features independent of any model.
  D) They quickly eliminate features.

**Correct Answer:** B
**Explanation:** Wrapper Methods can be model-specific and often yield better accuracy because they evaluate feature subsets based on specific model performance.

**Question 3:** Which technique is commonly used in Filter Methods?

  A) Forward Selection
  B) Correlation Coefficients
  C) Recursive Feature Elimination
  D) Lasso Regression

**Correct Answer:** B
**Explanation:** Correlation Coefficients are a common technique in Filter Methods used to evaluate the relationship between features and the target variable.

**Question 4:** What feature selection method integrates the selection process within the model training?

  A) Filter Methods
  B) Wrapper Methods
  C) Embedded Methods
  D) None of the above

**Correct Answer:** C
**Explanation:** Embedded Methods incorporate feature selection as part of the model learning process, assessing feature importance during training.

### Activities
- Choose a feature selection method (Filter, Wrapper, or Embedded) and provide a detailed example of its application on a dataset, including a discussion on its advantages and disadvantages.

### Discussion Questions
- What challenges might you encounter when applying different feature selection techniques to a real-world dataset?
- How might feature selection influence the interpretability of machine learning models?
- In your opinion, which feature selection method is the most effective, and why?

---

## Section 7: Practical Examples of Feature Engineering

### Learning Objectives
- Illustrate real-world applications of feature engineering.
- Analyze the impact of feature engineering on outcomes in various contexts.
- Identify effective feature engineering techniques relevant to specific industry applications.

### Assessment Questions

**Question 1:** What is feature engineering?

  A) The process of collecting large datasets
  B) The methodology of creating features to improve model performance
  C) A way to visualize data
  D) A technique for tuning hyperparameters

**Correct Answer:** B
**Explanation:** Feature engineering involves creating or transforming features to enhance the effectiveness of machine learning algorithms.

**Question 2:** Which of the following is an example of feature engineering in healthcare applications?

  A) Using patient age alone to predict health outcomes
  B) Relying solely on demographic data without contextualization
  C) Creating interaction features between age and cholesterol levels
  D) Ignoring changes in health indicators over time

**Correct Answer:** C
**Explanation:** Combining features such as age and cholesterol levels captures more complex relationships that can enhance prediction accuracy.

**Question 3:** In e-commerce, what type of features can enhance clickthrough rate prediction?

  A) User location
  B) Ad placement and user interaction history
  C) Randomly selected features
  D) Only the ad's visual aesthetics

**Correct Answer:** B
**Explanation:** Ad placement and user interaction history are critical features that directly relate to the likelihood of a user clicking on an ad.

**Question 4:** What is the benefit of creating lag features in finance applications?

  A) They simplify the model structure
  B) They aid in capturing trends over time based on historical behavior
  C) They require less data processing
  D) They eliminate the need for feature selection

**Correct Answer:** B
**Explanation:** Lag features capture historical trends which are essential for predicting future behaviors, such as credit risk.

### Activities
- Design a simple feature engineering strategy for a dataset of your choosing. Specify the features you would create, how you would derive them, and the expected impact on model performance.
- Perform a hands-on exercise in a coding environment where you create interaction features from a predefined dataset using Python and Pandas.

### Discussion Questions
- What are some challenges you face when implementing feature engineering in real-world projects?
- Can you think of a situation where feature engineering significantly improved model performance in your experience or projects?
- How do you determine which features to engineer or derive in a new project?

---

## Section 8: Case Study: Feature Engineering Impact

### Learning Objectives
- Understand the role of feature engineering in enhancing machine learning model performance.
- Identify and apply various feature engineering techniques on real datasets.

### Assessment Questions

**Question 1:** What was the primary goal of the case study discussed in the slide?

  A) Predicting weather patterns
  B) Enhancing model performance for house prices
  C) Evaluating stock market trends
  D) Improving customer satisfaction

**Correct Answer:** B
**Explanation:** The case study focused on enhancing model performance specifically for predicting house prices through effective feature engineering.

**Question 2:** Which feature engineering technique was used to handle missing values in the case study?

  A) Removal of rows with missing data
  B) Imputation using the mean age of similar properties
  C) Filling with zeros
  D) Ignoring the missing values entirely

**Correct Answer:** B
**Explanation:** The case study involved imputing missing values for the 'age of the house' using the mean age of similar properties, which helps maintain data integrity.

**Question 3:** What effect did log transformation have on the house price variable?

  A) Made the data less skewed
  B) Increased complexity
  C) Produced more outliers
  D) Had no effect

**Correct Answer:** A
**Explanation:** Applying log transformation to the price variable reduced skewness and improved the linear relationship between features and the target variable.

**Question 4:** What was the improvement in model accuracy after implementing feature engineering?

  A) From 50% to 70%
  B) From 60% to 85%
  C) No change in accuracy
  D) From 70% to 90%

**Correct Answer:** B
**Explanation:** The model's accuracy improved from 60% to 85% after thoughtful feature engineering was applied.

### Activities
- Choose a dataset of your choice and identify three potential features that could be engineered to improve model performance. Explain your reasoning for each feature.
- Using any sample dataset, perform one-hot encoding and log transformation using Python (Pandas). Provide your code and results.

### Discussion Questions
- What challenges might arise during the feature engineering process, and how can they be overcome?
- Why is domain knowledge important in feature engineering, and how can it impact model outcomes?

---

## Section 9: Challenges in Feature Engineering

### Learning Objectives
- Identify common challenges in feature engineering, including data bias, dimensionality issues, and feature correlation.
- Develop strategies to address these challenges through techniques such as PCA and VIF.

### Assessment Questions

**Question 1:** What is data bias in feature engineering?

  A) Using too many features
  B) When the training dataset does not represent the actual population
  C) Reducing the number of features for model training
  D) Failing to select appropriate algorithms

**Correct Answer:** B
**Explanation:** Data bias occurs when the dataset used for model training does not represent the actual population, leading to skewed predictions.

**Question 2:** What does the 'Curse of Dimensionality' refer to?

  A) The phenomenon where data becomes dense as dimensions increase
  B) The sparsity of data as the number of features increases
  C) Overfitting resulting from too few features
  D) The complexity of selecting the best model

**Correct Answer:** B
**Explanation:** The Curse of Dimensionality refers to the sparsity of data as the number of dimensions (features) increases, complicating learning.

**Question 3:** What technique can help reduce the number of dimensions in data?

  A) Variance Inflation Factor (VIF)
  B) Neural Networks
  C) Principal Component Analysis (PCA)
  D) Cross-Validation

**Correct Answer:** C
**Explanation:** Principal Component Analysis (PCA) is a common technique used to reduce the number of dimensions while retaining significant information.

**Question 4:** Which method can be used to identify correlated features?

  A) Feature Engineering
  B) Variance Inflation Factor (VIF)
  C) Normalization
  D) K-Nearest Neighbors

**Correct Answer:** B
**Explanation:** Variance Inflation Factor (VIF) can be used to identify and measure the extent to which multicollinearity exists among predictor variables.

### Activities
- Using a dataset of your choice, identify instances of data bias. Present how this bias might affect model outcomes and suggest ways to mitigate it.
- Perform a PCA on a high-dimensional dataset to illustrate the reduction in dimensions. Document how dimensionality reduction affects model performance.

### Discussion Questions
- Can you recall a time when you encountered data bias in your own work? How did it shape your approach?
- In your opinion, what is the most challenging aspect of dimensionality in feature engineering, and why?
- How do you think correlated features affect the interpretability of a model?

---

## Section 10: Best Practices in Feature Engineering

### Learning Objectives
- Outline best practices for feature engineering.
- Apply feature engineering techniques in practical scenarios.
- Evaluate the importance of domain knowledge in feature creation.

### Assessment Questions

**Question 1:** Which practice is recommended for effective feature engineering?

  A) Use as many features as possible
  B) Ignore feature correlations
  C) Prioritize features based on domain knowledge
  D) Randomly select features

**Correct Answer:** C
**Explanation:** Prioritizing features based on domain knowledge is critical for effective feature engineering.

**Question 2:** What is an important step in feature selection?

  A) Rely exclusively on model performance
  B) Use statistical tests to filter irrelevant features
  C) Randomly select features for training
  D) Always include all available features

**Correct Answer:** B
**Explanation:** Using statistical tests helps ensure the selected features have meaningful relationships with the target variable.

**Question 3:** How can overfitting be avoided during feature engineering?

  A) By adding more complex features
  B) By ignoring cross-validation
  C) By using regularization techniques
  D) By selecting random features

**Correct Answer:** C
**Explanation:** Regularization techniques like L1 and L2 penalize complexity, helping to prevent overfitting.

**Question 4:** What role does exploratory data analysis (EDA) play in feature engineering?

  A) It is not necessary for feature engineering
  B) It helps to visualize and understand data distributions and relationships
  C) It only focuses on the final model performance
  D) It is used only for final feature selection

**Correct Answer:** B
**Explanation:** EDA is crucial for identifying potential features by understanding the data's distributions and relationships.

### Activities
- Conduct exploratory data analysis on a provided dataset and identify potential features. Then create a list of best practices that you followed during this analysis.
- Implement LASSO regression on a given dataset to perform feature selection and report on the features that were retained.

### Discussion Questions
- What challenges have you faced while performing feature engineering, and how did you overcome them?
- How can the inclusion of domain knowledge change the approach to feature engineering in different industries?

---

## Section 11: Conclusion

### Learning Objectives
- Summarize the key points of feature engineering, including its importance and techniques.
- Understand the broader implications of feature engineering in enhancing machine learning model performance.

### Assessment Questions

**Question 1:** What is the fundamental role of feature engineering in machine learning?

  A) It solely focuses on data cleaning.
  B) It enhances the accuracy and effectiveness of models.
  C) It substitutes the need for model selection.
  D) It is only necessary in supervised learning.

**Correct Answer:** B
**Explanation:** Feature engineering enhances the accuracy and effectiveness of machine learning models by improving the relevance and usefulness of features.

**Question 2:** Which of the following is NOT a benefit of feature selection?

  A) Reduces model complexity.
  B) Avoids overfitting.
  C) Guarantees a faster machine learning algorithm.
  D) Improves interpretability.

**Correct Answer:** C
**Explanation:** While feature selection can improve computational efficiency and reduce processing time, it does not guarantee a faster algorithm as it depends on various factors.

**Question 3:** Why is domain knowledge important in feature engineering?

  A) It allows for better software implementation.
  B) It enables the creation of more relevant and meaningful features.
  C) It reduces the amount of data needed.
  D) It simplifies the feature selection process.

**Correct Answer:** B
**Explanation:** Domain knowledge helps in identifying and creating features that are most relevant to the problem domain, leading to better model performance.

**Question 4:** What is a common technique used for dimensionality reduction in feature selection?

  A) Decision Trees
  B) Recursive Feature Elimination
  C) Linear Regression
  D) Neural Networks

**Correct Answer:** B
**Explanation:** Recursive Feature Elimination is a method for selecting relevant features by recursively removing less important features and assessing the model performance.

### Activities
- Create a feature engineering plan for a dataset of your choice. Identify at least three features you would modify or create to enhance model performance.
- Analyze a dataset and select five features you believe are the most relevant for predicting a specific outcome. Justify your selections.

### Discussion Questions
- What challenges have you faced in feature engineering during your machine learning projects, and how did you address them?
- Can you give an example where feature engineering significantly changed the outcome of a machine learning model?

---

## Section 12: Q&A

### Learning Objectives
- Increase understanding of feature engineering and selection techniques in machine learning.
- Facilitate peer interaction and discussion to address common challenges faced during implementation.
- Clarify any outstanding queries related to the chapter on feature engineering.

### Assessment Questions

**Question 1:** What is the primary goal of feature engineering?

  A) To simplify the model.
  B) To create and modify features that enhance model performance.
  C) To reduce the number of features only.
  D) To visualize data.

**Correct Answer:** B
**Explanation:** Feature engineering involves using domain knowledge to create or modify features that improve the performance of machine learning models.

**Question 2:** Which of the following is an example of a wrapper method for feature selection?

  A) Chi-square Test
  B) Recursive Feature Elimination (RFE)
  C) Lasso regression
  D) Correlation matrix

**Correct Answer:** B
**Explanation:** Recursive Feature Elimination (RFE) is a wrapper method that evaluates feature subsets based on the performance of a specific model.

**Question 3:** One-hot encoding is a technique primarily used for which type of data?

  A) Numerical data
  B) Time-series data
  C) Categorical data
  D) Text data

**Correct Answer:** C
**Explanation:** One-hot encoding is used to convert categorical variables into a format that can be provided to machine learning algorithms to improve predictions.

**Question 4:** What does feature selection aim to accomplish?

  A) Transform all data into numerical form.
  B) Select the most relevant subset of features.
  C) Increase the dimensionality of the dataset.
  D) Eliminate all irrelevant features.

**Correct Answer:** B
**Explanation:** Feature selection is about choosing the most relevant subset of features to improve model performance while possibly reducing overfitting.

### Activities
- Create a small dataset and perform one-hot encoding on it using Python. Present the before and after results to showcase the effects of feature engineering.
- Using a dataset of your choice, implement a feature selection technique (e.g., Recursive Feature Elimination) and discuss the impact it has on your model's performance.

### Discussion Questions
- How have you applied feature engineering in a project before, and what were the outcomes?
- What factors do you consider when deciding which features to engineer or select?

---

## Section 13: Resources and Further Reading

### Learning Objectives
- Foster an ongoing interest in feature engineering and related topics.
- Promote independent learning and resource exploration.
- Enhance practical skills in applying feature engineering and selection techniques.

### Assessment Questions

**Question 1:** Which book focuses specifically on feature engineering techniques and strategies?

  A) The Elements of Statistical Learning
  B) Feature Engineering for Machine Learning
  C) Pattern Recognition and Machine Learning
  D) Feature Selection for High-Dimensional Data

**Correct Answer:** B
**Explanation:** The book 'Feature Engineering for Machine Learning' by Alice Zheng and Amanda Casari provides an overview of feature engineering techniques and strategies.

**Question 2:** What is the primary focus of the online course offered by Coursera?

  A) Neural network optimization
  B) Feature extraction, transformation, and selection techniques
  C) Deep learning applications
  D) Statistical analysis methods

**Correct Answer:** B
**Explanation:** The Coursera course titled 'Feature Engineering for Machine Learning' focuses on hands-on exercises related to feature extraction, transformation, and selection techniques.

**Question 3:** Which method is NOT a feature selection technique mentioned in the slide?

  A) Recursive Feature Elimination (RFE)
  B) Principal Component Analysis (PCA)
  C) LASSO regression
  D) Tree-based methods

**Correct Answer:** B
**Explanation:** While Principal Component Analysis (PCA) is a dimensionality reduction technique, the slide discusses Recursive Feature Elimination (RFE), LASSO regression, and Tree-based methods specifically as feature selection techniques.

### Activities
- Select a dataset from Kaggle or similar platforms and apply feature engineering techniques. Document the methods used and the impact on model performance.
- Write a brief summary of the key takeaways from one of the recommended books or online courses, focusing on a specific feature engineering or selection method.

### Discussion Questions
- What are some challenges you have faced in feature engineering, and how did you address them?
- How does feature engineering differ between various machine learning algorithms?
- Can you think of a scenario where feature selection may not be necessary? Why?

---

