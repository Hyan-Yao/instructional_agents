# Assessment: Slides Generation - Week 4: Supervised Learning - Classification Algorithms

## Section 1: Introduction to Supervised Learning

### Learning Objectives
- Understand concepts from Introduction to Supervised Learning

### Activities
- Practice exercise for Introduction to Supervised Learning

### Discussion Questions
- Discuss the implications of Introduction to Supervised Learning

---

## Section 2: Classification Algorithms Overview

### Learning Objectives
- Identify different classification algorithms used in supervised learning.
- Understand the role of classification algorithms in predictive modeling.
- Differentiate between supervised and unsupervised learning methods.

### Assessment Questions

**Question 1:** Which of the following is a classification algorithm?

  A) Linear Regression
  B) Decision Trees
  C) K-Means Clustering
  D) PCA

**Correct Answer:** B
**Explanation:** Decision Trees are classification algorithms used to predict categorical outcomes.

**Question 2:** What is the primary goal of classification algorithms?

  A) To cluster data points
  B) To reduce dimensionality
  C) To categorize input data into predefined classes
  D) To estimate continuous outcomes

**Correct Answer:** C
**Explanation:** Classification algorithms assign input data points to one of the predefined classes based on learned relationships.

**Question 3:** Which algorithm would you use for a binary classification task?

  A) K-Means Clustering
  B) Decision Trees
  C) Principal Component Analysis
  D) Gaussian Mixture Models

**Correct Answer:** B
**Explanation:** Decision Trees can be used for binary classification tasks as they work well with categorical outcomes.

**Question 4:** In supervised learning, how does the model learn?

  A) By clustering similar data points
  B) By identifying patterns in unlabeled data
  C) By training on labeled datasets containing input-output pairs
  D) By memorizing examples without any feedback

**Correct Answer:** C
**Explanation:** In supervised learning, models learn from labeled datasets that consist of input-output pairs.

### Activities
- Create a mind map illustrating different types of classification algorithms, including examples and use cases.
- Choose one classification algorithm discussed and explain its working mechanism, providing a real-world application for it.

### Discussion Questions
- What are the advantages and disadvantages of using Decision Trees compared to other classification algorithms?
- How can the choice of features impact the performance of classification algorithms?
- Discuss a scenario where classification algorithms could significantly improve decision-making in business.

---

## Section 3: What are Decision Trees?

### Learning Objectives
- Define what decision trees are and their key components.
- Explain the structure and terminology associated with decision trees.

### Assessment Questions

**Question 1:** What is a decision tree primarily used for?

  A) Data preprocessing
  B) Feature selection
  C) Classification and regression
  D) Data visualization

**Correct Answer:** C
**Explanation:** Decision trees can be used for both classification and regression tasks.

**Question 2:** Which of the following represents the topmost node in a decision tree?

  A) Branch
  B) Leaf Node
  C) Internal Node
  D) Root Node

**Correct Answer:** D
**Explanation:** The Root Node is the topmost node in the tree and represents the entire dataset.

**Question 3:** What does the term 'splitting' refer to in decision trees?

  A) Merging nodes
  B) Dividing a node into sub-nodes
  C) Evaluating model performance
  D) Finalizing the model

**Correct Answer:** B
**Explanation:** 'Splitting' refers to the process of dividing a node into two or more sub-nodes based on decision rules applied to the features of the dataset.

**Question 4:** What do leaf nodes in a decision tree represent?

  A) Feature tests
  B) The predicted output
  C) Internal decision rules
  D) Class labels only

**Correct Answer:** B
**Explanation:** Leaf nodes are the endpoints of the tree and indicate the predicted output, which can be a class label or value.

### Activities
- Draw a simple decision tree diagram based on a hypothetical dataset, such as classifying animals based on characteristics like size and habitat.
- Use an online decision tree tool to visualize a decision tree for a given dataset and present your findings to the class.

### Discussion Questions
- What are some advantages and disadvantages of using decision trees compared to other machine learning algorithms?
- How can decision trees be optimized for better performance?
- Discuss a real-world scenario where decision trees might be an effective model choice.

---

## Section 4: How Decision Trees Work

### Learning Objectives
- Describe the splitting criteria used in decision trees.
- Understand the mechanism of decision tree learning.
- Explain the concept of impurity, including Gini impurity and entropy, in the context of decision trees.

### Assessment Questions

**Question 1:** What is the main goal of splitting in decision trees?

  A) To minimize the overall size of the tree
  B) To create child nodes that are as pure as possible
  C) To ensure all nodes contain equal amounts of data
  D) To maximize the depth of the tree

**Correct Answer:** B
**Explanation:** The goal of splitting in decision trees is to create child nodes that are as pure as possible, meaning they contain instances predominantly from a single class.

**Question 2:** Which splitting criterion indicates lower impurity?

  A) Higher Gini Impurity score
  B) Lower Entropy
  C) Higher Information Gain
  D) All of the above

**Correct Answer:** C
**Explanation:** Higher Information Gain indicates a better split, while lower Gini Impurity and lower Entropy also correspond to more homogeneous groups.

**Question 3:** In decision trees, what is the function of pruning?

  A) To increase the size of the tree
  B) To enhance the interpretability of the tree
  C) To remove pure nodes
  D) To add more features to the model

**Correct Answer:** B
**Explanation:** Pruning is a technique used in decision trees to enhance generalization by trimming branches that do not provide significant predictive power.

**Question 4:** What are the components of a decision tree?

  A) Root nodes, leaf nodes, and branches
  B) Input layer, hidden layers, and output layer
  C) Training set, validation set, and test set
  D) Features, parameters, and outputs

**Correct Answer:** A
**Explanation:** A decision tree consists of root nodes (initial node), branches (decisions based on criteria), and leaf nodes (final outcomes).

### Activities
- Create a simple decision tree for classifying fruits based on characteristics like color, size, and sweetness. Present your tree and explain the choices for each split.

### Discussion Questions
- What are the advantages and disadvantages of using decision trees compared to other classification algorithms?
- How does the choice of splitting criterion affect the performance of a decision tree?
- In what scenarios may pruning be particularly important in decision tree models?

---

## Section 5: Advantages of Decision Trees

### Learning Objectives
- Identify the key advantages of using decision trees for classification tasks.
- Explain how decision trees can be utilized in various real-world applications.
- Discuss the scenarios in which decision trees may outperform other models.

### Assessment Questions

**Question 1:** What is an advantage of using decision trees?

  A) They require feature scaling.
  B) They can handle both numerical and categorical data.
  C) They are always accurate.
  D) They do not require any preprocessing.

**Correct Answer:** B
**Explanation:** Decision trees can handle both numerical and categorical data without needing special preprocessing.

**Question 2:** How do decision trees handle outliers compared to other models?

  A) They ignore outliers completely.
  B) They are highly sensitive to outliers.
  C) They mitigate the influence of outliers by segregating them.
  D) They always perform worse with outliers.

**Correct Answer:** C
**Explanation:** Decision trees are less influenced by outliers as they primarily affect the leaves that contain them.

**Question 3:** Which of the following is true about decision trees?

  A) They assume a linear relationship between features.
  B) They require strict data distribution.
  C) They can capture non-linear relationships.
  D) They only work with categorical data.

**Correct Answer:** C
**Explanation:** Decision trees capture non-linear relationships effectively, which is a significant advantage over linear models.

**Question 4:** What facilitates the interpretability of decision trees?

  A) Their complex algorithms.
  B) Their visual representation as tree structures.
  C) Their computational complexity.
  D) The use of extensive mathematical formulas.

**Correct Answer:** B
**Explanation:** Decision trees are easy to visualize and understand as they represent decisions in a straightforward tree-like structure.

### Activities
- Create and present a decision tree for a real-world problem, such as predicting whether a student will pass or fail based on study habits and attendance.
- Organize a group discussion to identify various industries where decision trees could be applied and share practical examples.

### Discussion Questions
- In your opinion, what are the limitations of decision trees despite their advantages?
- Can you think of a scenario where a decision tree may not be the best model to use? Why or why not?
- How might ensemble methods like Random Forests improve upon the basic decision tree model?

---

## Section 6: Limitations of Decision Trees

### Learning Objectives
- Outline the limitations associated with decision trees.
- Understand the concept of overfitting and its impact on model performance.
- Recognize the issues of stability and bias in decision trees and their implications.

### Assessment Questions

**Question 1:** What is a limitation of decision trees?

  A) They are very interpretable.
  B) They have a tendency to overfit the data.
  C) They can work with small datasets.
  D) They require complex calculations.

**Correct Answer:** B
**Explanation:** Decision trees can overfit the training data, which may reduce their accuracy on unseen data.

**Question 2:** How can decision trees exhibit instability?

  A) By always producing the same output for a given input.
  B) By changing structure with small variations in the input data.
  C) By providing poor interpretability.
  D) By handling categorical variables efficiently.

**Correct Answer:** B
**Explanation:** Decision trees can change significantly with small changes in training data, leading to different structures.

**Question 3:** What is a major consequence of decision trees favoring majority classes?

  A) It leads to higher overall accuracy.
  B) It may result in misclassification of minority classes.
  C) It simplifies the decision-making process.
  D) It guarantees a better tree structure.

**Correct Answer:** B
**Explanation:** When one class dominates, the model may neglect the minority class, leading to misclassifications.

**Question 4:** Why can decision trees struggle with continuous variables?

  A) They cannot handle them at all.
  B) They require converting them into categorical variables.
  C) Continuous variables provide no information.
  D) Continuous variables can only be analyzed with neural networks.

**Correct Answer:** B
**Explanation:** Decision trees often discretize continuous variables, which can lead to information loss.

### Activities
- Analyze a sample decision tree model using a provided dataset, focusing on identifying overfitting signs and suggesting pruning techniques.
- Create visualizations for different decision tree structures generated from slight variations in a dataset to illustrate instability.

### Discussion Questions
- In what scenarios do you think decision trees would still be a preferable choice despite their limitations?
- What strategies can we implement to mitigate the limitations faced by decision trees?

---

## Section 7: What is K-Nearest Neighbors (KNN)?

### Learning Objectives
- Define K-Nearest Neighbors (KNN) and describe its classification methodology.
- Explain the role of distance metrics in KNN and their implications on classification.
- Understand the effects of choosing different 'k' values in KNN on model performance.

### Assessment Questions

**Question 1:** What does KNN use to determine the class of a data point?

  A) The training phase results
  B) The average of all data points
  C) The majority class among its 'k' closest neighbors
  D) The weight assigned to the data point

**Correct Answer:** C
**Explanation:** KNN determines the class of a data point based on the majority class among its 'k' closest neighbors.

**Question 2:** Which distance metric is commonly used in KNN?

  A) Manhattan Distance
  B) Euclidean Distance
  C) Hamming Distance
  D) Cosine Similarity

**Correct Answer:** B
**Explanation:** The Euclidean Distance is commonly used in KNN to measure how close data points are to each other.

**Question 3:** What happens if 'k' is chosen too small?

  A) The model becomes too generalized
  B) The algorithm runs faster
  C) The model becomes more sensitive to noise
  D) The accuracy increases

**Correct Answer:** C
**Explanation:** A small 'k' can lead to the model being overly sensitive to noise, as it only considers a few neighbors.

**Question 4:** What is a defining characteristic of KNN?

  A) It does not use any training data.
  B) It is a lazy learner.
  C) It requires extensive processing of data during training.
  D) It uses a complex mathematical model.

**Correct Answer:** B
**Explanation:** KNN is classified as a lazy learning algorithm; it does not build a model until it needs to make predictions.

### Activities
- Create a simple 2D scatter plot with labeled points. Choose a new point and manually calculate the distances to the nearest neighbors to classify it using KNN.
- Using the provided Python code snippet, modify the dataset by adding more points and observe how the prediction changes with different values of 'k'.

### Discussion Questions
- In what scenarios might KNN be a preferred algorithm compared to other classification methods?
- What are the implications of using KNN on high-dimensional datasets, and how might it differ from using it on 2D datasets?

---

## Section 8: How KNN Works

### Learning Objectives
- Understand the mechanics of the KNN algorithm and its process of classification.
- Explain the role of distance metrics in the KNN algorithm and their effects on classification outcomes.
- Demonstrate the voting process used in KNN for classifying new data points.

### Assessment Questions

**Question 1:** What is the purpose of the K parameter in KNN?

  A) To define the number of clusters in the dataset
  B) To determine the number of nearest neighbors to consider
  C) To adjust the learning rate of the algorithm
  D) To set the threshold for classification

**Correct Answer:** B
**Explanation:** The K parameter defines how many nearest neighbors to consider when making a classification decision.

**Question 2:** Which distance metric is calculated as the square root of the sum of squared differences?

  A) Manhattan distance
  B) Chebyshev distance
  C) Euclidean distance
  D) Minkowski distance

**Correct Answer:** C
**Explanation:** Euclidean distance is calculated as the square root of the sum of the squared differences between coordinates.

**Question 3:** What happens if K is set to an even number?

  A) It ensures a majority vote
  B) It may lead to a tie in voting
  C) It improves classification accuracy
  D) It reduces computational complexity

**Correct Answer:** B
**Explanation:** Setting K to an even number can lead to a tie in the voting system, which may complicate classification.

**Question 4:** Which of the following is a characteristic of KNN?

  A) It's a parametric algorithm
  B) It requires labeled training data
  C) It can only work with linear decision boundaries
  D) It is computationally efficient for very large datasets

**Correct Answer:** B
**Explanation:** KNN is a supervised algorithm, meaning it requires labeled training data to make predictions.

### Activities
- Implement the KNN algorithm on a sample dataset in Python, experimenting with different values of K and distance metrics. Compare the results.
- Create a visualization of a KNN decision boundary for a 2D dataset to better understand how changes in K affect the classifications.

### Discussion Questions
- How does the choice of distance metric affect the performance of the KNN algorithm in different datasets?
- What challenges might arise when using KNN in high-dimensional spaces, and how can they be addressed?
- In what scenarios might KNN be a less effective choice for classification compared to other algorithms?

---

## Section 9: Advantages of K-Nearest Neighbors

### Learning Objectives
- Identify and discuss the advantages of K-Nearest Neighbors.
- Recognize scenarios where KNN may be particularly effective.
- Differentiate KNN from other classification algorithms based on its features.

### Assessment Questions

**Question 1:** What is a significant advantage of KNN?

  A) KNN is always the most accurate algorithm.
  B) It is simple to implement and understand.
  C) It does not require a labeled dataset.
  D) KNN models are the fastest to compute.

**Correct Answer:** B
**Explanation:** KNN is known for its simplicity, making it easy for practitioners to implement.

**Question 2:** Which of the following statements about KNN is true?

  A) KNN requires a training phase to build a model.
  B) KNN makes assumptions about data distribution.
  C) KNN can effectively handle complex datasets.
  D) KNN cannot be used for multi-class classification.

**Correct Answer:** C
**Explanation:** KNN is versatile and can handle complex datasets, including multi-class classification problems.

**Question 3:** What does KNN rely on to make classifications?

  A) The average value of the features.
  B) The nearest neighbors in feature space.
  C) A pre-defined model generated during training.
  D) A linear combination of features.

**Correct Answer:** B
**Explanation:** KNN classifies data points based on the majority class of the nearest neighbors in feature space.

**Question 4:** What is one reason KNN is flexible in various applications?

  A) It only uses one distance metric.
  B) It does not need data to be normalized.
  C) It can adapt to different distance metrics.
  D) It generates a separate model for each class.

**Correct Answer:** C
**Explanation:** KNN's ability to adapt to different distance metrics allows it to be customized for specific problem domains.

### Activities
- Implement a KNN classification algorithm using a sample dataset. Visualize the data points and highlight the nearest neighbors to the test point.
- Choose two different distance metrics (e.g., Euclidean and Manhattan) and analyze how they affect the classification results on the same dataset.

### Discussion Questions
- What are some limitations or challenges that KNN might face when applied to large datasets?
- In what scenarios might you prefer a different classification algorithm over KNN?

---

## Section 10: Limitations of K-Nearest Neighbors

### Learning Objectives
- Describe the limitations and challenges associated with KNN.
- Understand how these limitations can influence the application and performance of KNN on different datasets.

### Assessment Questions

**Question 1:** What is a primary computational challenge of KNN?

  A) It processes data instantaneously.
  B) It requires calculating distances to all training samples.
  C) It uses a fixed number of clusters.
  D) It does not require much memory.

**Correct Answer:** B
**Explanation:** KNN requires calculating the distance to all training samples, leading to high time complexity and computational challenges.

**Question 2:** How does KNN handle noise in the dataset?

  A) It ignores noise completely.
  B) It is not influenced by noise.
  C) Its performance may degrade due to nearby outliers.
  D) It uses only the most relevant data points.

**Correct Answer:** C
**Explanation:** KNN is sensitive to noise, as outliers can disproportionately affect the categorization of instances based on nearest neighbors.

**Question 3:** What impact does the choice of 'k' have on KNN classification?

  A) A higher k always leads to better accuracy.
  B) A smaller k can result in underfitting.
  C) Choosing 'k' has no effect on classification results.
  D) A small k may lead to overfitting.

**Correct Answer:** D
**Explanation:** A small k value makes KNN sensitive to noise, potentially leading to overfitting due to excessive complexity.

**Question 4:** What is the effect of the curse of dimensionality on KNN?

  A) It makes distance metrics less effective.
  B) It improves classification accuracy.
  C) It reduces the memory requirements of KNN.
  D) It has no effect on KNN performance.

**Correct Answer:** A
**Explanation:** As dimensionality increases, data points tend to become equidistant, which undermines the effectiveness of distance-based algorithms like KNN.

### Activities
- Analyze a given dataset to identify the effects of noise on KNN performance by plotting the classification results with varying values of 'k'.
- Have students implement KNN on a small dataset and document memory usage as the dataset size increases.

### Discussion Questions
- In what scenarios do you think KNN would be an inappropriate choice for classification? Why?
- How can we preprocess a dataset to mitigate the limitations of KNN?

---

## Section 11: Model Evaluation Metrics for Classification

### Learning Objectives
- Understand various model evaluation metrics for classification.
- Analyze how different metrics provide insights into classifier performance.
- Differentiate between precision and recall and know when to prioritize which.

### Assessment Questions

**Question 1:** Which metric is used to measure the correctness of positive predictions?

  A) Recall
  B) Precision
  C) Specificity
  D) F1 Score

**Correct Answer:** B
**Explanation:** Precision measures the proportion of true positive results in all positive predictions made by the model.

**Question 2:** What is the formula for calculating Recall?

  A) TP / (TP + FN)
  B) TP / (TP + FP)
  C) (TP + TN) / Total Instances
  D) 2 * (Precision * Recall) / (Precision + Recall)

**Correct Answer:** A
**Explanation:** Recall is calculated as True Positives divided by the sum of True Positives and False Negatives.

**Question 3:** Which of the following best describes F1 Score?

  A) The proportion of correctly predicted instances
  B) The harmonic mean of precision and recall
  C) The number of true positives in a dataset
  D) The error rate of a model

**Correct Answer:** B
**Explanation:** The F1 Score is calculated as the harmonic mean of precision and recall, providing a balance between them.

**Question 4:** In which scenario would Recall be prioritized over Precision?

  A) Spam detection
  B) Disease detection in medical tests
  C) Document classification
  D) Customer churn prediction

**Correct Answer:** B
**Explanation:** In medical tests, minimizing the number of missed positive cases (high recall) is often more critical than achieving high precision.

### Activities
- Given a confusion matrix, calculate the accuracy, precision, recall, and F1 score.
- Create a small dataset of classification results and compute the relevant metrics to analyze model performance.

### Discussion Questions
- Why might a model with high accuracy still not be considered a good model?
- How can understanding precision and recall assist in designing better classification systems?
- What might be some real-world implications of focusing on recall over precision, or vice versa?

---

## Section 12: Real-World Applications of Decision Trees and KNN

### Learning Objectives
- Identify and explain various real-world applications of Decision Trees and KNN.
- Discuss the implications and effectiveness of these algorithms in different sectors.

### Assessment Questions

**Question 1:** What is a primary use of Decision Trees in the healthcare industry?

  A) Predicting weather patterns
  B) Diagnosing diseases based on symptoms
  C) Stock market predictions
  D) Cybersecurity threats

**Correct Answer:** B
**Explanation:** Decision Trees are utilized in healthcare to analyze patient symptoms and history for diagnosing diseases.

**Question 2:** In which scenario is K-Nearest Neighbors (KNN) particularly effective?

  A) Classifying high-dimensional data sets
  B) Recommending products based on user behavior
  C) Performing time series forecasting
  D) Statistical analysis of variance

**Correct Answer:** B
**Explanation:** KNN is widely used in recommendation systems, where it helps suggest products by analyzing similar user behaviors.

**Question 3:** Which of the following is a benefit of using Decision Trees?

  A) They require a large amount of computation time.
  B) They provide clear and interpretable visualizations.
  C) They are always more accurate than other models.
  D) They do not handle missing values well.

**Correct Answer:** B
**Explanation:** One of the key advantages of Decision Trees is their ability to display decision-making paths in an intelligible format.

**Question 4:** KNN relies on which of the following to make classifications?

  A) Logistic regression models
  B) Linear transformations of data
  C) The K closest neighbors in the feature space
  D) A pre-defined categorical matrix

**Correct Answer:** C
**Explanation:** KNN classifies data points by examining the K closest neighbors in the feature space, making it an instance-based algorithm.

### Activities
- Conduct a case study presentation on how Decision Trees or KNN have been applied in a specific industry, detailing the algorithm's impact and results.

### Discussion Questions
- How do you think Decision Trees and KNN can evolve with advancements in technology?
- What are the potential limitations of Decision Trees and KNN in real-world applications?
- Can you think of any ethical considerations when using these algorithms in sensitive fields like healthcare or finance?

---

## Section 13: Conclusion

### Learning Objectives
- Summarize the key points discussed about decision trees and KNN.
- Highlight the importance of these models in supervised learning.
- Distinguish between the working principles of decision trees and KNN.

### Assessment Questions

**Question 1:** What is a key characteristic of decision trees?

  A) They can only handle numerical data.
  B) They use binary splits based on feature values.
  C) They require extensive feature engineering.
  D) They are always more accurate than KNN.

**Correct Answer:** B
**Explanation:** Decision trees classify data by making binary splits based on feature values, allowing them to handle a variety of data types.

**Question 2:** In KNN, what does 'k' represent?

  A) The number of features in the dataset.
  B) The maximum depth of the decision tree.
  C) The number of closest neighbors used for classification.
  D) The accuracy of the model.

**Correct Answer:** C
**Explanation:** In KNN, 'k' refers to the number of nearest neighbors considered when classifying a new data point.

**Question 3:** Which of the following statements is true about KNN?

  A) It is a parametric learning method.
  B) It cannot be used for multi-class classification.
  C) It requires distance calculations for classification.
  D) It is less interpretable than decision trees.

**Correct Answer:** C
**Explanation:** KNN is an instance-based learning method that classifies new points by calculating distances to existing data points.

**Question 4:** Which area can benefit from using decision trees?

  A) Only image processing.
  B) Only numerical analysis.
  C) Multiple domains like finance, healthcare, and marketing.
  D) Only binary classification tasks.

**Correct Answer:** C
**Explanation:** Decision trees can be applied across various domains to solve both binary and multi-class classification issues.

### Activities
- Create a decision tree diagram using sample data, including at least three features.
- Implement a KNN classifier using a dataset of your choice and present the results.

### Discussion Questions
- What are the advantages and disadvantages of using decision trees compared to KNN?
- How might decision trees contribute to the development of ensemble methods like Random Forests?
- In what scenarios might KNN not perform well as a classification algorithm?

---

## Section 14: Questions and Discussion

### Learning Objectives
- Encourage discussion and questions to clarify doubts on different algorithms.
- Foster a collaborative learning environment by engaging students in sharing experiences and applications.

### Assessment Questions

**Question 1:** What is the primary goal of supervised learning?

  A) To model data without labels
  B) To train a model on a labeled dataset
  C) To generate new samples from data
  D) To perform unsupervised clustering

**Correct Answer:** B
**Explanation:** The primary goal of supervised learning is to train a model on a labeled dataset where each input is associated with a specific output label.

**Question 2:** Which of the following classification algorithms is based on the feature values used to split data?

  A) k-Nearest Neighbors
  B) Decision Trees
  C) Naive Bayes
  D) Support Vector Machines

**Correct Answer:** B
**Explanation:** Decision Trees use feature values to create branches that facilitate the prediction of classes based on input data.

**Question 3:** What does F1-score measure in classification algorithms?

  A) The accuracy of predictions
  B) The balance between precision and recall
  C) The speed of the algorithm
  D) The number of features in the dataset

**Correct Answer:** B
**Explanation:** F1-score is a measure that combines precision and recall, providing a single score that balances both metrics, useful in imbalanced classification scenarios.

**Question 4:** What challenge does class imbalance pose in classification tasks?

  A) It increases model complexity
  B) It skews the learning process towards the majority class
  C) It solely affects training duration
  D) It enhances model accuracy

**Correct Answer:** B
**Explanation:** Class imbalance can cause models to perform poorly as they may become biased towards predicting the majority class more often, neglecting the minority class.

### Activities
- In groups, brainstorm examples of real-world applications of classification algorithms. Prepare to present your example and discuss why classification is relevant in that context.
- Choose a classification algorithm (e.g., decision trees, k-NN) and provide a brief outline of an approach to implement it for a specific problem (e.g., medical diagnosis).

### Discussion Questions
- What are some potential ethical implications of using classification algorithms in practice?
- Can you think of a classification scenario in your daily life? How does it affect you?
- How do you determine which classification algorithm to use for a given dataset?

---

