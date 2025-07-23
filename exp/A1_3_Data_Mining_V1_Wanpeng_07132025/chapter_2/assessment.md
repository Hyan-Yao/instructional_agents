# Assessment: Slides Generation - Week 2: Data Preprocessing

## Section 1: Introduction to Data Preprocessing

### Learning Objectives
- Understand the role of data preprocessing in the data mining lifecycle.
- Identify why accurate preprocessing is necessary for effective analysis.
- Familiarize with the key steps involved in data preprocessing: cleaning, transformation, and reduction.

### Assessment Questions

**Question 1:** What is a primary outcome of effective data preprocessing?

  A) Reduced required processing time
  B) Improved accuracy in data analysis
  C) Increased size of the dataset
  D) Simplified data collection methods

**Correct Answer:** B
**Explanation:** Effective data preprocessing focuses on cleaning and structuring data, which directly enhances the accuracy of subsequent analyses.

**Question 2:** Which step in data preprocessing involves filling in missing values?

  A) Data Transformation
  B) Data Reduction
  C) Data Cleaning
  D) Data Integration

**Correct Answer:** C
**Explanation:** Data cleaning specifically targets the correction of inaccuracies and filling missing values within the dataset.

**Question 3:** What does data normalization achieve during the preprocessing stage?

  A) Increases data size
  B) Ensures uniformity in data distribution
  C) Eliminates outliers
  D) Merges datasets from different sources

**Correct Answer:** B
**Explanation:** Normalization is a data transformation technique that scales numerical data to ensure that it has a mean of zero and standard deviation of one, leading to a uniform distribution.

**Question 4:** Why is it necessary to perform data reduction?

  A) To eliminate unnecessary data and focus on relevant features
  B) To encrypt sensitive data
  C) To gather more data points
  D) To increase the dimensionality of data

**Correct Answer:** A
**Explanation:** Data reduction helps in reducing the volume of data while retaining essential characteristics which improves processing efficiency.

### Activities
- Identify a dataset you are familiar with and analyze its preprocessing requirements. Document any inconsistencies, missing values, or noise that could affect data analysis.
- Create a small dataset that includes missing entries or duplicates. Practice applying basic data cleaning techniques such as imputation and deduplication.

### Discussion Questions
- Discuss how a lack of proper data preprocessing could impact results in a specific domain, such as healthcare or finance.
- What preprocessing techniques do you think are most critical for your field of study or work, and why?

---

## Section 2: Motivation for Data Preprocessing

### Learning Objectives
- Recognize real-world challenges that require data preprocessing.
- Discuss examples from AI applications demonstrating the need for data preprocessing.
- Understand the impact of data quality issues on analytical accuracy and AI performance.

### Assessment Questions

**Question 1:** Which of the following is a challenge that necessitates data preprocessing?

  A) Excessive data volume
  B) Inconsistent data formats
  C) Technical expertise
  D) Insufficient data collection

**Correct Answer:** B
**Explanation:** Inconsistent data formats can lead to misinterpretation of data, necessitating preprocessing.

**Question 2:** What is one potential consequence of not addressing missing data?

  A) Increased performance of the model
  B) Greater accuracy in insights generated
  C) Misleading analysis results
  D) More efficient data processing

**Correct Answer:** C
**Explanation:** Missing data can create gaps in analysis, leading to inaccurate and misleading results if not properly managed.

**Question 3:** What technique can be used to handle irrelevant features in a dataset?

  A) Data normalization
  B) Feature selection
  C) Data augmentation
  D) Data visualization

**Correct Answer:** B
**Explanation:** Feature selection is the process of identifying and selecting the most relevant features while removing the irrelevant ones.

**Question 4:** Why is noise in data problematic for AI model performance?

  A) It increases the computational cost.
  B) It reduces the trustworthiness of the model.
  C) It provides more data for training.
  D) It makes data easier to analyze.

**Correct Answer:** B
**Explanation:** Noise can skew the training process and lead to models that perform poorly because they are learning from misleading information.

**Question 5:** In the context of ChatGPT, why is data preprocessing critical for the deployment of AI models?

  A) It decreases the volume of data.
  B) It ensures uniformity in training data.
  C) It reduces the number of interactions needed for training.
  D) It eliminates the need for model checks.

**Correct Answer:** B
**Explanation:** Uniformity in training data prevents inconsistencies during model training and leads to better performance during deployment.

### Activities
- Choose a dataset relevant to your field of interest. Identify and present on specific challenges related to data quality, such as missing data or inconsistencies, and discuss how you would preprocess this data to improve analysis outcomes.

### Discussion Questions
- Can you think of a scenario where data preprocessing could drastically change the insights gained from data? What would that scenario look like?
- How do you think advancements in data preprocessing technologies might influence the future of AI applications?

---

## Section 3: Data Cleaning

### Learning Objectives
- Define data cleaning and its significance in ensuring data quality.
- Identify different methods for handling missing values and removing duplicates.
- Understand the consequences of poor data quality on analysis and decision-making.

### Assessment Questions

**Question 1:** What is the primary goal of data cleaning?

  A) To increase data accessibility
  B) To remove irrelevant data
  C) To ensure high data quality
  D) To enrich data sets

**Correct Answer:** C
**Explanation:** Data cleaning primarily aims to ensure high data quality by correcting or removing erroneous data.

**Question 2:** Which method involves replacing missing data with an estimated value?

  A) Deletion
  B) Imputation
  C) Aggregation
  D) Normalization

**Correct Answer:** B
**Explanation:** Imputation is the most common method for addressing missing data by replacing it with estimated values.

**Question 3:** What consequence can arise from poor data quality?

  A) Improved decision-making
  B) Cost efficiency
  C) Misleading insights
  D) Enhanced data interoperability

**Correct Answer:** C
**Explanation:** Poor data quality often leads to misleading insights, which can impact strategic decisions negatively.

**Question 4:** Which of the following is a method to identify duplicates in a dataset?

  A) Data normalization
  B) Automated tools or specific SQL queries
  C) Mean imputation
  D) Data aggregation

**Correct Answer:** B
**Explanation:** Automated tools or SQL queries can efficiently identify duplicate rows based on specific key fields.

### Activities
- Practice cleaning a sample dataset that contains missing values and duplicates using your preferred programming language or data analysis software.
- Create a report outlining the impact of missing values on analysis outcomes and suggest proper handling techniques.

### Discussion Questions
- What real-world examples can you think of where poor data quality led to significant problems?
- How would you prioritize data cleaning tasks if working with a large, messy dataset?
- In your opinion, how can organizations create a culture of data quality and cleanliness?

---

## Section 4: Techniques for Data Cleaning

### Learning Objectives
- Understand the common techniques used for data cleaning.
- Apply imputation methods and outlier detection techniques.
- Recognize the importance of deduplication in maintaining data quality.
- Evaluate the effectiveness of different data cleaning techniques on dataset quality.

### Assessment Questions

**Question 1:** What is the purpose of imputation methods in data cleaning?

  A) To remove entire records
  B) To substitute missing values
  C) To merge datasets
  D) To visualize data

**Correct Answer:** B
**Explanation:** Imputation methods are specifically designed to substitute missing values with appropriate estimates.

**Question 2:** Which of the following is a common method for detecting outliers?

  A) Mean Imputation
  B) Z-Score
  C) K-Nearest Neighbors
  D) One-Hot Encoding

**Correct Answer:** B
**Explanation:** The Z-Score method calculates how far a data point is from the mean, helping to identify potential outliers.

**Question 3:** What can outliers potentially affect in a dataset?

  A) Data integrity
  B) Data visualization
  C) Data collection methods
  D) Data migration

**Correct Answer:** A
**Explanation:** Outliers can distort statistical analyses and may lead to incorrect interpretations of the dataset, affecting data integrity.

**Question 4:** What does the Interquartile Range (IQR) method help identify?

  A) Missing values
  B) Duplicate records
  C) Outliers
  D) Data distributions

**Correct Answer:** C
**Explanation:** The IQR method is specifically used to identify outliers in the dataset by determining the range of middle 50% of the data.

### Activities
- Using a dataset with missing values, apply mean and median imputation methods to see the differences in results.
- Identify and remove outliers in a given numeric dataset using the Z-Score method and Interquartile Range (IQR) method.
- Create a script to detect and remove duplicate entries from a dataset that contains user records.

### Discussion Questions
- How would you decide which imputation method to use for your dataset?
- What factors might lead you to consider a data point as an outlier?
- Discuss the potential consequences of failing to clean data before analysis.

---

## Section 5: Data Transformation

### Learning Objectives
- Understand the significance of data transformation in data preprocessing.
- Identify and apply normalization and standardization techniques effectively.

### Assessment Questions

**Question 1:** What is the primary goal of data normalization?

  A) To ensure all variables are on the same scale.
  B) To remove all outliers from the dataset.
  C) To convert categorical data into numerical data.
  D) To increase the size of the dataset.

**Correct Answer:** A
**Explanation:** Normalization aims to bring all variables into a consistent scale, which helps improve model performance.

**Question 2:** What is a key difference between normalization and standardization?

  A) Normalization converts data to a range from 0 to 1; standardization transforms data to have mean 0 and standard deviation 1.
  B) Both normalization and standardization convert data to a scale of [0, 1].
  C) Normalization is used only for categorical data; standardization is for numerical data.
  D) Normalization focuses on outliers while standardization ignores them.

**Correct Answer:** A
**Explanation:** Normalization scales data to a specific range while standardization adjusts the data to have a mean of 0 and standard deviation of 1.

**Question 3:** Why is data transformation important for certain machine learning algorithms?

  A) Algorithms require unstructured data for better accuracy.
  B) Some algorithms are sensitive to the scale of the data.
  C) Data transformation reduces the need for training data.
  D) All models perform the same irrespective of data scaling.

**Correct Answer:** B
**Explanation:** Certain algorithms, like KNN and logistic regression, rely on the scale of the data, making transformation essential for effective performance.

### Activities
- Given a small dataset, apply both normalization and standardization. Compare the results and discuss how each transformed dataset affects the outcome of a simple linear regression model.

### Discussion Questions
- Can you think of scenarios where normalization might be preferred over standardization and vice versa?
- How would you address the presence of outliers in your dataset when applying transformation techniques?

---

## Section 6: Handling Categorical Data

### Learning Objectives
- Understand strategies for handling categorical variables.
- Apply techniques such as one-hot encoding and label encoding.
- Identify when to use each encoding method based on the nature of categorical data.

### Assessment Questions

**Question 1:** Which technique is used for converting categorical variables into numerical format?

  A) One-hot encoding
  B) Label encoding
  C) Both A and B
  D) None of the above

**Correct Answer:** C
**Explanation:** Both one-hot encoding and label encoding are techniques used to convert categorical variables into numerical format.

**Question 2:** What is the primary advantage of using one-hot encoding?

  A) It avoids assuming any order among categories.
  B) It reduces the number of features in the dataset.
  C) It is always the best method for all categorical data.
  D) It combines multiple categories into one.

**Correct Answer:** A
**Explanation:** One-hot encoding creates distinct binary columns for each category, thus avoiding any assumptions of order among the categories.

**Question 3:** In which situation should you prefer label encoding over one-hot encoding?

  A) When the categorical data is nominal.
  B) When the categories are ordinal in nature.
  C) When there are too many categories.
  D) When you have continuous numerical data.

**Correct Answer:** B
**Explanation:** Label encoding is preferred when the categories have a meaningful order, as in the case of ordinal data.

**Question 4:** What is a potential drawback of one-hot encoding?

  A) It increases the dimensionality of the dataset.
  B) It does not allow for the use of categorical variables in machine learning.
  C) It converts continuous variables into categorical ones.
  D) It provides better interpretability.

**Correct Answer:** A
**Explanation:** One-hot encoding can lead to a high-dimensional feature space, which may result in the 'curse of dimensionality' as the number of categories increases.

### Activities
- Take a given dataset containing a categorical variable and apply both label encoding and one-hot encoding using Python. Compare the results and discuss the implications of the transformations.

### Discussion Questions
- What challenges might arise when handling categorical data with many categories?
- How does the choice of encoding technique affect the performance of machine learning models?

---

## Section 7: Data Reduction

### Learning Objectives
- Introduce the importance of data reduction in data analysis.
- Discuss techniques such as dimensionality reduction and feature selection and their applications.

### Assessment Questions

**Question 1:** What is the primary benefit of data reduction?

  A) It increases data dimensionality
  B) It enhances processing speed
  C) It reduces remove errors
  D) It increases data density

**Correct Answer:** B
**Explanation:** Data reduction improves efficiency by enhancing processing speed and reducing the complexity of data analysis.

**Question 2:** Which of the following is an example of dimensionality reduction?

  A) LASSO Regression
  B) Principal Component Analysis (PCA)
  C) Recursive Feature Elimination
  D) Chi-square Test

**Correct Answer:** B
**Explanation:** Principal Component Analysis (PCA) is a technique that transforms data to a lower-dimensional space by maximizing variance.

**Question 3:** What does feature selection involve?

  A) Adding more features to the dataset
  B) Selecting a subset of the most important features
  C) Reducing the size of data through averaging
  D) None of the above

**Correct Answer:** B
**Explanation:** Feature selection is the process of selecting a subset of the most relevant features for use in model construction.

**Question 4:** Which method is NOT a filter method of feature selection?

  A) Chi-square Test
  B) ANOVA
  C) Recursive Feature Elimination
  D) Correlation Coefficient

**Correct Answer:** C
**Explanation:** Recursive Feature Elimination is a wrapper method, not a filter method, as it uses model performance to select features.

### Activities
- Evaluate a dataset for opportunities for dimensionality reduction. Identify features that could be reduced or transformed using PCA or t-SNE.
- Conduct a hands-on activity where students apply filter, wrapper, and embedded methods to select features from a provided dataset.

### Discussion Questions
- How can data reduction techniques impact the performance of machine learning models?
- What challenges might arise when applying dimensionality reduction to certain datasets?
- Can you think of situations where maintaining higher dimensions might be necessary despite potential inefficiencies?

---

## Section 8: Dimensionality Reduction Techniques

### Learning Objectives
- Identify popular dimensionality reduction techniques, specifically PCA and t-SNE.
- Understand the computational methods behind PCA and t-SNE and their applications in data visualization.

### Assessment Questions

**Question 1:** What is the primary goal of Principal Component Analysis (PCA)?

  A) To find the linear combinations of features that maximize variance
  B) To classify data points into categories
  C) To cluster similar data points together
  D) To perform regression analysis

**Correct Answer:** A
**Explanation:** The primary goal of PCA is to identify the linear combinations of features that capture the most variance in the data.

**Question 2:** Which of the following techniques is specifically designed for visualizing high-dimensional data?

  A) Linear Regression
  B) t-Distributed Stochastic Neighbor Embedding (t-SNE)
  C) k-Means Clustering
  D) Decision Trees

**Correct Answer:** B
**Explanation:** t-Distributed Stochastic Neighbor Embedding (t-SNE) excels in visualizing high-dimensional data by preserving local structures.

**Question 3:** What is a key limitation of t-SNE?

  A) It cannot preserve local structures.
  B) It is computationally efficient for all sizes of datasets.
  C) It does not generally preserve the distances between clusters.
  D) It can only be applied to binary data.

**Correct Answer:** C
**Explanation:** t-SNE does well to preserve local structures but often does not maintain global distances between different clusters.

**Question 4:** What must be done to the data before applying PCA?

  A) Apply k-means clustering
  B) Standardize the dataset
  C) Normalize the data
  D) Select random features

**Correct Answer:** B
**Explanation:** The data must be standardized (mean = 0, variance = 1) to ensure that PCA gives equal weight to all features.

### Activities
- Experiment with PCA on the Iris dataset using Python libraries such as sklearn to visualize the reduced dimensions.
- Implement t-SNE on a dataset of your choice and create a scatter plot to visualize the clusters formed.

### Discussion Questions
- In what scenarios would you choose PCA over t-SNE, and why?
- How do dimensionality reduction techniques impact the performance of machine learning models?

---

## Section 9: Feature Selection Methods

### Learning Objectives
- Identify and describe various feature selection methods.
- Explain the significance of feature selection in enhancing model performance.

### Assessment Questions

**Question 1:** What is the primary goal of feature selection?

  A) To increase the number of features analyzed in a model
  B) To improve model performance by selecting relevant features
  C) To ensure that every feature is included in the analysis
  D) To optimize the size of the dataset

**Correct Answer:** B
**Explanation:** The primary goal of feature selection is to improve model performance by identifying and selecting the most relevant features.

**Question 2:** Which of the following is NOT a benefit of feature selection?

  A) Enhanced interpretability of the model
  B) Increased likelihood of overfitting
  C) Reduced training time
  D) Improved model accuracy

**Correct Answer:** B
**Explanation:** In fact, feature selection helps reduce overfitting by simplifying the model.

**Question 3:** Which method evaluates feature subsets based on model performance?

  A) Filter Methods
  B) Wrapper Methods
  C) Embedded Methods
  D) None of the above

**Correct Answer:** B
**Explanation:** Wrapper Methods evaluate feature subsets by training a model with them and assessing its performance.

**Question 4:** Which technique is an example of an embedded method for feature selection?

  A) Recursive Feature Elimination (RFE)
  B) Correlation Coefficient
  C) Chi-Squared Test
  D) LASSO Regression

**Correct Answer:** D
**Explanation:** LASSO (Least Absolute Shrinkage and Selection Operator) is an embedded method that incorporates feature selection into the model training process.

### Activities
- Use a dataset of your choice to perform feature selection using filter, wrapper, and embedded methods. Compare the model performance for each method to determine which one yields the best results.

### Discussion Questions
- How does the choice of feature selection method impact model performance for different types of datasets?
- Can feature selection methods lead to biased results, and if so, how?

---

## Section 10: Integrating Data Preprocessing in the Data Mining Pipeline

### Learning Objectives
- Discuss the integration of data preprocessing in the data mining lifecycle.
- Understand its impact on subsequent stages and overall model performance.
- Recognize key methods and techniques used in data preprocessing.

### Assessment Questions

**Question 1:** What is the primary goal of data preprocessing in the data mining pipeline?

  A) To collect data from various sources
  B) To transform raw data into a clean, usable format
  C) To visualize data insights
  D) To deploy models in real applications

**Correct Answer:** B
**Explanation:** Data preprocessing aims to transform raw data into a clean and usable format, preparing it for effective analysis.

**Question 2:** Which of the following is a common technique used in data cleaning?

  A) Dimensionality Reduction
  B) Normalization
  C) Missing Value Imputation
  D) Data Consolidation

**Correct Answer:** C
**Explanation:** Missing Value Imputation is a common technique used in data cleaning to fill in or address gaps in data.

**Question 3:** Why is normalization important in data transformation?

  A) It increases the dimensionality of the dataset
  B) It helps in making features comparable by scaling them to a similar range
  C) It merely organizes data visually
  D) It has no impact on data mining models

**Correct Answer:** B
**Explanation:** Normalization helps in making features comparable by scaling them to a similar range, thus improving model performance.

**Question 4:** How does data preprocessing reduce computational costs?

  A) By adding more features to the dataset
  B) By using advanced algorithms
  C) By simplifying the data, leading to faster processing times
  D) By increasing the dataset size

**Correct Answer:** C
**Explanation:** By simplifying the data (e.g., through data reduction or cleaning), preprocessing can lead to faster processing times and reduced computational burden.

### Activities
- Create a flowchart that outlines the steps of the data mining pipeline, including specific activities involved in data preprocessing.
- Examine a sample dataset and identify potential preprocessing steps required. Document your findings and suggest methods for handling missing values, normalizing data, and removing duplicates.

### Discussion Questions
- How can inadequate data preprocessing affect the final outcomes of a data mining project?
- Discuss the trade-offs between data reduction and the potential loss of important information during preprocessing.
- What new techniques or technologies could enhance data preprocessing in the future?

---

## Section 11: Examples of Data Preprocessing in Practice

### Learning Objectives
- Showcase practical examples of data preprocessing and its impact on different industries.
- Analyze case studies to illustrate the significance of preprocessing steps in enhancing data analysis.

### Assessment Questions

**Question 1:** What is the primary purpose of data preprocessing in data mining?

  A) To collect data
  B) To transform and clean data for analysis
  C) To analyze data without any changes
  D) To store data for future use

**Correct Answer:** B
**Explanation:** Data preprocessing transforms and cleans data to ensure it is suitable for analysis, improving the quality of insights drawn from the data.

**Question 2:** Which preprocessing step was used in the e-commerce recommendation system to quantify user engagement?

  A) Data Cleaning
  B) Feature Engineering
  C) Normalization
  D) Stop Word Removal

**Correct Answer:** B
**Explanation:** Feature engineering in the e-commerce case study involved creating a new feature called 'purchase frequency' to better quantify user engagement.

**Question 3:** What technique was employed to handle missing data in predicting patient outcomes in healthcare?

  A) Median imputation
  B) Mean imputation
  C) Deletion of records
  D) Ignoring missing values

**Correct Answer:** A
**Explanation:** The healthcare case study utilized median imputation to handle missing data for variables like patient age and blood pressure readings.

**Question 4:** In the context of sentiment analysis, why is text normalization important?

  A) It increases the volume of data available
  B) It reduces computation time
  C) It standardizes text format, improving analysis accuracy
  D) It removes all linguistic variations

**Correct Answer:** C
**Explanation:** Text normalization ensures that the text data is in a consistent format, which is crucial for accurate analysis of sentiment.

### Activities
- Analyze a data preprocessing case study from a different industry and present the steps taken and their outcomes.
- Create a preprocessing plan for a real-world dataset focusing on actions for cleaning and transforming the data.

### Discussion Questions
- What are some additional data preprocessing techniques that can be used in different types of datasets?
- How can inadequate data preprocessing affect the outcomes of machine learning models?

---

## Section 12: Summary and Key Takeaways

### Learning Objectives
- Recap important points discussed in the chapter.
- Understand the relevance of preprocessing techniques in data mining.
- Recognize the importance of data quality in resulting insights.

### Assessment Questions

**Question 1:** What is the main purpose of data cleaning in the preprocessing stage?

  A) To transform data into a visual format
  B) To remove irrelevant features
  C) To identify and correct errors in the dataset
  D) To collect more data

**Correct Answer:** C
**Explanation:** Data cleaning is essential to ensure that the dataset is accurate and reliable by identifying and correcting errors.

**Question 2:** Which of the following techniques is primarily used for data transformation?

  A) One-hot encoding
  B) K-means clustering
  C) Decision trees
  D) Linear regression

**Correct Answer:** A
**Explanation:** One-hot encoding is a data transformation technique that converts categorical data into a numerical format suitable for analysis.

**Question 3:** How does effective data integration enhance data mining?

  A) By complicating the dataset
  B) By creating a structured view from multiple sources
  C) By focusing only on data cleaning
  D) By ignoring irrelevant data

**Correct Answer:** B
**Explanation:** Effective data integration combines data from different sources, allowing for a cohesive view that is essential for thorough analysis.

**Question 4:** What is the main benefit of feature selection in data preprocessing?

  A) To increase the size of the dataset
  B) To remove useful features
  C) To identify relevant features and reduce dimensionality
  D) To collect more categorical data

**Correct Answer:** C
**Explanation:** Feature selection helps to simplify the dataset by identifying important features while minimizing irrelevant or redundant information.

### Activities
- Create a flowchart that outlines the data preprocessing steps discussed in the chapter, including key techniques.

### Discussion Questions
- What challenges might an analyst face when attempting to clean a large dataset?
- In what situations might data transformation disproportionately affect the analysis outcomes?

---

## Section 13: Discussion and Q&A

### Learning Objectives
- Encourage participants to explore and articulate data preprocessing concepts.
- Facilitate a deeper understanding of the techniques and their real-world applications in data preparation.

### Assessment Questions

**Question 1:** Which data preprocessing technique focuses on removing inaccuracies or inconsistencies in data?

  A) Data Transformation
  B) Data Cleaning
  C) Data Normalization
  D) Feature Selection

**Correct Answer:** B
**Explanation:** Data cleaning is the process aimed at correcting or removing errors and inconsistencies in the data.

**Question 2:** What is min-max normalization primarily used for?

  A) To improve computational speed
  B) To encode categorical variables
  C) To scale numerical values to a common range
  D) To eliminate outliers

**Correct Answer:** C
**Explanation:** Min-max normalization scales numerical values to a predefined range, usually between 0 and 1, ensuring that the values are comparable.

**Question 3:** What impact does effective feature selection have on machine learning models?

  A) Increases model complexity
  B) Reduces interpretability
  C) Minimizes overfitting
  D) Slows down processing time

**Correct Answer:** C
**Explanation:** Effective feature selection helps in minimizing overfitting by reducing the number of irrelevant features contributing to model confusion.

**Question 4:** Why is data transformation important in data preprocessing?

  A) To collect more data
  B) To improve the execution speed of models
  C) To convert data into a suitable format for analysis
  D) To visualize the data effectively

**Correct Answer:** C
**Explanation:** Data transformation modifies the structure and format of the data to make it suitable for analysis and modeling.

### Activities
- Create a small dataset containing missing values and implement a technique such as mean imputation to handle those missing values. Discuss the implications of your chosen method.

### Discussion Questions
- What common challenges do you face in data preprocessing within your projects?
- Can you provide examples from your experience regarding the impact of efficient data cleaning on outcomes?
- How do you decide which data preprocessing techniques to apply based on the characteristics of your dataset?

---

