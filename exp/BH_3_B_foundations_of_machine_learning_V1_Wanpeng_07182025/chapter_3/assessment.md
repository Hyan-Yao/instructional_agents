# Assessment: Slides Generation - Chapter 3: Data Preprocessing

## Section 1: Introduction to Data Preprocessing

### Learning Objectives
- Understand the importance of data preprocessing in the machine learning workflow.
- Identify common techniques used in data preprocessing.
- Apply data preprocessing techniques such as handling missing values and scaling features in practical scenarios.

### Assessment Questions

**Question 1:** What is the primary purpose of data preprocessing in machine learning?

  A) To improve model interpretability
  B) To transform raw data into a clean and usable format
  C) To increase the number of features in the dataset
  D) To visualize data more effectively

**Correct Answer:** B
**Explanation:** Data preprocessing is essential for transforming raw data, which often contains errors and inconsistencies, into a format that is clean and usable for machine learning algorithms.

**Question 2:** Which of the following techniques is used to handle missing values?

  A) Normalization
  B) One-hot encoding
  C) Imputation
  D) Feature scaling

**Correct Answer:** C
**Explanation:** Imputation is a common technique for handling missing values by filling them with substitutes such as mean, median, or mode of the dataset.

**Question 3:** What is one effect of scaling features during preprocessing?

  A) It increases model complexity
  B) It helps algorithms converge faster
  C) It eliminates the need for data cleaning
  D) It has no effect on model performance

**Correct Answer:** B
**Explanation:** Scaling features can help algorithms like gradient descent converge faster by ensuring that all features contribute equally to the distance calculations.

**Question 4:** In one-hot encoding, how many columns will be created for the feature with three unique values?

  A) 1
  B) 2
  C) 3
  D) 4

**Correct Answer:** C
**Explanation:** One-hot encoding creates a separate binary column for each unique value in the categorical variable. Thus, three unique values will lead to three binary columns.

### Activities
- Given a sample dataset with missing age values, demonstrate how to perform imputation using the mean age.
- Using a small dataset, apply normalization and standardization techniques and compare the results.

### Discussion Questions
- Why is data preprocessing considered a critical step in machine learning?
- How can improper preprocessing affect the outcome of a machine learning model?
- Discuss examples of datasets you have worked with that required significant preprocessing.

---

## Section 2: Data Quality and Importance of Cleaning

### Learning Objectives
- Understand the importance of data quality in the context of machine learning.
- Identify common issues related to data quality and their potential impacts.
- Implement effective strategies for cleaning data, including handling noise, missing values, and duplicates.

### Assessment Questions

**Question 1:** What is the primary consequence of having noise in the dataset?

  A) Improved model accuracy
  B) Obscured true patterns within the data
  C) Increased interpretability of the model
  D) Reduced computational load

**Correct Answer:** B
**Explanation:** Noise introduces random errors or variance that can obscure true patterns, negatively affecting model outcomes.

**Question 2:** Which of the following statements about missing values is true?

  A) They should always be removed from the dataset.
  B) They have no impact on model performance.
  C) They can lead to reduced statistical power if not handled properly.
  D) They indicate that the dataset is too large.

**Correct Answer:** C
**Explanation:** Missing values can greatly affect the accuracy of the model, leading to potential misinterpretations of results.

**Question 3:** What approach can be used to handle duplicates in a dataset?

  A) Ignoring duplicates
  B) Removing all records with duplicates
  C) Aggregating duplicate records
  D) All of the above

**Correct Answer:** C
**Explanation:** Aggregating duplicate records is a common approach, while simply removing all duplicates may lose valuable data.

### Activities
- Using a sample dataset, identify and report the noise, missing values, and duplicates present in the data, and suggest cleaning methods for each issue.
- Write a Python function to implement imputation for missing values in a given dataset based on the mean or median and apply it to a provided CSV file.

### Discussion Questions
- What are some real-world implications of poor data quality in machine learning projects?
- Can you think of a scenario where noise in the data leads to harmful outcomes?

---

## Section 3: Techniques for Data Cleaning

### Learning Objectives
- Understand the importance of handling missing values and be able to apply various imputation methods.
- Learn how to detect and remove duplicate entries in datasets to enhance data quality.
- Gain skills in identifying and correcting inconsistent data entries to improve data reliability.

### Assessment Questions

**Question 1:** What is the primary purpose of handling missing values in a dataset?

  A) To remove all outliers
  B) To ensure the dataset is complete and reliable
  C) To increase the amount of data collected
  D) To enhance the performance of machine learning algorithms

**Correct Answer:** B
**Explanation:** Handling missing values is crucial to maintain the reliability and completeness of the dataset, ensuring that analysis yields accurate results.

**Question 2:** Which of the following is a method for removing duplicates from a dataset?

  A) Using the 'fillna' function
  B) Using 'drop_duplicates' in Pandas
  C) Applying 'groupby' operation
  D) Standardizing format

**Correct Answer:** B
**Explanation:** 'drop_duplicates' is a built-in function in Pandas used to identify and remove duplicate entries from a DataFrame.

**Question 3:** What is one technique to correct inconsistent data formats?

  A) Adding more data to the dataset
  B) Standardization of values
  C) Merging datasets
  D) Creating new entries

**Correct Answer:** B
**Explanation:** Standardization of values ensures that all entries are in a consistent format, which is essential for accurate data interpretation.

**Question 4:** When should you consider removing entries with missing values?

  A) Always remove them without consideration
  B) When these entries are insignificant to overall data analysis
  C) If the majority of data in these entries is missing
  D) Both B and C

**Correct Answer:** D
**Explanation:** You should consider removing entries with missing values when these entries contribute little to the dataset or when the majority of information is absent, as keeping them could skew results.

### Activities
- Exercise: Using a provided dataset, identify all missing values, apply at least two different imputation techniques, and evaluate how each affects the dataset.
- Practical Task: Utilize Pandas to deduplicate a sample customer dataset by identifying duplicate entries and utilizing the 'drop_duplicates' method. Report before and after counts of records.

### Discussion Questions
- What potential biases could arise from improperly handling missing data?
- Can you think of a scenario where removing duplicates might lead to loss of critical information? Discuss.
- How would you approach the standardization of categorical data in your specific field or application?

---

## Section 4: Normalization Techniques

### Learning Objectives
- Understand the definitions and formulas for Min-Max Scaling and Z-score Standardization.
- Identify appropriate situations for using Min-Max Scaling versus Z-score Standardization.
- Recognize the impact of normalization techniques on machine learning models.
- Calculate normalized values using both techniques for provided datasets.

### Assessment Questions

**Question 1:** What is the range of values produced by Min-Max Scaling?

  A) [0, 1]
  B) [-1, 1]
  C) [μ - 3σ, μ + 3σ]
  D) Any real number

**Correct Answer:** A
**Explanation:** Min-Max Scaling transforms feature values to lie within a specified range, commonly [0, 1].

**Question 2:** Which normalization technique is preferred when data is normally distributed?

  A) Min-Max Scaling
  B) Z-score Standardization
  C) Both methods
  D) Neither method

**Correct Answer:** B
**Explanation:** Z-score Standardization is preferred for normally distributed data as it standardizes the dataset to have a mean of 0 and a standard deviation of 1.

**Question 3:** What is a potential drawback of using Min-Max Scaling?

  A) It does not scale data
  B) It is not reversible
  C) It is sensitive to outliers
  D) It only works for categorical data

**Correct Answer:** C
**Explanation:** Min-Max Scaling is sensitive to outliers, which can skew the scaling range and affect the normalized values.

**Question 4:** What is the formula for Z-score Standardization?

  A) Z = (X - μ) / σ
  B) Z = (X - Xmax) / (Xmax - Xmin)
  C) Z = (X - median)
  D) Z = (X - mode)

**Correct Answer:** A
**Explanation:** The formula for Z-score Standardization is Z = (X - μ) / σ, where μ is the mean and σ is the standard deviation.

### Activities
- Given a set of data, perform both Min-Max Scaling and Z-score Standardization and compare the results. Analyze how the choice of normalization method affects data distributions.
- Using a dataset with known outliers, apply Min-Max Scaling and discuss how the results differ from applying Z-score Standardization.

### Discussion Questions
- How might the choice of normalization technique affect model performance when dealing with real-world data?
- Can you propose a scenario where Z-score Standardization may fail to provide a good transformation? What could be alternatives?
- What role do outliers play in your choice of normalization technique, and how should you address them?

---

## Section 5: Data Transformation Techniques

### Learning Objectives
- Understand the definition and purpose of data transformation techniques.
- Describe the use of log and power transformations in data preprocessing.
- Apply log and power transformations to sample datasets and evaluate their effects on data distribution.

### Assessment Questions

**Question 1:** What is the purpose of log transformation?

  A) To increase the variance in the data
  B) To reduce right skewness in the data
  C) To simplify categorical variables
  D) To combine multiple datasets

**Correct Answer:** B
**Explanation:** Log transformation is used to reduce right skewness, making the data more symmetric and easier to analyze.

**Question 2:** Which of the following is a type of power transformation?

  A) Logarithmic
  B) Square root
  C) Linear
  D) Exponential

**Correct Answer:** B
**Explanation:** Square root is a type of power transformation used for stabilizing variance and reducing skewness.

**Question 3:** What is the Box-Cox transformation primarily used for?

  A) Reducing noise in data
  B) Creating categorical variables from continuous ones
  C) Stabilizing variance for normally distributed data
  D) None of the above

**Correct Answer:** C
**Explanation:** The Box-Cox transformation is specifically designed to stabilize variance and make data more normally distributed.

**Question 4:** Which transformation would you use on positively skewed data?

  A) Log transformation
  B) Standardization
  C) One-hot encoding
  D) Feature selection

**Correct Answer:** A
**Explanation:** Log transformation is effective for reducing positive skewness in data.

### Activities
- Given a dataset with positively skewed income values, apply a log transformation and a square root transformation. Plot both transformations on a histogram and discuss the differences in shape.
- Using the Box-Cox transformation, determine the appropriate lambda for a given dataset and apply the transformation. Discuss the outcomes in terms of data distribution.

### Discussion Questions
- How do different data distributions affect the choice of transformation technique?
- What are potential drawbacks of using data transformation, and how might they impact data analysis?

---

## Section 6: Feature Engineering

### Learning Objectives
- Understand concepts from Feature Engineering

### Activities
- Practice exercise for Feature Engineering

### Discussion Questions
- Discuss the implications of Feature Engineering

---

## Section 7: Handling Categorical Variables

### Learning Objectives
- Understand the definitions and differences between One-Hot Encoding and Label Encoding.
- Identify appropriate encoding techniques for nominal and ordinal categorical variables.
- Evaluate the implications of different encoding methods on model performance.

### Assessment Questions

**Question 1:** What is One-Hot Encoding?

  A) Assigning unique integers to categories.
  B) Creating binary columns for each category.
  C) Replacing categories with their frequencies.
  D) Using ordinal values to represent categories.

**Correct Answer:** B
**Explanation:** One-Hot Encoding creates new binary columns for each category, allowing categorical variables to be represented numerically.

**Question 2:** Which encoding technique is better suited for ordinal data?

  A) One-Hot Encoding
  B) Label Encoding
  C) Frequency Encoding
  D) Binary Encoding

**Correct Answer:** B
**Explanation:** Label Encoding is suitable for ordinal data as it retains the meaningful order of categories.

**Question 3:** What is a potential drawback of using One-Hot Encoding?

  A) It can introduce bias.
  B) It may lead to high dimensionality.
  C) It is not compatible with machine learning models.
  D) It can only be used with nominal data.

**Correct Answer:** B
**Explanation:** One-Hot Encoding can lead to high dimensionality, especially when there are many categories, affecting model performance.

**Question 4:** Why is it important to select an appropriate encoding technique?

  A) To optimize the visual presentation of data.
  B) To ensure numerical inputs are suitable for ML algorithms.
  C) To convert numerical data into categorical data.
  D) To minimize processing time.

**Correct Answer:** B
**Explanation:** Selecting the appropriate encoding technique is crucial to ensure that categorical variables are represented correctly as numerical inputs suitable for machine learning algorithms.

### Activities
- Using a small dataset, practice implementing both One-Hot Encoding and Label Encoding in Python. Compare the results and discuss which technique is more suitable for different scenarios.
- Given a dataset with multiple categorical variables, determine which encoding technique you would use for each variable and justify your choices.

### Discussion Questions
- What challenges might arise from high-dimensional data created by One-Hot Encoding?
- In what scenarios could Label Encoding lead to misleading results? Discuss with examples.

---

## Section 8: Outlier Detection and Treatment

### Learning Objectives
- Understand the definition and significance of outliers in data analysis.
- Learn to calculate outlier detection metrics like Z-scores and IQR.
- Identify outliers in datasets using Z-scores and IQR.
- Apply appropriate methods for treating outliers in data analysis.

### Assessment Questions

**Question 1:** What is a Z-score?

  A) A measure of the average of a dataset
  B) A measure of how many standard deviations a value is from the mean
  C) A calculation to find the median
  D) A method to remove outliers

**Correct Answer:** B
**Explanation:** A Z-score indicates how many standard deviations a particular data point is from the mean, helping to identify outliers.

**Question 2:** Which of the following is a criterion for identifying outliers using the IQR method?

  A) Below the mean
  B) Above the median
  C) Below Q1 - 1.5*IQR or above Q3 + 1.5*IQR
  D) Above the mean + 2*standard deviation

**Correct Answer:** C
**Explanation:** The IQR method identifies outliers as values that fall below Q1 - 1.5*IQR or above Q3 + 1.5*IQR.

**Question 3:** What is the main reason for detecting outliers in datasets?

  A) To improve data processing speed
  B) To ensure accurate statistical analysis
  C) To find the mean easily
  D) To increase dataset size

**Correct Answer:** B
**Explanation:** Detecting outliers is important because they can distort statistical analyses and lead to misleading conclusions.

**Question 4:** Which treatment method is NOT commonly used for outliers?

  A) Removal
  B) Transformation
  C) Imputation
  D) Normalization

**Correct Answer:** D
**Explanation:** Normalization is a data preprocessing method but is not a treatment method specifically for outliers.

### Activities
- Given a dataset, calculate the Z-scores and determine which values are outliers. Document your findings.
- Using a provided dataset, compute the IQR and identify any outliers based on the IQR method. Share your results in a brief report.

### Discussion Questions
- What impact do you think outliers have on your specific field of study or profession?
- Can you think of a situation where outliers might provide valuable insights rather than being dismissed? Discuss.

---

## Section 9: Data Preprocessing Pipeline in Python

### Learning Objectives
- Understand the importance of data preprocessing in machine learning.
- Identify and apply various preprocessing methods such as scaling, encoding, and imputation using Scikit-learn.
- Construct a data preprocessing pipeline that automates the preprocessing steps for a given dataset.

### Assessment Questions

**Question 1:** What is the primary purpose of a data preprocessing pipeline?

  A) To reduce dataset size
  B) To automate data transformations
  C) To visualize data
  D) To change the format of datasets

**Correct Answer:** B
**Explanation:** A data preprocessing pipeline automates repetitive data transformations, promoting efficiency and consistency in preparing data for machine learning algorithms.

**Question 2:** Which method is used for adjusting the range of numeric features?

  A) One-Hot Encoding
  B) Label Encoding
  C) Normalization
  D) K-Nearest Neighbors

**Correct Answer:** C
**Explanation:** Normalization is a technique used to adjust the range of numeric features, typically rescaling them to a [0, 1] range.

**Question 3:** What does One-Hot Encoding achieve?

  A) Fills missing values based on mean or median
  B) Converts categorical variables into a format suitable for algorithms
  C) Centers data around the mean
  D) Scales data to unit variance

**Correct Answer:** B
**Explanation:** One-Hot Encoding converts categorical variables into binary columns, making them suitable for machine learning algorithms that require numerical input.

**Question 4:** What is the strategy used in SimpleImputer to fill missing numerical values in the pipeline?

  A) With a constant value
  B) By using K-Nearest Neighbors
  C) By using mean or median
  D) Excluding data rows with missing values

**Correct Answer:** C
**Explanation:** SimpleImputer can fill missing numerical values using mean or median, helping to preserve as much valuable data as possible.

### Activities
- Create your own data preprocessing pipeline using a dataset of your choice. Use Scikit-learn to implement scaling, encoding, and imputation, and then apply the pipeline to transform your data.
- Analyze a given dataset with missing values and outliers. Determine which preprocessing techniques (imputation and scaling) would be most appropriate and justify your choices.

### Discussion Questions
- What challenges might arise when preprocessing data for machine learning? How can they be addressed?
- In what scenarios would you prefer normalization over standardization, or vice versa?
- How does the choice of imputation method impact the overall model performance?

---

## Section 10: Case Study: Impact of Data Preprocessing

### Learning Objectives
- Understand the importance of data preprocessing in machine learning.
- Identify common data preprocessing techniques and their impacts on model performance.
- Apply data preprocessing methods to a dataset and evaluate the outcomes.

### Assessment Questions

**Question 1:** What was the initial accuracy of the logistic regression model before preprocessing?

  A) 70%
  B) 80%
  C) 65%
  D) 75%

**Correct Answer:** C
**Explanation:** The initial accuracy of the logistic regression model was 65%, as stated in the case study.

**Question 2:** Which technique was used for imputing missing values in numerical features?

  A) Mode Imputation
  B) Mean Imputation
  C) Median Imputation
  D) Random Imputation

**Correct Answer:** B
**Explanation:** Mean imputation was used for numerical features to fill in missing values.

**Question 3:** What preprocessing technique was applied to categorical variables?

  A) Label Encoding
  B) One-Hot Encoding
  C) Min-Max Scaling
  D) Binarization

**Correct Answer:** B
**Explanation:** One-Hot Encoding was used to transform categorical variables into a format suitable for model training.

**Question 4:** After preprocessing, what was the accuracy of the logistic regression model?

  A) 75%
  B) 85%
  C) 80%
  D) 90%

**Correct Answer:** C
**Explanation:** After preprocessing, the model's accuracy improved to 80%.

**Question 5:** Which of the following was NOT identified as a key issue during initial model performance?

  A) Missing values
  B) Irrelevant features
  C) Unscaled numerical data
  D) Excessive dimensionality

**Correct Answer:** D
**Explanation:** Excessive dimensionality was not mentioned; the issues were related to missing values, irrelevant features, and unscaled numerical data.

### Activities
- Design a simple preprocessing pipeline using Scikit-learn for a hypothetical dataset containing both numerical and categorical features. Include steps for missing value imputation and feature scaling.
- Using a dataset of your choice, apply one of the preprocessing techniques discussed in the case study. Document the changes in model performance before and after preprocessing.

### Discussion Questions
- Why is data quality crucial for machine learning model performance?
- Discuss the potential challenges and limitations of data preprocessing.
- How might different preprocessing techniques affect the interpretability of the model?

---

## Section 11: Practical Exercise

### Learning Objectives
- Understand the importance of data preprocessing in enhancing model performance.
- Apply common data cleaning and transformation techniques using Python.
- Implement feature engineering to create new variables from existing data.

### Assessment Questions

**Question 1:** What is the primary goal of data cleaning?

  A) To create new data features
  B) To improve data quality by correcting errors
  C) To visualize data
  D) To collect more data

**Correct Answer:** B
**Explanation:** The primary goal of data cleaning is to improve data quality by identifying and correcting errors or inconsistencies.

**Question 2:** Which technique is commonly used for handling missing values?

  A) Normalization
  B) Imputation
  C) Feature Engineering
  D) Scaling

**Correct Answer:** B
**Explanation:** Imputation is a common technique for handling missing values, which can involve replacing missing entries with mean, median, or mode.

**Question 3:** What method can be used to detect outliers using IQR?

  A) Z-score method
  B) Mean calculation
  C) Q1 and Q3 calculation
  D) Random sampling

**Correct Answer:** C
**Explanation:** Outliers can be detected using the Interquartile Range (IQR) method, which involves calculating the first and third quartiles (Q1 and Q3).

**Question 4:** What does normalization do to data?

  A) Converts it to categorical data
  B) Scales it within a specific range, such as 0 to 1
  C) Removes all missing values
  D) Replaces outliers with mean values

**Correct Answer:** B
**Explanation:** Normalization scales the data within a specific range, which helps in preparing the data for machine learning algorithms.

### Activities
- Perform data cleaning on the Titanic dataset by identifying and imputing missing values.
- Detect and handle outliers in the 'Fare' feature using the IQR method.
- Normalize the 'Fare' feature using MinMaxScaler from sklearn.
- Create a new feature 'FamilySize' by summing 'SibSp' and 'Parch'.

### Discussion Questions
- Why is data preprocessing important before applying machine learning models?
- Can you think of scenarios where data normalization might not be necessary?
- What challenges might arise when dealing with missing values in datasets?

---

## Section 12: Ethical Considerations in Data Preprocessing

### Learning Objectives
- Understand the concept of data bias and its implications in various fields.
- Identify potential sources of bias in datasets and methodologies for addressing them.
- Apply ethical frameworks to evaluate data preprocessing techniques.

### Assessment Questions

**Question 1:** What is data bias?

  A) A representation of diverse population characteristics
  B) A systematic error in the data that leads to skewed results
  C) An accurate reflection of socio-economic factors
  D) An unbiased collection of random data samples

**Correct Answer:** B
**Explanation:** Data bias refers to systematic errors in how data is collected or represented, leading to skewed results and potentially inequitable outcomes.

**Question 2:** Which of the following is an example of data bias?

  A) Using a dataset that is inclusive of all demographics
  B) A hiring algorithm trained primarily on data from a specific demographic group
  C) Collecting random samples from a large population to form a dataset
  D) Implementing daily updates to improve predictive accuracy

**Correct Answer:** B
**Explanation:** A hiring algorithm based on historical data from certain demographics may perpetuate existing biases, leading to discrimination against other qualified candidates.

**Question 3:** Why is it important to ensure transparency in data preprocessing?

  A) To make it easier to change the dataset later
  B) To avoid the need for documentation
  C) To foster trust and accountability in the model-building process
  D) To reduce the cost of data collection

**Correct Answer:** C
**Explanation:** Transparency in data preprocessing fosters trust and accountability, allowing stakeholders to understand how and why specific decisions were made regarding data handling.

**Question 4:** Which approach helps reduce bias during data preprocessing?

  A) Ignoring demographic factors in data collection
  B) Using datasets that predominantly feature one group
  C) Engaging diverse stakeholders in the data collection process
  D) Keeping preprocessing techniques secret

**Correct Answer:** C
**Explanation:** Engaging diverse stakeholders in the data collection process ensures that a wide range of perspectives are included, helping to create a more representative dataset and reducing bias.

### Activities
- Conduct a case study analysis where students identify potential biases in a given dataset used for a predictive model. Discuss the implications of these biases on diverse groups.
- Implement a simple Python script to assess bias in a provided dataset. Students will use statistical methods to evaluate the representativeness of the data.

### Discussion Questions
- How can we balance the need for representative datasets with the practical limitations of data collection?
- What are some real-world consequences of biased algorithms in areas such as healthcare and hiring?
- In what ways can stakeholder engagement during the data collection phase help combat bias?

---

## Section 13: Conclusion

### Learning Objectives
- Understand the importance of data preprocessing in machine learning workflows.
- Identify and apply essential techniques in data preprocessing, including data cleaning, transformation, and feature engineering.
- Recognize best practices in maintaining data integrity during preprocessing.

### Assessment Questions

**Question 1:** What is the primary benefit of data preprocessing in machine learning?

  A) It complicates the model
  B) It transforms raw data into a usable format
  C) It reduces data size
  D) It eliminates the need for algorithms

**Correct Answer:** B
**Explanation:** Data preprocessing is essential because it transforms raw data into a format that can be effectively used by machine learning algorithms.

**Question 2:** Which of the following techniques is NOT typically part of data preprocessing?

  A) Data Cleaning
  B) Feature Engineering
  C) Model Training
  D) Data Transformation

**Correct Answer:** C
**Explanation:** Model Training is a separate phase in the machine learning workflow and not part of data preprocessing, which includes cleaning, transforming, and engineering features.

**Question 3:** What is one common method for handling missing values in data?

  A) Ignoring them
  B) Imputing with the mean
  C) Copying data from other records
  D) Deleting the dataset

**Correct Answer:** B
**Explanation:** Imputing missing values with the mean of the existing values is a common and basic technique for handling missing data.

**Question 4:** What does One-Hot Encoding achieve?

  A) It reduces model complexity.
  B) It transforms categorical variables into numerical format.
  C) It increases data size.
  D) It eliminates outliers.

**Correct Answer:** B
**Explanation:** One-Hot Encoding is a technique used to convert categorical variables into a format that can be provided to machine learning algorithms.

### Activities
- Perform data preprocessing on a sample dataset by handling missing values, applying normalization, and encoding categorical variables. Document your methods and results.
- Create a visual report demonstrating the distribution of a dataset before and after applying preprocessing techniques.

### Discussion Questions
- Why is it important to maintain the integrity of the data during preprocessing? Can you provide an example of how a transformation can alter the data's meaning?
- Discuss the implications of biased data in machine learning. How can data preprocessing help mitigate these biases?

---

