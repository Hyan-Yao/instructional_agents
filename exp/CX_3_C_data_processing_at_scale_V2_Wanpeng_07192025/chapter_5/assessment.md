# Assessment: Slides Generation - Week 5: Data Wrangling Techniques

## Section 1: Introduction to Data Wrangling Techniques

### Learning Objectives
- Understand the concept and significance of data wrangling.
- Identify common data issues and applicable techniques for data cleaning.
- Discuss the importance of clean data in the context of decision-making and analysis.

### Assessment Questions

**Question 1:** What is data wrangling?

  A) The process of storing data
  B) Transforming and preparing raw data for analysis
  C) Analyzing data in real-time
  D) None of the above

**Correct Answer:** B
**Explanation:** Data wrangling, also known as data munging, involves transforming raw data into a format that can be effectively analyzed.

**Question 2:** Why is data cleaning important?

  A) It reduces data entry costs
  B) It makes the data look more professional
  C) It improves the quality and reliability of data analysis
  D) It has no real significance

**Correct Answer:** C
**Explanation:** Cleaning data ensures that the data used for analysis is accurate and reliable, leading to correct conclusions and insights.

**Question 3:** What could be a potential consequence of using poor-quality data?

  A) Enhanced decision-making
  B) Time saved in the analysis process
  C) Incorrect conclusions
  D) Improved customer targeting

**Correct Answer:** C
**Explanation:** Using poor-quality data can result in incorrect conclusions, which can adversely affect decisions and strategies.

**Question 4:** Which of the following is NOT a common data issue addressed in data wrangling?

  A) Missing values
  B) Inconsistent formats
  C) Increased computational speed
  D) Duplicates

**Correct Answer:** C
**Explanation:** Increased computational speed is not a data issue; rather, it can be a benefit of properly wrangled data.

**Question 5:** What is normalization in data wrangling?

  A) Removing all outliers from data
  B) Standardizing data on a common scale
  C) Merging two datasets
  D) Changing data types for analysis

**Correct Answer:** B
**Explanation:** Normalization is the process of standardizing values on a common scale without distorting the differences in ranges to allow for effective comparisons.

### Activities
- 1. Given a dataset with missing values and inconsistent date formats, write a short Python script using Pandas to clean the data.
- 2. Provide a dataset with duplicate entries and ask students to identify and remove the duplicates while maintaining the integrity of the data.

### Discussion Questions
- What impact do you think data quality has on business outcomes?
- Can you share any experiences where data wrangling significantly influenced your findings or decisions?

---

## Section 2: What is Data Wrangling?

### Learning Objectives
- Understand the definition and importance of data wrangling in data analysis.
- Identify and describe the key components involved in the data wrangling process.
- Apply techniques for data cleaning, transformation, and enrichment to prepare datasets for analysis.

### Assessment Questions

**Question 1:** What is the main goal of data wrangling?

  A) Collect raw data
  B) Clean and organize data for analysis
  C) Visualize data trends
  D) Store data in a database

**Correct Answer:** B
**Explanation:** The main goal of data wrangling is to clean and organize raw data to make it suitable for analysis.

**Question 2:** Which of the following is NOT a component of data wrangling?

  A) Data collection
  B) Data visualization
  C) Data cleaning
  D) Data transformation

**Correct Answer:** B
**Explanation:** Data visualization is a step that typically follows data wrangling, while the others are key components of the data wrangling process.

**Question 3:** Why is data validation important in the data wrangling process?

  A) To improve data collection methods
  B) To enhance data visualization options
  C) To ensure the accuracy and quality of the transformed data
  D) To reduce the dataset size

**Correct Answer:** C
**Explanation:** Data validation is critical to ensure that the data is accurate and high quality after transformation, which is essential for making informed decisions.

**Question 4:** What does data enrichment involve?

  A) Handling missing data
  B) Combining data with external sources
  C) Normalizing numerical data
  D) Replacing duplicate entries

**Correct Answer:** B
**Explanation:** Data enrichment involves combining the original dataset with external data sources to provide additional context or features.

**Question 5:** What can be a consequence of poorly wrangled data?

  A) Improved data insights
  B) Enhanced business outcomes
  C) Misleading conclusions
  D) Automated data analysis

**Correct Answer:** C
**Explanation:** Poorly wrangled data can lead to misleading conclusions and ineffective decision-making, which can negatively impact business strategies.

### Activities
- Using a provided raw sales dataset, students will identify and correct errors, fill in missing values, and transform the data into a format suitable for analysis.
- In groups, students will compare their wrangled datasets and discuss the different approaches taken for cleaning data and the challenges faced.

### Discussion Questions
- What challenges have you faced in data wrangling, and how did you overcome them?
- In your opinion, which component of data wrangling is the most critical, and why?
- How does the quality of the data affect the decision-making process in businesses?

---

## Section 3: Data Quality Issues

### Learning Objectives
- Identify and define common data quality issues including missing values, duplicates, and outliers.
- Analyze the impact of these issues on data integrity and validity.
- Employ basic methods for detecting and addressing data quality issues in practical scenarios.

### Assessment Questions

**Question 1:** What is a common issue related to missing values?

  A) Increased accuracy in data analysis
  B) Biased results or loss of information
  C) Improved data integrity
  D) Enhanced data visualization

**Correct Answer:** B
**Explanation:** Missing values can lead to biased results or loss of information in analysis, which directly impacts the overall data quality.

**Question 2:** Which of the following is an example of a duplicate?

  A) A single entry with multiple missing values
  B) A survey result submitted multiple times by the same participant
  C) An outlier score significantly higher than normal
  D) A record without any associated values

**Correct Answer:** B
**Explanation:** A survey result submitted multiple times by the same participant is an example of duplicates, which can inflate metrics like total responses.

**Question 3:** What is a valid method to handle outliers in a dataset?

  A) Ignore them completely
  B) Use statistical imputation
  C) Analyze them with domain knowledge
  D) Fill them with mean values

**Correct Answer:** C
**Explanation:** Analyzing outliers with domain knowledge helps to determine whether they deserve correction, exclusion, or further investigation.

**Question 4:** What can be used to visualize outliers in a dataset?

  A) Heatmap
  B) Pie chart
  C) Box plot
  D) Line graph

**Correct Answer:** C
**Explanation:** Box plots are effective for visualizing the distribution of data and highlighting outliers.

### Activities
- Given a dataset (provided in CSV format), identify and report the number of missing values, duplicates, and outliers using Python libraries such as Pandas and Matplotlib/Seaborn.
- Create visual representations of outliers in a given dataset, utilizing box plots. Discuss the insights gleaned from the results.

### Discussion Questions
- Why do you think data quality issues are often overlooked in data analysis?
- How can poor data quality affect business decisions in real-world scenarios?
- What are some additional data quality issues you may think of that are not covered in this slide?

---

## Section 4: Techniques for Data Cleaning

### Learning Objectives
- To understand the importance of data cleaning in the data wrangling process.
- To identify and differentiate between techniques of data removal, imputation, and transformation.
- To apply data cleaning techniques on sample datasets and assess their impact on data analysis.

### Assessment Questions

**Question 1:** What is the primary purpose of data removal in data cleaning?

  A) To simplify analysis by eliminating redundant data
  B) To increase the dataset size
  C) To alter the data for better interpretation
  D) To predict missing values

**Correct Answer:** A
**Explanation:** Data removal is intended to simplify the dataset by eliminating redundant or unnecessary data, thus aiding in clearer analysis.

**Question 2:** Which method is NOT commonly used for imputation?

  A) Mean Imputation
  B) Median Imputation
  C) Normalization
  D) Predictive Imputation

**Correct Answer:** C
**Explanation:** Normalization is a transformation technique, not an imputation method. It adjusts the scale of data rather than filling in missing values.

**Question 3:** What is the main aim of data transformation?

  A) To delete irrelevant entries
  B) To forecast future trends
  C) To prepare data into a suitable format for analysis
  D) To visualize data effectively

**Correct Answer:** C
**Explanation:** Data transformation aims to modify the data into formats that are better suited for analysis, such as normalizing or encoding.

**Question 4:** Why is it important to choose the correct imputation method?

  A) It affects the speed of data processing
  B) It can significantly alter the analysis results
  C) It helps in increasing the dataset size
  D) It is a legal requirement in data handling

**Correct Answer:** B
**Explanation:** The method chosen for imputation can greatly influence the outcomes of data analysis, potentially leading to biased results if not appropriate.

### Activities
- Using software like Pandas, load a dataset with missing values and practice filling in missing values using different imputation techniques (mean, median, predictive).
- Select a dataset and identify any duplicates. Perform data removal and document the changes. Analyze how this impacts the dataset's usability.

### Discussion Questions
- In what scenarios might data removal result in the loss of valuable information?
- How would you decide which imputation method to use for your data?
- What challenges have you faced in data cleaning during previous projects or assignments?

---

## Section 5: Data Transformation Techniques

### Learning Objectives
- Understand the importance of normalization in data preparation for analysis.
- Be able to apply Min-Max and Z-score normalization techniques.
- Gain insights into the aggregation process and its significance in summarizing data effectively.

### Assessment Questions

**Question 1:** What is the purpose of normalization in data transformation?

  A) To combine multiple data points into a single value
  B) To adjust values to a common scale
  C) To increase the volume of the dataset
  D) To filter out irrelevant data

**Correct Answer:** B
**Explanation:** Normalization adjusts values to a common scale without distorting differences in the data ranges, ensuring all variables contribute equally to distance computations.

**Question 2:** Which of the following methods is NOT commonly used for normalization?

  A) Min-Max Normalization
  B) Z-score Normalization
  C) Total Count Normalization
  D) None of the above

**Correct Answer:** C
**Explanation:** Total Count Normalization is not a recognized method for normalization; the common methods are Min-Max and Z-score normalization.

**Question 3:** What does aggregation typically involve in the context of data transformation?

  A) Changing data types to improve analysis
  B) Summarizing multiple data points into single values
  C) Filtering out unnecessary data entries
  D) Visualizing data trends with graphs

**Correct Answer:** B
**Explanation:** Aggregation involves summarizing or combining multiple data points into a single value, providing a condensed view of the data while retaining meaningful information.

**Question 4:** When would you use Z-score Normalization?

  A) When data is not normally distributed
  B) When you want values between 0 and 1
  C) When you need to standardize the mean and standard deviation
  D) When performing aggregation

**Correct Answer:** C
**Explanation:** Z-score Normalization is used when you need the values to have a mean of 0 and a standard deviation of 1, which is particularly helpful when the data is normally distributed.

### Activities
- Using a sample dataset provided, apply both Min-Max and Z-score normalization to the data and compare the results.
- Using Python and Pandas, create a new dataset and demonstrate the aggregation of data by calculating the total revenue generated for each product.

### Discussion Questions
- Discuss the impact of using different normalization techniques on the outcome of a machine learning model.
- In what scenarios would aggregation not be beneficial? Provide examples.

---

## Section 6: Data Wrangling Tools

### Learning Objectives
- Identify and describe the features and use cases of various data wrangling tools such as Pandas, Dplyr, and Apache Spark.
- Apply the appropriate data wrangling techniques using Pandas and Dplyr on sample datasets.
- Understand the scenarios where Apache Spark is advantageous compared to other data processing tools.

### Assessment Questions

**Question 1:** Which library is most suited for data manipulation in Python?

  A) Dplyr
  B) Pandas
  C) Apache Spark
  D) NumPy

**Correct Answer:** B
**Explanation:** Pandas is a powerful data manipulation library in Python, designed for small to medium-sized datasets.

**Question 2:** What is the primary function of the Dplyr library in R?

  A) Data visualization
  B) Data manipulation
  C) Machine learning
  D) Data storage

**Correct Answer:** B
**Explanation:** Dplyr is specifically designed for data manipulation tasks in R, showcasing a set of intuitive and readable functions.

**Question 3:** What advantage does Apache Spark have over traditional data processing libraries?

  A) It requires less programming knowledge
  B) It can handle large datasets efficiently
  C) It is free of cost
  D) It cannot process data in real-time

**Correct Answer:** B
**Explanation:** Apache Spark is a distributed computing framework that excels in handling big data, processing large datasets efficiently through in-memory computation.

**Question 4:** Which operator is commonly used in Dplyr for chaining operations?

  A) - 
  B) + 
  C) %>% 
  D) & 

**Correct Answer:** C
**Explanation:** Dplyr uses the pipe operator (%>%) to facilitate the chaining of multiple data manipulation operations, enhancing code readability.

### Activities
- Choose a dataset (e.g., a CSV file) and use Pandas to clean the dataset by removing missing values and duplicates. Then, provide a summary of the cleaned data.
- Using Dplyr in R, conduct a series of manipulations on a specified dataset: filter out records, mutate a new column, and summarize the results grouped by a categorical variable.

### Discussion Questions
- How might the choice of data wrangling tools influence the efficiency of your data analysis?
- What factors should be considered when choosing between Pandas, Dplyr, and Apache Spark for a specific data project?
- Can you predict future trends in data wrangling tools? What features or improvements would you like to see in these libraries?

---

## Section 7: Hands-on Data Wrangling

### Learning Objectives
- Understand the importance of data wrangling in data science.
- Be able to identify and handle missing values effectively.
- Learn how to standardize formats and remove duplicates in datasets.
- Gain practical experience in transforming and filtering data using Pandas.

### Assessment Questions

**Question 1:** What is the primary purpose of data wrangling?

  A) To visualize data
  B) To prepare raw data for analysis
  C) To create a database
  D) To document data

**Correct Answer:** B
**Explanation:** The primary purpose of data wrangling is to transform and prepare raw data into a more usable format for analysis.

**Question 2:** Which of the following methods can be used to handle missing values?

  A) Dropping the entire dataset
  B) Filling with mean or median
  C) Ignoring them
  D) All of the above

**Correct Answer:** B
**Explanation:** Filling missing values with the mean or median is a common and effective method. Ignoring missing values can lead to inaccurate analyses, and dropping entire datasets is usually not feasible.

**Question 3:** What function in Pandas is used to drop duplicate rows?

  A) drop_na()
  B) drop_duplicates()
  C) remove_duplicates()
  D) clean()

**Correct Answer:** B
**Explanation:** The correct function to remove duplicate rows in a DataFrame is drop_duplicates().

**Question 4:** When should you consider standardizing data formats?

  A) When there are only a few entries
  B) When discrepancies in formats can lead to inaccurate analyses
  C) Never, format varies for analysis
  D) Only when asked by a supervisor

**Correct Answer:** B
**Explanation:** Standardizing data formats is crucial when discrepancies can affect the accuracy and reliability of analyses, especially when merging or comparing different datasets.

### Activities
- Using the provided sample dataset, identify and handle missing values in the 'Salary' column using at least two different methods - filling with the mean and dropping those rows. Document your process.
- Standardize the 'JoinDate' column to a consistent format and then convert it to a datetime object.
- Remove any duplicate records from the dataset based on the 'EmployeeID' and verify the changes.

### Discussion Questions
- What challenges have you faced in data wrangling in your previous experiences?
- How do you determine the best method for handling missing data in a dataset?
- In what scenarios might it be acceptable to ignore or keep duplicate records?

---

## Section 8: Challenges in Data Wrangling

### Learning Objectives
- Understand the common challenges in data wrangling and their implications on data analysis.
- Identify strategies to handle missing values, format inconsistencies, outliers, duplicates, and data type mismatches.
- Apply data transformation techniques in Python for cleaning data.

### Assessment Questions

**Question 1:** What is the most suitable method to handle missing values when they are few in a dataset?

  A) Imputation
  B) Deletion
  C) Ignoring them
  D) Duplicating the entries

**Correct Answer:** B
**Explanation:** When missing values are minimal, deleting the affected rows or columns can be effective without substantially skewing the dataset.

**Question 2:** Which technique helps to standardize different data formats in data wrangling?

  A) Duplicating data
  B) Transformation functions
  C) Regex pattern matching
  D) None of the above

**Correct Answer:** C
**Explanation:** Regular expressions (regex) can be utilized to match patterns and standardize various data formats effectively.

**Question 3:** What statistical method can be used to identify outliers in a dataset?

  A) Data Deletion
  B) IQR method
  C) Data Duplication
  D) Data Type Conversion

**Correct Answer:** B
**Explanation:** The Interquartile Range (IQR) method is widely used to identify outliers by measuring the spread of the middle 50% of a dataset.

**Question 4:** When dealing with duplicate entries in a dataset, what is one effective strategy?

  A) Keep duplicates as is
  B) Aggregate duplicate entries
  C) Ignore them
  D) Add additional duplicates

**Correct Answer:** B
**Explanation:** Aggregating duplicate entries allows for a coherent dataset that avoids redundancy and provides clearer insights.

**Question 5:** Why is it important to check for data type mismatches?

  A) It only matters for missing values
  B) It can adversely affect the analysis
  C) It has no effect on data processing
  D) It is only important for visualization

**Correct Answer:** B
**Explanation:** Incorrect data types can lead to errors in data processing and analysis, making validation essential during data wrangling.

### Activities
- Review a sample dataset with missing values. Identify the approach you would take to handle these missing values and provide a justification for your choice.
- Given a dataset with inconsistent date formats, write a Python function using regex to standardize the date formats.
- Analyze a provided dataset for outliers and write a brief report on your findings and the methodology you used to identify them. Include visualizations if necessary.
- Use a sample dataset with duplicate records to apply deduplication methods and present the cleaned dataset.

### Discussion Questions
- What are additional challenges you have faced in data wrangling that were not covered in the slide?
- How can organizations implement protocols to minimize data wrangling challenges before data analysis begins?
- In what ways can automation assist in the data wrangling process to improve efficiency?

---

## Section 9: Best Practices in Data Wrangling

### Learning Objectives
- Understand the importance of data wrangling and best practices involved.
- Identify and clean common data issues such as duplicates and missing values.
- Apply data transformation techniques such as normalization effectively.

### Assessment Questions

**Question 1:** What is the primary purpose of data wrangling?

  A) To create data visualizations
  B) To transform raw data into a usable format
  C) To run statistical analyses
  D) To store data in databases

**Correct Answer:** B
**Explanation:** The primary purpose of data wrangling is to transform and map raw data into an understandable format so that accurate and useful analysis can be performed.

**Question 2:** Which of the following is NOT a recommended practice in data wrangling?

  A) Removing duplicates
  B) Ignoring missing values
  C) Documenting the wrangling process
  D) Visualizing intermediate results

**Correct Answer:** B
**Explanation:** Ignoring missing values can lead to incomplete or biased analysis, which is why it is essential to handle them appropriately.

**Question 3:** What technique can be used to fill missing values in a dataset?

  A) Data encryption
  B) Data standardization
  C) Imputation
  D) Data concatenation

**Correct Answer:** C
**Explanation:** Imputation is a technique used to fill missing values in a dataset. Common methods include using the mean or median of the column.

**Question 4:** Why is normalizing data important in data wrangling?

  A) It increases data complexity
  B) It ensures comparability of metrics
  C) It eliminates the need for data cleaning
  D) It guarantees accurate data visualization

**Correct Answer:** B
**Explanation:** Normalizing data helps to scale the features within a common range, ensuring that they can be compared effectively.

### Activities
- Choose a dataset from an online repository (e.g., Kaggle) and perform the following steps: identify data types, clean the dataset by removing duplicates and handling missing values, and then normalize at least two numerical columns. Document each step with comments and notes.

### Discussion Questions
- What challenges have you encountered while wrangling data in your projects, and how did you address them?
- How does the quality of your data affect the insights you can derive from it?

---

## Section 10: Ethical Considerations in Data Processing

### Learning Objectives
- Understand the ethical principles surrounding data privacy and protection.
- Identify and explain key terms such as data minimization, informed consent, and anonymization.
- Evaluate compliance requirements related to data privacy regulations such as GDPR.

### Assessment Questions

**Question 1:** What is the main purpose of data minimization?

  A) To collect as much data as possible for future use.
  B) To reduce the risk associated with data processing.
  C) To increase data processing speed.
  D) To enhance data analysis capabilities.

**Correct Answer:** B
**Explanation:** Data minimization reduces the risk associated with data processing by limiting the amount of personal data collected.

**Question 2:** Which principle allows individuals to request the deletion of their data under GDPR?

  A) Right to Access
  B) Right to Rectification
  C) Right to Erasure
  D) Right to Data Portability

**Correct Answer:** C
**Explanation:** The Right to Erasure allows individuals to request the deletion of their personal data.

**Question 3:** What is a key practice for ensuring data security?

  A) Storing data in an unencrypted format.
  B) Implementing strong encryption methods.
  C) Sharing data freely without restrictions.
  D) Collecting data from unsecured sources.

**Correct Answer:** B
**Explanation:** Implementing strong encryption methods is a key practice for ensuring data security and protecting data from unauthorized access.

**Question 4:** What is the significance of the concept 'Privacy by Design'?

  A) It focuses on protecting data only after it has been collected.
  B) It means data protection must be considered during the design phase of projects.
  C) It allows companies to ignore privacy concerns until they arise.
  D) It emphasizes user consent only after data collection.

**Correct Answer:** B
**Explanation:** 'Privacy by Design' emphasizes that data protection should be integrated into the development process from the very beginning.

### Activities
- Conduct a case study analysis on a recent data breach incident and identify what ethical considerations were overlooked.
- Create a brief poster or presentation outlining the key principles of GDPR and best practices for data privacy.

### Discussion Questions
- In your opinion, how can organizations balance the need for data collection with the ethical implications of user privacy?
- What measures would you propose to enhance transparency in data processing practices?

---

## Section 11: Summary of Key Techniques

### Learning Objectives
- Understand and define key data wrangling techniques.
- Demonstrate the ability to clean, transform, aggregate, filter, and merge data effectively.

### Assessment Questions

**Question 1:** What is the main goal of data cleaning?

  A) To combine datasets
  B) To identify and fix errors in the dataset
  C) To enhance data visualization
  D) To filter irrelevant data

**Correct Answer:** B
**Explanation:** Data cleaning aims to identify and correct errors in the dataset, which is crucial for ensuring the reliability and validity of analysis.

**Question 2:** Which technique involves changing the format of data for analysis?

  A) Data Aggregation
  B) Data Transformation
  C) Data Filtering
  D) Data Merging

**Correct Answer:** B
**Explanation:** Data transformation refers to changing the format to a more suitable structure, making data compatible with analytical tools.

**Question 3:** Why is data aggregation significant?

  A) It allows for detailed data analysis at the record level.
  B) It simplifies the analysis and helps to identify overarching trends.
  C) It ensures all data is completely accurate.
  D) It is used only for filtering datasets.

**Correct Answer:** B
**Explanation:** Data aggregation combines data from multiple records into summary statistics, simplifying analysis and revealing trends.

**Question 4:** What does the technique of filtering allow analysts to do?

  A) Combine multiple datasets into one.
  B) Select subsets of data based on specific criteria.
  C) Change the format of data.
  D) Increase the size of the dataset.

**Correct Answer:** B
**Explanation:** Filtering allows analysts to focus on relevant data points, enhancing the accuracy of their insights.

**Question 5:** What is the purpose of merging datasets?

  A) To eliminate any duplicates.
  B) To enhance data visualization.
  C) To combine records from multiple datasets based on a common key.
  D) To aggregate data into summary statistics.

**Correct Answer:** C
**Explanation:** Merging datasets combines records from different sources based on a common key, providing richer insights.

### Activities
- Using a sample dataset, perform data cleaning by identifying and rectifying at least three different types of errors.
- Transform a dataset by changing the date format and scaling a numeric variable to a standard range.
- Aggregate sales data from transactions to monthly summaries and identify any trends in customer spending.
- Filter a dataset to focus on high-value customers who have spent over a specified amount within a defined period.

### Discussion Questions
- How do data cleaning techniques influence the overall integrity of data analysis?
- In what ways can data transformation facilitate better interactions with analytics tools?
- What challenges do you foresee when aggregating data from various sources?

---

## Section 12: Q&A and Discussion

### Learning Objectives
- Understand the key concepts and techniques of data wrangling.
- Be able to articulate the importance of data cleaning, transformation, and integration.
- Engage in discussions about the real-world applications of data wrangling techniques.

### Assessment Questions

**Question 1:** What is the primary purpose of data wrangling?

  A) To analyze data
  B) To visualize data
  C) To convert raw data into a usable format
  D) To store data

**Correct Answer:** C
**Explanation:** Data wrangling is essential for transforming raw data into a structured format that can be easily analyzed.

**Question 2:** Which of the following techniques is NOT commonly associated with data cleaning?

  A) Removing duplicates
  B) Filling in missing values
  C) Generating reports
  D) Outlier detection

**Correct Answer:** C
**Explanation:** Generating reports is typically related to data presentation rather than data cleaning, which focuses on preparing datasets for analysis.

**Question 3:** What process involves merging different datasets to provide a unified view?

  A) Data Aggregation
  B) Data Transformation
  C) Data Integration
  D) Data Visualization

**Correct Answer:** C
**Explanation:** Data integration refers to the process of combining data from different sources to create a cohesive dataset for analysis.

**Question 4:** What is a common challenge faced during the data cleaning process?

  A) Data visualization
  B) Handling missing data
  C) Data anonymization
  D) Data storage

**Correct Answer:** B
**Explanation:** Handling missing data is a significant challenge in data cleaning, as improper handling can lead to inaccurate analysis.

**Question 5:** Which tool is widely used for data wrangling and analysis?

  A) Microsoft Word
  B) SQL
  C) Adobe Photoshop
  D) Notepad

**Correct Answer:** B
**Explanation:** SQL is extensively used for data wrangling, allowing users to query, update, and manipulate data stored in databases.

### Activities
- Select a dataset you have worked with recently. Identify and perform at least three data cleaning techniques on it, such as removing duplicates, filling in missing values, and detecting outliers. Present your findings in class.
- Create a short presentation (3-5 slides) outlining how you would apply data transformation techniques, such as normalization and pivoting, on a raw dataset of your choice.

### Discussion Questions
- What challenges have you faced during data wrangling activities? How did you overcome them?
- Can anyone share a specific scenario where data wrangling significantly impacted the outcome of a data analysis project?

---

