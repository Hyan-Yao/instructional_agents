# Assessment: Slides Generation - Week 5: Data Manipulation in Python

## Section 1: Introduction to Data Manipulation in Python

### Learning Objectives
- Understand the significance of data manipulation in the data science workflow.
- Understand and utilize Pandas for data manipulation, specifically with DataFrames.

### Assessment Questions

**Question 1:** What is the primary purpose of data manipulation in data science?

  A) To visualize data
  B) To prepare data for analysis
  C) To store data permanently
  D) To generate random data

**Correct Answer:** B
**Explanation:** Data manipulation is essential in preparing raw data for effective analysis and extracting meaningful insights.

**Question 2:** Which of the following is NOT a function of data manipulation?

  A) Data Cleaning
  B) Data Transformation
  C) Data Storage
  D) Data Aggregation

**Correct Answer:** C
**Explanation:** Data manipulation focuses on adjusting and transforming data, while data storage concerns how data is preserved.

**Question 3:** Which Python library is most commonly used for data manipulation?

  A) NumPy
  B) Matplotlib
  C) Pandas
  D) Seaborn

**Correct Answer:** C
**Explanation:** Pandas is the leading library for data manipulation and analysis, providing robust tools for handling and processing data structures.

**Question 4:** What structure is primarily used in Pandas for data manipulation?

  A) Array
  B) List
  C) DataFrame
  D) Tuple

**Correct Answer:** C
**Explanation:** DataFrames are the fundamental data structures in Pandas that allow for manipulation of two-dimensional labeled data.

### Activities
- Create a DataFrame in Python using Pandas with a dataset of your choice which includes at least five entries. Perform data cleaning by removing any duplicate entries.
- Analyze a given dataset to find aggregate values (mean, sum) for specific columns using Pandas.

### Discussion Questions
- Can you think of a scenario in a project where data manipulation could significantly change the results?
- What are the potential consequences of inadequate data manipulation before analysis?

---

## Section 2: What is Pandas?

### Learning Objectives
- Understand the purpose and functionality of the Pandas library in data manipulation.
- Identify and utilize Pandas data structures, specifically Series and DataFrames.
- Apply basic data manipulation techniques using Pandas, such as filtering and aggregating data.

### Assessment Questions

**Question 1:** What is the primary data structure used in Pandas for handling two-dimensional data?

  A) Series
  B) DataFrame
  C) Array
  D) List

**Correct Answer:** B
**Explanation:** The primary data structure in Pandas for handling two-dimensional data is called a DataFrame, which is similar to a spreadsheet.

**Question 2:** Which of the following operations can NOT be performed directly with Pandas?

  A) Data filtering
  B) Data exporting to CSV
  C) Data visualization
  D) Database management

**Correct Answer:** D
**Explanation:** Pandas does not manage databases directly; it is used for data manipulation and can interface with databases for data import/export, but database management is not a function of Pandas.

**Question 3:** Which method would you use to read a CSV file into a Pandas DataFrame?

  A) pd.read_csv()
  B) pd.load_csv()
  C) pd.import_csv()
  D) pd.get_csv()

**Correct Answer:** A
**Explanation:** To read a CSV file into a Pandas DataFrame, the method used is pd.read_csv().

**Question 4:** In which scenario would you likely use the Pandas library?

  A) Rendering graphics for a video game
  B) Performing numerical simulations
  C) Analyzing social media data by cleaning and manipulating datasets
  D) Building a web framework

**Correct Answer:** C
**Explanation:** Pandas is specifically used for data manipulation and analysis, making it ideal for scenarios like analyzing and cleaning datasets.

### Activities
- Create a Pandas DataFrame from a dictionary containing information about your favorite movies (title, year, genre) and perform basic operations such as filtering for movies released after a certain year.
- Using a CSV file containing sales data, practice importing the data using Pandas and perform data cleaning by handling missing values and filtering specific product categories.

### Discussion Questions
- In what scenarios do you think Pandas would be more useful than Excel for data analysis?
- Discuss the advantages and disadvantages of using Pandas compared to other data manipulation libraries in Python.

---

## Section 3: Key Features of Pandas

### Learning Objectives
- Understand the basic data structures in Pandas: Series and DataFrames.
- Recognize the ease of use and performance optimizations provided by Pandas.
- Learn how to perform basic operations such as filtering and computing statistics with Pandas.

### Assessment Questions

**Question 1:** Which of the following is a primary data structure in Pandas?

  A) List
  B) Series
  C) Dictionary
  D) Tuple

**Correct Answer:** B
**Explanation:** Series is a one-dimensional labeled array that can hold various types of data and is one of the core data structures in Pandas.

**Question 2:** What is a DataFrame in Pandas?

  A) A one-dimensional array
  B) A two-dimensional array with labeled axes
  C) A method for data visualization
  D) A file format for data storage

**Correct Answer:** B
**Explanation:** A DataFrame is a two-dimensional labeled data structure with columns of potentially different types, making it similar to a spreadsheet.

**Question 3:** Which function would you use to compute the mean of a column named 'Salary' in a DataFrame?

  A) df['Salary'].average()
  B) df['Salary'].sum()
  C) df['Salary'].mean()
  D) df.mean('Salary')

**Correct Answer:** C
**Explanation:** The correct method to calculate the mean of the 'Salary' column in a DataFrame is df['Salary'].mean().

**Question 4:** What advantage does Pandas have in handling large datasets?

  A) It stores data in an external database.
  B) It supports chunking and lazy loading.
  C) It compresses data automatically.
  D) It converts data types frequently.

**Correct Answer:** B
**Explanation:** Pandas uses techniques like chunking and lazy loading to efficiently handle datasets that are larger than the available memory.

### Activities
- Create a Pandas DataFrame with the following data: Name, Age, Salary for three individuals. Then, filter the DataFrame to display only those individuals older than 30.
- Using a Series, create a list of monthly temperatures in degrees Celsius for a week (Monday to Sunday) and calculate the average temperature.

### Discussion Questions
- How might the features of Pandas improve the efficiency of data analysis tasks compared to using raw Python data structures?
- Discuss scenarios where using a DataFrame is more advantageous than using a traditional Python list or dictionary.

---

## Section 4: Understanding DataFrames

### Learning Objectives
- Understand the structure and characteristics of a DataFrame.
- Differentiate between DataFrames and other data structures like lists, dictionaries, and NumPy arrays.
- Gain proficiency in creating and manipulating DataFrames using Pandas.

### Assessment Questions

**Question 1:** What is a DataFrame primarily used for?

  A) Basic arithmetic operations
  B) Data manipulation and analysis
  C) Storing large binary files
  D) Creating user interfaces

**Correct Answer:** B
**Explanation:** DataFrames are specifically designed for data manipulation and analysis, making them ideal for handling structured datasets.

**Question 2:** Which of the following is a key characteristic of a DataFrame?

  A) It can only contain integer values.
  B) It has labeled axes.
  C) It is a one-dimensional structure.
  D) It cannot be resized.

**Correct Answer:** B
**Explanation:** DataFrames have labeled axes for both rows and columns, which aids in intuitive data retrieval and manipulation.

**Question 3:** How does a DataFrame differ from a dictionary?

  A) A DataFrame stores data in key-value pairs.
  B) A DataFrame has a structure for rows and columns.
  C) A DataFrame can only hold homogeneous data.
  D) A DataFrame is limited to numerical data.

**Correct Answer:** B
**Explanation:** While a dictionary can hold key-value pairs, a DataFrame provides an organized structure for rows and columns that enhances data manipulation capabilities.

**Question 4:** What type of data can a DataFrame hold?

  A) Only integers
  B) Only floats
  C) Homogeneous data types only
  D) Heterogeneous data types

**Correct Answer:** D
**Explanation:** DataFrames can hold heterogeneous data types, allowing different columns to contain different types of data (e.g., integers, floats, strings).

### Activities
- Create a DataFrame from a CSV file using Pandas. Import the CSV, display the DataFrame, and perform basic descriptive statistics on the data.
- Using the sample data provided in the slide (Name, Age, City), add another person to the DataFrame and demonstrate how to display the updated DataFrame.

### Discussion Questions
- What are some advantages of using DataFrames over traditional data structures in Python?
- In what scenarios might you choose to work with a DataFrame instead of a list or dictionary?
- Can you think of a project that would greatly benefit from using DataFrames? How would you apply DataFrames in that project?

---

## Section 5: Creating DataFrames

### Learning Objectives
- Understand the various methods to create a DataFrame in pandas.
- Gain familiarity with creating DataFrames from lists, dictionaries, CSV files, and NumPy arrays.
- Apply the knowledge of DataFrames to manipulate real data effectively.

### Assessment Questions

**Question 1:** Which of the following methods can be used to create a DataFrame in pandas?

  A) From a list of dictionaries
  B) From a dictionary of lists
  C) From a CSV file
  D) All of the above

**Correct Answer:** D
**Explanation:** DataFrames can be created from various sources including lists, dictionaries, and CSV files.

**Question 2:** What function is used to read a CSV file into a DataFrame?

  A) pd.open_csv()
  B) pd.read_csv()
  C) pd.load_csv()
  D) pd.import_csv()

**Correct Answer:** B
**Explanation:** The correct function to read a CSV file and convert it into a DataFrame is pd.read_csv().

**Question 3:** When creating a DataFrame from a dictionary, what do the keys represent?

  A) Row indexes
  B) Column labels
  C) The data itself
  D) None of the above

**Correct Answer:** B
**Explanation:** In a dictionary, the keys are used as column labels, while the corresponding values are used as the data for those columns.

**Question 4:** What is one advantage of creating a DataFrame from a NumPy array?

  A) It provides better performance for numerical data.
  B) It requires less coding.
  C) It can only handle numerical data.
  D) None of the above.

**Correct Answer:** A
**Explanation:** Creating a DataFrame from a NumPy array is advantageous because it provides better performance and is particularly useful when handling numerical data.

### Activities
- Create a DataFrame from the following dataset stored as a list of dictionaries: [{'ID': 1, 'Name': 'Alice'}, {'ID': 2, 'Name': 'Bob'}, {'ID': 3, 'Name': 'Charlie'}].
- Read a CSV file containing employee data (ensure 'employees.csv' is in the working directory) and create a DataFrame. Display the first 5 rows of the DataFrame.

### Discussion Questions
- Discuss the advantages of using a DataFrame over other data structures for data analysis in Python.
- How does the choice of data source (list vs. CSV file) affect the process of creating a DataFrame?

---

## Section 6: Data Inspection

### Learning Objectives
- Understand the importance of data inspection in the context of DataFrames.
- Be able to utilize the head(), tail(), and info() methods for inspecting data in Pandas.

### Assessment Questions

**Question 1:** What does the head() method in Pandas do?

  A) Displays the first 5 rows of a DataFrame
  B) Displays the last 5 rows of a DataFrame
  C) Shows a summary of the DataFrame
  D) Deletes the first 5 rows of a DataFrame

**Correct Answer:** A
**Explanation:** The head() method is used to display the first 5 rows of a DataFrame by default.

**Question 2:** Which method gives you a concise summary of a DataFrame's structure?

  A) tail()
  B) info()
  C) describe()
  D) shape()

**Correct Answer:** B
**Explanation:** The info() method provides a concise summary of the DataFrame, including the number of entries and data types.

**Question 3:** What is the default number of rows displayed by the tail() method?

  A) 3 rows
  B) 5 rows
  C) 10 rows
  D) 1 row

**Correct Answer:** B
**Explanation:** The tail() method, like head(), displays the last 5 rows by default.

**Question 4:** Which of the following methods can be used to quickly check for missing values in a DataFrame?

  A) head()
  B) tail()
  C) info()
  D) all of the above

**Correct Answer:** C
**Explanation:** The info() method provides information on non-null counts, which can help in detecting missing values.

### Activities
- Load a CSV file into a DataFrame using Pandas and utilize the head(), tail(), and info() methods to inspect its structure. Report back on what you observe regarding data types and any missing values.

### Discussion Questions
- What challenges might arise when inspecting large datasets, and how could you address them?
- Why is it important to understand a DataFrame's structure before performing any data cleaning or analysis?

---

## Section 7: Data Selection and Filtering

### Learning Objectives
- Understand how to select and filter rows and columns using Pandas.
- Apply filtering techniques to manipulate data based on specific conditions.
- Utilize methods like .loc, .iloc, and .query for efficient data selection.

### Assessment Questions

**Question 1:** Which method is used to select specific rows by label?

  A) df.iloc[]
  B) df.loc[]
  C) df.select()
  D) df.filter()

**Correct Answer:** B
**Explanation:** The 'loc' method allows selection based on row labels, while 'iloc' is for positional indexing.

**Question 2:** What operator is used to filter data based on multiple conditions where both must be true?

  A) ||
  B) &
  C) !
  D) and

**Correct Answer:** B
**Explanation:** The '&' operator is used to combine conditions such that both must be true for filtering.

**Question 3:** How do you select multiple columns from a DataFrame?

  A) df['column1', 'column2']
  B) df[['column1', 'column2']]
  C) df.columns[['column1', 'column2']]
  D) df.select(['column1', 'column2'])

**Correct Answer:** B
**Explanation:** To select multiple columns, you must pass a list of column names within double brackets.

**Question 4:** What is the purpose of the `.query()` method in DataFrame filtering?

  A) To summarize data
  B) To merge DataFrames
  C) To filter data using a query-like syntax
  D) To group data

**Correct Answer:** C
**Explanation:** The `.query()` method allows you to filter data using a more readable syntax similar to SQL.

### Activities
- Create a DataFrame using sample data for employees with at least 'Name', 'Age', 'Gender', and 'Salary' columns.
- Using the DataFrame, practice selecting different combinations of columns and filtering based on provided conditions, such as filtering for 'Salary' greater than 60000 or 'Age' less than 30.

### Discussion Questions
- What are some potential use cases for data selection and filtering in real-world data analysis?
- How does the choice of selection method affect the readability and efficiency of your code when working with large datasets?

---

## Section 8: Data Cleaning Techniques

### Learning Objectives
- Understand the importance of data cleaning for quality analysis.
- Identify common data cleaning techniques such as handling missing values, removing duplicates, and converting data types.
- Implement basic data cleaning operations using Pandas in Python.

### Assessment Questions

**Question 1:** What is the primary reason for handling missing values in a dataset?

  A) To increase the size of the dataset
  B) To improve data quality and prevent misleading analyses
  C) To eliminate duplicates
  D) To sort the data

**Correct Answer:** B
**Explanation:** Handling missing values is crucial to improve data quality, as they can otherwise lead to unreliable analyses and insights.

**Question 2:** Which of the following functions in Pandas is used to remove duplicate rows from a DataFrame?

  A) dropna()
  B) drop_duplicates()
  C) fillna()
  D) astype()

**Correct Answer:** B
**Explanation:** The drop_duplicates() function is used to identify and remove duplicate entries in a Pandas DataFrame.

**Question 3:** What technique can be used to handle a categorical variable with missing values?

  A) Remove the column
  B) Impute with the mode
  C) Do nothing
  D) Convert to numerical format

**Correct Answer:** B
**Explanation:** Imputing missing values in categorical data can be performed using the mode, which is the most frequently occurring value.

**Question 4:** In which scenario would you use data type conversions?

  A) To change a string representation of a number to an integer
  B) To append new data
  C) To visualize data
  D) To merge different datasets

**Correct Answer:** A
**Explanation:** Data type conversions are used to change data formats for proper analysis; for example, converting strings that represent numbers into integer data types.

### Activities
- Using the provided code snippets, create your own DataFrame with at least 5 rows containing some missing values and duplicates. Perform the necessary data cleaning steps to handle missing values through imputation and remove duplicate rows.
- Choose a dataset you frequently use in your analysis. Describe at least three instances where data cleaning techniques were applied and their impact on the analysis.

### Discussion Questions
- Why do you think handling missing values is significant in data analysis?
- Can you think of examples where data type mismatches could lead to errors in analysis?
- In what scenarios do you believe data cleaning might be overlooked, and what are the potential consequences?

---

## Section 9: Data Manipulation Operations

### Learning Objectives
- Understand the fundamental data manipulation operations in Pandas.
- Apply sorting, grouping, and aggregating techniques on sample datasets.
- Analyze the output of different data manipulation functions and their practical applications.

### Assessment Questions

**Question 1:** What method is used in Pandas to sort a DataFrame?

  A) groupby()
  B) sort_values()
  C) aggregate()
  D) filter()

**Correct Answer:** B
**Explanation:** The sort_values() method is used to sort a DataFrame based on specified column(s).

**Question 2:** Which function allows you to split data into subsets based on unique values of a column?

  A) merge()
  B) append()
  C) groupby()
  D) concat()

**Correct Answer:** C
**Explanation:** The groupby() function splits the data into subsets based on unique values of one or more columns.

**Question 3:** When aggregating data in Pandas, which of the following functions can you apply?

  A) min
  B) max
  C) mean
  D) All of the above

**Correct Answer:** D
**Explanation:** You can apply various aggregation functions like min, max, and mean when summarizing grouped data.

**Question 4:** What is the output of the following code snippet? df.groupby('Category').mean()

  A) Total number of entries in each category
  B) Mean of all numerical columns in each group
  C) Size of each group
  D) The first entry of each group

**Correct Answer:** B
**Explanation:** df.groupby('Category').mean() returns the mean of all numerical columns for each unique value in 'Category'.

### Activities
- Using the provided data, create a Pandas DataFrame and implement sorting by one of the columns. Capture and display the results.
- Develop a script that groups data by a specific column and calculates the aggregate functions (mean, sum, count) for another column. Present your findings.

### Discussion Questions
- How can sorting data impact the outcomes of an analysis?
- In what scenarios might grouping data provide misleading information?
- Can you think of a real-world example where data aggregation is critical for decision-making?

---

## Section 10: Merging and Joining DataFrames

### Learning Objectives
- Understand concepts from Merging and Joining DataFrames

### Activities
- Practice exercise for Merging and Joining DataFrames

### Discussion Questions
- Discuss the implications of Merging and Joining DataFrames

---

## Section 11: Real-World Applications

### Learning Objectives
- Understand the diverse applications of Pandas in real-world scenarios.
- Identify how Pandas can enhance data manipulation tasks across different industries.
- Learn practical skills in data cleaning, manipulation, and analysis using Pandas.

### Assessment Questions

**Question 1:** Which of the following tasks can Pandas NOT perform?

  A) Data cleaning
  B) Data visualization
  C) Data transformation
  D) Statistical modeling

**Correct Answer:** D
**Explanation:** Pandas excels at data cleaning, transformation, and basic data visualization via integration with libraries like Matplotlib, but it doesn't offer built-in statistical modeling capabilities.

**Question 2:** What is the purpose of calculating moving averages in financial data analysis with Pandas?

  A) To remove outliers
  B) To identify trends over time
  C) To increase dataset size
  D) To convert categorical data to numerical

**Correct Answer:** B
**Explanation:** Moving averages are used to smooth out short-term fluctuations and highlight longer-term trends in data.

**Question 3:** How does Pandas handle missing values?

  A) By deleting entire rows only
  B) By filling with zeros only
  C) By allowing custom methods to fill missing data
  D) By throwing an error

**Correct Answer:** C
**Explanation:** Pandas provides multiple methods to handle missing values, including filling with mean, median, or custom values, as well as dropping them.

**Question 4:** When segmenting customers in marketing analytics, what does an aggregate function like 'sum' do?

  A) Counts total customers
  B) Calculates the maximum sales
  C) Sums total sales for each segment
  D) Changes customer demographics

**Correct Answer:** C
**Explanation:** 'Sum' aggregates sales within each customer segment to provide insights into which segment performs better in terms of sales.

### Activities
- Use a sample dataset to practice data cleaning. Load a CSV file containing sales data, remove duplicates, and fill any missing values before performing basic analysis.
- Collect live data via web scraping using BeautifulSoup and store it in a DataFrame using Pandas. Analyze how the data structure changes and what challenges arise.

### Discussion Questions
- What are some challenges you face when cleaning and preparing data for analysis, and how can Pandas help address these issues?
- Can you think of other industries where data manipulation is crucial? How might Pandas be used in these fields?

---

## Section 12: Best Practices in Data Manipulation

### Learning Objectives
- Understand the importance of exploring the data structure before manipulation.
- Identify appropriate strategies for cleaning data effectively.
- Implement vectorized operations to enhance performance when working with data.
- Recognize the significance of checking for duplicates to maintain data integrity.
- Apply best practices in documentation and version control throughout the data manipulation process.

### Assessment Questions

**Question 1:** Which Pandas method is best for quickly checking the structure of a DataFrame?

  A) .head()
  B) .describe()
  C) .info()
  D) .dropna()

**Correct Answer:** C
**Explanation:** The .info() method provides a concise summary of the DataFrame including data types and non-null counts.

**Question 2:** What is an effective method for handling missing data in a DataFrame?

  A) Fill them with zeros
  B) Drop them unconditionally
  C) Both A and B
  D) Ignore them

**Correct Answer:** C
**Explanation:** Depending on the context, you can either fill missing values (e.g., with zeros) or drop rows with missing values to maintain data integrity.

**Question 3:** What is the benefit of using vectorized operations in Pandas?

  A) Slower execution time
  B) Reduces code readability
  C) Increases execution speed
  D) Requires more memory

**Correct Answer:** C
**Explanation:** Vectorized operations leverage optimized C and Fortran libraries, leading to faster computations compared to traditional Python loops.

**Question 4:** Why is it important to check for duplicates in your data?

  A) To decrease dataset size
  B) To improve aesthetic presentation
  C) To ensure data integrity
  D) To increase processing time

**Correct Answer:** C
**Explanation:** Checking for duplicates helps maintain data integrity, ensuring that analyses are based on accurate and unique data.

**Question 5:** What practice is recommended for documenting your data manipulation process?

  A) Avoid comments to reduce clutter
  B) Use consistent variable names
  C) Only document key functions
  D) Write minimal comments and rely on variable names

**Correct Answer:** B
**Explanation:** Using consistent variable names helps in understanding the code's purpose and improves readability for others or for future reference.

### Activities
- Activity 1: Create a Pandas DataFrame from a CSV file containing missing values. Document your process of cleaning the data, including how you will handle missing values and duplicates.
- Activity 2: Write a short script using vectorized operations in Pandas to calculate a new column in a DataFrame. Share your code and explain the advantages of using vectorization.

### Discussion Questions
- What challenges have you faced when cleaning a dataset, and how did you overcome them?
- How do you ensure that your data manipulations can be replicated by someone else?
- In what scenarios would you choose to fill in missing values instead of dropping them?

---

## Section 13: Summary and Key Takeaways

### Learning Objectives
- Understand the core functionalities of the Pandas library for data manipulation.
- Demonstrate the ability to perform basic data cleaning and transformation tasks using Pandas.

### Assessment Questions

**Question 1:** What is the primary library for data manipulation in Python?

  A) NumPy
  B) Pandas
  C) Matplotlib
  D) Scikit-learn

**Correct Answer:** B
**Explanation:** Pandas is the main library used for data manipulation tasks in Python due to its powerful data structures.

**Question 2:** Which method is used to read a CSV file into a DataFrame in Pandas?

  A) pd.read_excel()
  B) pd.read_json()
  C) pd.read_csv()
  D) pd.read_table()

**Correct Answer:** C
**Explanation:** The pd.read_csv() method is specifically designed to read CSV files into a DataFrame.

**Question 3:** What function can you use to remove missing values from a DataFrame?

  A) df.dropna()
  B) df.remove_na()
  C) df.eliminate_na()
  D) df.clean()

**Correct Answer:** A
**Explanation:** The df.dropna() function is used to remove any rows that contain missing values in a DataFrame.

**Question 4:** What is a DataFrame primarily used for?

  A) A one-dimensional array
  B) A simple text file
  C) Storing two-dimensional labeled data
  D) A collection of unrelated data types

**Correct Answer:** C
**Explanation:** A DataFrame is a two-dimensional labeled data structure often used to represent tabular data.

### Activities
- Download a sample dataset from a public data repository (like Kaggle) and perform the following tasks using Pandas: 1) Load the dataset; 2) Inspect the first few rows; 3) Clean any missing values; 4) Add a new calculated column; 5) Group the data by one of the existing columns and compute the average of another column.

### Discussion Questions
- In what ways does data manipulation with Pandas differ from data manipulation using other programming languages or tools?
- How can data manipulation skills be applied in real-world scenarios you are interested in?

---

## Section 14: Q&A Session

### Learning Objectives
- Understand and apply basic data manipulation techniques using Pandas.
- Differentiate between data manipulation and data analysis.
- Demonstrate the cleaning and transformation of datasets using appropriate Pandas functions.
- Recognize the importance of data manipulation in real-world applications.

### Assessment Questions

**Question 1:** Which of the following functions in Pandas is used to remove missing values?

  A) dropna()
  B) fillna()
  C) clean()
  D) remove_na()

**Correct Answer:** A
**Explanation:** The dropna() function in Pandas is specifically designed to remove rows or columns with missing values.

**Question 2:** What is the main difference between data manipulation and data analysis?

  A) Data manipulation is the same as data analysis
  B) Data manipulation is primarily about cleaning and transforming data, while data analysis is about interpreting that data
  C) Data analysis is for visualizing data only
  D) There’s no difference

**Correct Answer:** B
**Explanation:** Data manipulation involves altering data to make it suitable for analysis, while data analysis interprets the cleaned data to derive insights.

**Question 3:** Which Pandas function can be used to group data by certain criteria?

  A) merge()
  B) groupby()
  C) combine()
  D) aggregate()

**Correct Answer:** B
**Explanation:** The groupby() function in Pandas is used to split the data into groups based on some criteria, which can then be aggregated or transformed.

**Question 4:** What type of data structure does Pandas primarily use for data manipulation?

  A) Tuple
  B) List
  C) DataFrame
  D) Array

**Correct Answer:** C
**Explanation:** Pandas primarily uses DataFrames, which are 2-dimensional labeled data structures, for data manipulation.

**Question 5:** How can missing values in a DataFrame be filled with a specified value using Pandas?

  A) fill_values()
  B) fillna()
  C) set_na()
  D) replace_na()

**Correct Answer:** B
**Explanation:** The fillna() function in Pandas is used to fill missing values with a specific value.

### Activities
- Create a small DataFrame using Pandas and practice cleaning it by applying dropna() to remove missing values and fillna() to replace them.
- Use the groupby() function on a DataFrame of your choice (or create one) to compute the average of a numerical column grouped by a categorical column.

### Discussion Questions
- In what ways has data manipulation played an important role in your work or studies?
- Can anyone share a particular data transformation challenge they encountered and how they resolved it?
- What are some specific industries where data manipulation techniques can lead to significant insights?

---

