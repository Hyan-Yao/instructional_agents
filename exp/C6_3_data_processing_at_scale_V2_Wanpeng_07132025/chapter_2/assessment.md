# Assessment: Slides Generation - Week 2: Introduction to Apache Spark

## Section 1: Introduction to Apache Spark

### Learning Objectives
- Understand the fundamental concepts and architecture of Apache Spark.
- Recognize the advantages of Apache Spark in big data processing.
- Identify the various components of Apache Spark and their applications.
- Appreciate the relevance of Apache Spark in current industry practices.

### Assessment Questions

**Question 1:** What is one of the main advantages of Apache Spark over traditional big data processing frameworks?

  A) It stores data on disk only.
  B) It uses in-memory data processing.
  C) It works only with Java.
  D) It can only process batch data.

**Correct Answer:** B
**Explanation:** Apache Spark leverages in-memory data processing, which significantly improves the speed of computations compared to disk-based frameworks like Hadoop.

**Question 2:** Which component of Apache Spark is specifically designed for real-time data processing?

  A) Spark SQL
  B) MLlib
  C) Spark Streaming
  D) GraphX

**Correct Answer:** C
**Explanation:** Spark Streaming is the component of Apache Spark specifically designed for processing real-time data streams and analyzing them as they come in.

**Question 3:** Which of the following programming languages is NOT supported by Apache Spark?

  A) Scala
  B) R
  C) Python
  D) Swift

**Correct Answer:** D
**Explanation:** Apache Spark supports Scala, R, Python, and Java, but does not natively support the Swift programming language.

**Question 4:** In which scenarios is Apache Spark particularly beneficial?

  A) Only for batch processing.
  B) For both real-time analytics and batch processing.
  C) Only for stream processing.
  D) It is not suitable for big data.

**Correct Answer:** B
**Explanation:** Apache Spark is versatile, allowing for both real-time analytics via Spark Streaming and batch processing, making it suitable for a wide range of applications.

### Activities
- Develop a simple Spark application that processes a real-time data stream from a Twitter API, analyzing sentiment based on tweets related to a specific topic.
- Create a Spark job to perform batch processing on a dataset to calculate average customer purchases, comparing results with traditional methods.

### Discussion Questions
- How can Apache Spark improve decision-making processes in organizations?
- What are the challenges of implementing Apache Spark in an existing infrastructure?
- In what scenarios would you choose Spark over other big data frameworks?

---

## Section 2: What is Apache Spark?

### Learning Objectives
- Understand the definition and purpose of Apache Spark.
- Recognize the key features that differentiate Spark from Hadoop.
- Identify the historical context and development of Apache Spark.
- Explore practical applications of Apache Spark in various industries.

### Assessment Questions

**Question 1:** What is a key differentiator of Apache Spark compared to Hadoop?

  A) In-memory processing
  B) Disk-based processing
  C) Limited to batch processing
  D) Requires more configuration

**Correct Answer:** A
**Explanation:** Apache Spark utilizes in-memory processing which allows for faster data computations compared to the disk-based processing of Hadoop.

**Question 2:** Which programming languages does Spark provide high-level APIs for?

  A) Java and C++
  B) Python and R
  C) Java, Scala, Python, and R
  D) C# and Ruby

**Correct Answer:** C
**Explanation:** Apache Spark provides high-level APIs in Java, Scala, Python, and R, making it very versatile for developers and data scientists.

**Question 3:** In which year did Apache Spark become an Apache Software Foundation project?

  A) 2009
  B) 2012
  C) 2014
  D) 2016

**Correct Answer:** C
**Explanation:** Apache Spark became an Apache Software Foundation project in 2014, which marked its official recognition and support for wider adoption.

**Question 4:** What is one of the built-in libraries provided by Apache Spark?

  A) MLlib
  B) Pandas
  C) NumPy
  D) Scikit-Learn

**Correct Answer:** A
**Explanation:** MLlib is one of the built-in libraries offered by Apache Spark for machine learning.

### Activities
- {'activity': "Design a data streaming pipeline using Apache Spark to analyze real-time sentiment from Twitter feeds. Create a plan that outlines the steps needed to connect to Twitter's API, process the streaming data, and conduct sentiment analysis."}

### Discussion Questions
- How do you think in-memory processing in Spark affects its performance in real-time data scenarios compared to traditional methods?
- In what scenarios would you choose to use Spark over Hadoop, and why?
- What are the implications of using Apache Spark's unified engine for data processing in terms of system architecture?

---

## Section 3: Spark Architecture Overview

### Learning Objectives
- Understand the fundamental components of Apache Spark architecture, including Driver, Cluster Manager, and Executors.
- Identify and explain the responsibilities and interactions of different components in a Spark application.

### Assessment Questions

**Question 1:** What is the primary role of the Driver in Spark architecture?

  A) To execute tasks and process data
  B) To manage the cluster resources
  C) To coordinate and schedule tasks
  D) To provide an interface for users

**Correct Answer:** C
**Explanation:** The Driver coordinates and schedules the tasks that need to be executed across the Spark cluster. It is essentially the control center of the Spark application.

**Question 2:** Which of the following is NOT a type of Cluster Manager used with Apache Spark?

  A) Standalone
  B) Apache Mesos
  C) Hadoop YARN
  D) Spark Server

**Correct Answer:** D
**Explanation:** Spark Server is not a recognized Cluster Manager for Apache Spark. The valid options are Standalone, Apache Mesos, and Hadoop YARN.

**Question 3:** What do Executors do in Spark architecture?

  A) Submit jobs to the Driver
  B) Allocate cluster resources
  C) Execute tasks and store intermediate data
  D) Convert user input into tasks

**Correct Answer:** C
**Explanation:** Executors are responsible for executing the tasks assigned by the Driver and managing the intermediate data that is generated during the processing.

**Question 4:** How does Spark improve the speed of processing large datasets?

  A) By running everything on disk
  B) Through in-memory computing
  C) By using only HDD storage
  D) By distributing tasks only across a single node

**Correct Answer:** B
**Explanation:** Spark utilizes in-memory computing, which allows it to process data much faster compared to traditional disk-based processing systems by storing intermediate results in memory.

### Activities
- Create a flowchart that illustrates the workflow of Apache Spark from job submission to result collection.
- Implement a simple Spark application that processes a dataset and generates output, demonstrating the interaction between the Driver and Executors.

### Discussion Questions
- How might the choice of Cluster Manager affect the performance of a Spark application?
- In what scenarios would in-memory processing provide significant benefits over traditional processing methods?

---

## Section 4: Core Abstractions in Spark

### Learning Objectives
- Understand the fundamentals of RDDs, DataFrames, and Datasets in Spark.
- Differentiate between the three core abstractions and their use cases.
- Apply Spark's abstractions in practical data processing scenarios.

### Assessment Questions

**Question 1:** What is an RDD in Apache Spark?

  A) A structured collection of data organized in named columns
  B) An immutable distributed collection of objects
  C) A library for real-time data processing
  D) A framework for machine learning

**Correct Answer:** B
**Explanation:** RDD stands for Resilient Distributed Dataset, which is an immutable distributed collection of objects that can be processed in parallel.

**Question 2:** What feature of DataFrames allows for SQL-like operations?

  A) Default schema
  B) Catalyst optimizer
  C) Lazy evaluation
  D) Type safety

**Correct Answer:** B
**Explanation:** DataFrames can leverage the Catalyst optimizer, which enables users to execute SQL queries on structured data easily.

**Question 3:** Which of the following is a primary benefit of using Datasets over DataFrames?

  A) They are always faster
  B) They provide type safety
  C) They require less code
  D) They can handle unstructured data more effectively

**Correct Answer:** B
**Explanation:** Datasets provide type safety, allowing developers to catch type-related errors at compile time, which DataFrames do not inherently provide.

**Question 4:** What happens when a transformation is applied to an RDD?

  A) The data is immediately updated
  B) A new RDD lineage is created but no computation happens yet
  C) The result is calculated instantly
  D) The RDD is converted to a DataFrame

**Correct Answer:** B
**Explanation:** Spark uses lazy evaluation for RDD transformations, meaning that no computations are performed until an action is called.

### Activities
- Create an RDD from a list of student grades and compute the average grade.
- Load a CSV file containing product sales into a DataFrame, then find the total sales for each product category using groupBy functionality.
- Transform a DataFrame into a Dataset using a case class in Scala and demonstrate filtering on the Dataset.

### Discussion Questions
- What are the advantages of using DataFrames over RDDs for structured data processing?
- In what scenarios might you prefer to use RDDs instead of DataFrames or Datasets?
- How does lazy evaluation improve the performance of data processing in Spark?

---

## Section 5: Resilient Distributed Datasets (RDDs)

### Learning Objectives
- Understand the definition and importance of RDDs in Apache Spark.
- Identify and explain the key features of RDDs including immutability, distribution, and fault tolerance.
- Demonstrate how RDDs enable fault tolerance through lineage and checkpointing.
- Apply knowledge of RDDs to create simple Spark applications utilizing RDD transformations.

### Assessment Questions

**Question 1:** What does RDD stand for?

  A) Resilient Distributed Data
  B) Resilient Data Distributions
  C) Resilient Distributed Datasets
  D) Robust Distributed Datasets

**Correct Answer:** C
**Explanation:** RDD stands for Resilient Distributed Datasets, which are the fundamental data structure in Apache Spark.

**Question 2:** Which feature of RDDs allows Spark to recover lost data?

  A) Lineage Tracking
  B) In-Memory Processing
  C) Immutable Nature
  D) Data Partitioning

**Correct Answer:** A
**Explanation:** Lineage Tracking allows Spark to keep track of the transformations applied and recover lost data by recomputing from the original dataset.

**Question 3:** What happens when an RDD transformation is applied?

  A) The original RDD is modified
  B) A new RDD is created
  C) The transformation is cached
  D) The RDD is deleted

**Correct Answer:** B
**Explanation:** RDD transformations are applied immutably, resulting in the creation of a new RDD while the original remains unchanged.

**Question 4:** What is the purpose of checkpointing in RDDs?

  A) To speed up processing
  B) To store RDDs in memory
  C) To save a snapshot and reduce lineage
  D) To prevent data corruption

**Correct Answer:** C
**Explanation:** Checkpointing saves a snapshot of the RDD to reliable storage, helping to reduce the lineage graph's length and preventing costly recomputation.

### Activities
- Implement a simple Spark application using RDDs to process a dataset. Load the dataset, perform at least two transformations (such as `map` and `filter`), and collect the results. Present the code and explain each step.
- Create a lineage graph for a set of transformations you perform on an RDD. Present how you would recover the lost data from a failed partition using Spark's lineage tracking.

### Discussion Questions
- How does the immutability feature of RDDs impact data integrity in distributed systems?
- In what scenarios would you prefer using RDDs over higher level abstractions like DataFrames or Datasets?
- Discuss real-world applications where the fault tolerance feature of RDDs is critical. What benefits does it provide?

---

## Section 6: DataFrames and Datasets

### Learning Objectives
- Understand the definitions and structures of DataFrames and Datasets in Apache Spark.
- Identify the advantages of DataFrames and Datasets over RDDs.
- Utilize DataFrames and Datasets for data manipulation tasks effectively.

### Assessment Questions

**Question 1:** What is a primary feature of DataFrames in Apache Spark?

  A) They are collections of untyped RDDs
  B) They are essentially static typed datasets
  C) They provide named columns and optimization
  D) They store data in a single partition

**Correct Answer:** C
**Explanation:** DataFrames provide a structured way to handle data with named columns, which enhances optimization and query performance.

**Question 2:** Which of the following best describes Datasets in Spark?

  A) They are non-distributed collections of data
  B) They offer compile-time type safety
  C) They are a type of RDD
  D) They do not support serialization

**Correct Answer:** B
**Explanation:** Datasets are a distributed collection that combines the benefits of RDDs and DataFrames, including strong type safety.

**Question 3:** Which optimizer is used for query optimization in DataFrames?

  A) GraphX Optimizer
  B) Catalyst Optimizer
  C) SQL Optimizer
  D) Tungsten Optimizer

**Correct Answer:** B
**Explanation:** The Catalyst Optimizer is used to optimize query execution plans for both DataFrames and Datasets.

**Question 4:** What type of data allows for compile-time type safety in Spark?

  A) RDDs
  B) DataFrames
  C) Datasets
  D) DataSets and DataFrames

**Correct Answer:** C
**Explanation:** Datasets provide compile-time type safety, which helps in catching errors during compilation rather than runtime.

### Activities
- Create a DataFrame in PySpark with sample data similar to the example given in the slide and perform a simple filtering operation.
- Define a case class in Scala and create a Dataset using that case class. Demonstrate operations like filtering and mapping.

### Discussion Questions
- In what scenarios would you prefer using DataFrames over RDDs and vice versa?
- How does the introduction of Datasets improve type safety in data manipulation?
- Can you think of a real-world application where DataFrames or Datasets would be beneficial? Discuss.

---

## Section 7: Basic Operations in Spark

### Learning Objectives
- Understand the difference between transformations and actions in Spark.
- Be able to identify common transformations and actions and their functions.
- Demonstrate proficiency in applying basic operations to manipulate datasets in Spark.

### Assessment Questions

**Question 1:** What is a key characteristic of transformations in Spark?

  A) They execute immediately.
  B) They are mutable.
  C) They create a new dataset without executing immediately.
  D) They return a final computed value.

**Correct Answer:** C
**Explanation:** Transformations are lazy and create new datasets without executing immediately until an action is invoked.

**Question 2:** Which of the following is an example of an action in Spark?

  A) filter()
  B) map()
  C) count()
  D) union()

**Correct Answer:** C
**Explanation:** The count() operation is an action that triggers execution and returns the number of elements in a dataset.

**Question 3:** What happens when an action is called on a dataset in Spark?

  A) The program crashes.
  B) The transformations are executed and results are returned.
  C) The dataset is permanently modified.
  D) Nothing happens.

**Correct Answer:** B
**Explanation:** When an action is invoked, Spark executes the transformations and returns the results to the driver program or external storage.

**Question 4:** Which transformation would you use to filter out elements in an RDD that do not meet certain criteria?

  A) map()
  B) flatMap()
  C) filter()
  D) reduce()

**Correct Answer:** C
**Explanation:** The filter() transformation allows you to select elements that satisfy a given condition, creating a new dataset.

### Activities
- Write a Spark program that creates an RDD from a list of names and then uses a transformation to convert all names to uppercase. Finally, use an action to collect the results and print them.
- Create a simulated dataset of employee salaries and apply transformations to filter out salaries below a threshold. Then, use an action to count the number of employees who meet the criteria.

### Discussion Questions
- Why is lazy evaluation an important feature of transformations in Spark?
- How might the choice of transformations and actions impact the performance of a Spark application?
- Can you think of scenarios outside of Spark where similar concepts of transformations and actions might apply?

---

## Section 8: Transformation and Action Operations

### Learning Objectives
- Understand the difference between transformations and actions in Apache Spark.
- Learn how to apply key transformations like map, filter, and flatMap.
- Gain practical experience using action commands such as count, collect, and first.

### Assessment Questions

**Question 1:** What type of operation is a 'map' in Apache Spark?

  A) Action
  B) Transformation
  C) Job
  D) None of the above

**Correct Answer:** B
**Explanation:** 'Map' is a transformation that applies a function to each element of the dataset, generating a new dataset.

**Question 2:** What does the 'collect()' action in Spark do?

  A) Performs a mathematical calculation
  B) Returns the number of elements
  C) Returns all elements to the driver program
  D) Filters elements based on a condition

**Correct Answer:** C
**Explanation:** 'Collect()' retrieves all the elements of an RDD and brings them back to the driver program as an array.

**Question 3:** Which of the following operations is an example of a transformation?

  A) count()
  B) collect()
  C) filter()
  D) first()

**Correct Answer:** C
**Explanation:** 'Filter()' is a transformation that creates a new dataset by filtering out elements that do not meet a certain condition.

**Question 4:** In Apache Spark, what happens when an action is called?

  A) The resource consumption is minimized
  B) The transformations are combined and executed
  C) Only some transformations are executed
  D) The job is cancelled

**Correct Answer:** B
**Explanation:** When an action is called, all the transformations defined in the logical plan are executed to return results.

### Activities
- Implement a simple Apache Spark job using PySpark to demonstrate the use of both transformations (map, filter) and actions (count, collect). Start with a dataset of integers, apply transformations to filter for even numbers, and then count them.

### Discussion Questions
- Why is it important that transformations in Spark are lazy? How does it benefit resource management?
- Can you think of a real-world application where using Spark's map transformation would be essential?

---

## Section 9: Working with Spark SQL

### Learning Objectives
- Define what a DataFrame is and describe its characteristics in Spark SQL.
- Explain how Spark SQL integrates with Spark Core and the benefits of using it.
- Summarize how Catalyst optimizes SQL query execution in Spark.

### Assessment Questions

**Question 1:** What is a DataFrame in Spark SQL?

  A) An unorganized collection of rows and columns
  B) A mutable collection of structured data
  C) An immutable distributed collection of data organized into named columns
  D) A command to execute SQL queries

**Correct Answer:** C
**Explanation:** A DataFrame is an immutable distributed collection of data organized into named columns that resembles a table in a relational database.

**Question 2:** How does Spark SQL optimize query execution?

  A) By simply executing queries sequentially
  B) By leveraging the Catalyst query optimizer
  C) By converting DataFrames to RDDs
  D) By reading raw data directly from the source

**Correct Answer:** B
**Explanation:** Spark SQL uses the Catalyst query optimizer to analyze and optimize the execution plan of SQL queries, enhancing performance.

**Question 3:** Which of the following formats is NOT supported by Spark SQL?

  A) JSON
  B) CSV
  C) HTML
  D) Parquet

**Correct Answer:** C
**Explanation:** Spark SQL supports various data formats including JSON, CSV, and Parquet, but it does not support HTML as a data source format.

**Question 4:** What is the primary purpose of creating a temporary view in Spark SQL?

  A) To store data permanently
  B) To enable complex SQL queries on a DataFrame
  C) To convert DataFrames into RDDs
  D) To improve data loading time

**Correct Answer:** B
**Explanation:** Creating a temporary view allows users to run complex SQL queries on a DataFrame easily without altering the original DataFrame.

### Activities
- Using a provided CSV file of employee records, create a DataFrame and generate a temporary view. Then write SQL queries to find the average salary by department and the maximum age of employees in each department.

### Discussion Questions
- How can you leverage Spark SQL in a real-time data processing pipeline?
- What advantages does Spark SQL offer over traditional relational database systems?
- In what scenarios might you prefer using DataFrame APIs over SQL queries in Spark SQL?

---

## Section 10: Example Use Cases of Apache Spark

### Learning Objectives
- Understand the various use cases of Apache Spark across different industries.
- Identify the benefits of using Apache Spark for real-time data processing and analytics.
- Gain familiarity with Spark's components, including MLlib and GraphX, and their applications.

### Assessment Questions

**Question 1:** What is one of the primary benefits of using Apache Spark for real-time data processing in retail?

  A) It reduces the need for data storage.
  B) It enables quick, data-driven decisions.
  C) It is the sole technology for machine learning.
  D) It only works with batch processing.

**Correct Answer:** B
**Explanation:** Real-time processing in Apache Spark allows businesses to analyze data as it comes in, enabling them to make immediate, informed decisions.

**Question 2:** Which Apache Spark component is used for machine learning?

  A) Spark SQL
  B) Spark Streaming
  C) MLlib
  D) GraphX

**Correct Answer:** C
**Explanation:** MLlib is the machine learning library in Apache Spark that provides scalable algorithms for various machine learning tasks.

**Question 3:** In which industry is Spark being used for predictive analytics to optimize treatment decisions?

  A) Finance
  B) Retail
  C) Healthcare
  D) Telecommunications

**Correct Answer:** C
**Explanation:** Healthcare providers utilize Spark to analyze patient records in order to identify trends that can aid in early disease detection.

**Question 4:** What advantage does Spark SQL offer to telecommunications companies?

  A) Increases call quality.
  B) Simplifies data ingestion for comprehensive reporting.
  C) Prevents network outages.
  D) Reduces customer complaints.

**Correct Answer:** B
**Explanation:** Spark SQL facilitates the ETL process, allowing telecommunications companies to integrate data from different sources for better reporting and analysis.

**Question 5:** Which library in Spark is specifically designed for graph processing?

  A) MLlib
  B) GraphX
  C) Spark SQL
  D) Spark Streaming

**Correct Answer:** B
**Explanation:** GraphX is the Spark library intended for graph processing, allowing for the analysis of relationships between data points.

### Activities
- Create a simple data processing pipeline using Apache Spark to analyze a dataset of your choice, focusing on deriving insights similar to the retail sales analysis example.
- Develop a machine learning model using MLlib on a provided dataset and document the steps taken to train the model, along with the results.

### Discussion Questions
- What other industries do you think could benefit from using Apache Spark for data processing? Explain why.
- Discuss the potential challenges that organizations might face when implementing Apache Spark in their data workflows.

---

## Section 11: Performance Considerations

### Learning Objectives
- Understand the importance of data partitioning for efficient data processing in Spark.
- Learn how to effectively use caching and persistence to optimize performance in Spark applications.
- Be able to configure Spark settings appropriately based on workload requirements.
- Identify the best data formats for efficient storage and processing in Spark.

### Assessment Questions

**Question 1:** What is the primary purpose of data partitioning in Spark?

  A) To increase the total data size
  B) To minimize data shuffling and balance the workload
  C) To simplify the data model
  D) To store data permanently

**Correct Answer:** B
**Explanation:** Data partitioning helps to distribute data evenly across the cluster, reducing data movement and improving parallel processing efficiency.

**Question 2:** Which caching method is best for storing data that is frequently accessed?

  A) write()
  B) cache()
  C) saveAsTextFile()
  D) collect()

**Correct Answer:** B
**Explanation:** Using the cache() method allows Spark to store an RDD or DataFrame in memory for faster access during later operations.

**Question 3:** Which serialization method is recommended for better performance in Spark?

  A) JSON serialization
  B) Kryo serialization
  C) Java serialization
  D) XML serialization

**Correct Answer:** B
**Explanation:** Kryo serialization is more efficient than Java serialization, leading to better performance in Spark applications.

**Question 4:** What is a common pitfall to avoid when organizing data operations in Spark?

  A) Using built-in functions
  B) Excessive shuffling of data
  C) Partitioning data effectively
  D) Caching frequently accessed data

**Correct Answer:** B
**Explanation:** Excessive shuffling can significantly hinder performance; thus, it's crucial to minimize shuffling operations.

**Question 5:** Which data format is often recommended for efficient I/O performance in Spark?

  A) CSV
  B) JSON
  C) Parquet
  D) Text

**Correct Answer:** C
**Explanation:** Parquet is a columnar storage file format optimized for use with Apache Spark, offering higher efficiency for I/O operations.

### Activities
- Create a Spark application that processes a dataset of your choice. Implement data partitioning based on a relevant key and demonstrate how it changes processing time before and after optimization.
- Write a Spark job that uses caching. Compare the performance of the job with and without caching to see the impact on execution time.
- Research a case study on Spark performance optimization and present the findings to the class, focusing on a specific application or dataset.

### Discussion Questions
- What challenges have you faced in optimizing Spark jobs, and how did you overcome them?
- How does data partitioning affect the performance of machine learning algorithms in Spark?
- In what scenarios would you prefer to use broadcast joins over regular joins, and why?

---

## Section 12: Conclusion and Further Learning

### Learning Objectives
- Understand the core functionalities of Apache Spark and its components.
- Recognize the advantages of using Spark for big data processing tasks.
- Apply Apache Spark to real-world data problems through practical exercises.

### Assessment Questions

**Question 1:** What is one major advantage of using Apache Spark over traditional data processing frameworks?

  A) It doesn't require any programming skills
  B) It processes data in-memory
  C) It only works with Java
  D) It is slower than Hadoop

**Correct Answer:** B
**Explanation:** Apache Spark processes data in-memory, which leads to significantly faster execution times compared to traditional disk-based processes.

**Question 2:** Which of the following is NOT a component of Apache Spark?

  A) Spark Streaming
  B) Spark SQL
  C) Spark GraphX
  D) Spark Notifier

**Correct Answer:** D
**Explanation:** Spark Notifier is not a component of Apache Spark; the correct components include Spark Streaming, Spark SQL, and Spark GraphX.

**Question 3:** Which library in Apache Spark is used for machine learning tasks?

  A) Spark SQL
  B) MLlib
  C) GraphX
  D) Spark Streaming

**Correct Answer:** B
**Explanation:** MLlib is the library within Apache Spark specifically designed for scalable machine learning tasks.

**Question 4:** What does RDD stand for in the context of Apache Spark?

  A) Rescaled Data Distribution
  B) Rapid Data Detection
  C) Resilient Distributed Dataset
  D) Real-time Data Development

**Correct Answer:** C
**Explanation:** RDD stands for Resilient Distributed Dataset, which is a core abstraction of Spark that allows for fault tolerance and parallel processing.

### Activities
- Design a data processing pipeline using Apache Spark to analyze live Twitter data for sentiment analysis. You will create a Spark Streaming application that retrieves tweets in real-time and classifies their sentiment based on text analysis, outputting insights into positive or negative trends.
- Implement a small project that reads a CSV file containing transaction data and utilizes Spark SQL to perform various transformations and aggregations, similar to the example shared in the slide.

### Discussion Questions
- How can you leverage Apache Spark for real-time data analytics in your field of study or industry?
- What are some potential challenges you might face when using Apache Spark for data processing and how can you mitigate them?
- In what scenarios do you think Spark would outperform traditional Hadoop MapReduce?

---

