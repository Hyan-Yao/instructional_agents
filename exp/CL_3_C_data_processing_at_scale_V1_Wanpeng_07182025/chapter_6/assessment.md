# Assessment: Slides Generation - Week 6: Advanced Data Processing Techniques

## Section 1: Introduction to Advanced Data Processing Techniques

### Learning Objectives
- Understand the importance of optimizing data processing workflows in Apache Spark.
- Identify key concepts related to data optimization, resource management, and serialization formats.
- Apply practical techniques for improving data processing efficiency in Spark.

### Assessment Questions

**Question 1:** What is a key benefit of using appropriate data serialization formats in Spark?

  A) It simplifies the code structure
  B) It reduces network I/O and speeds up data reads
  C) It automatically optimizes all Spark jobs
  D) It eliminates the need for data partitioning

**Correct Answer:** B
**Explanation:** Using appropriate data serialization formats like Parquet can significantly reduce disk space and improve the speed of data reads.

**Question 2:** How can lazy evaluation in Spark be beneficial for data processing?

  A) It allows for immediate data retrieval
  B) It prevents data duplication
  C) It optimizes the execution plan by delaying execution until an action is called
  D) It simplifies the setup of Spark jobs

**Correct Answer:** C
**Explanation:** Lazy evaluation helps in optimizing the execution plan, as transformations are only computed when an action is triggered.

**Question 3:** Which Spark command is used to repartition a DataFrame for better parallel processing?

  A) DataFrame.write.partitionBy()
  B) DataFrame.repartition()
  C) DataFrame.groupBy()
  D) DataFrame.cache()

**Correct Answer:** B
**Explanation:** The 'DataFrame.repartition()' command allows for adjusting the number of partitions in a DataFrame, enhancing parallel processing.

**Question 4:** Why is cluster resource management critical in Spark?

  A) It ensures that all jobs complete at the same time
  B) It prevents data loss during processing
  C) It helps in avoiding bottlenecks and resource wastage during job execution
  D) It automatically scales the cluster size

**Correct Answer:** C
**Explanation:** Effective resource management allows Spark jobs to run efficiently, without encountering bottlenecks or wasting hardware resources.

### Activities
- Conduct an exercise where students implement a sample Spark job, focusing on optimizing data processing using techniques discussed in the presentation. Students should utilize repartitioning and select appropriate data serialization formats.

### Discussion Questions
- In what real-world scenarios have you observed the impact of data processing optimization in your work or studies?
- What challenges do you anticipate when trying to implement the optimization techniques discussed?

---

## Section 2: Learning Objectives

### Learning Objectives
- Identify advanced strategies in data processing.
- Implement optimizations within the Spark environment effectively.

### Assessment Questions

**Question 1:** What is one key strategy for advanced data processing discussed in the lesson?

  A) Using only SQL for data operations
  B) Data partitioning
  C) Sequential processing
  D) Avoiding data transformations

**Correct Answer:** B
**Explanation:** Data partitioning is crucial for optimizing parallel processing in large datasets.

**Question 2:** Which method can improve performance in Spark when dealing with large datasets?

  A) Writing data to disk for every operation
  B) Using broadcast variables
  C) Ignoring data caching
  D) Using default configurations without tuning

**Correct Answer:** B
**Explanation:** Using broadcast variables reduces data movement across nodes, improving performance.

**Question 3:** What is one advantage of caching data in Spark?

  A) It increases the amount of data processed at one time
  B) It allows faster access to a dataset for repeated operations
  C) It automatically optimizes Spark configurations
  D) It eliminates the need for data partitioning

**Correct Answer:** B
**Explanation:** Caching allows faster access to a dataset for repeated operations, significantly improving performance.

**Question 4:** Which processing method is recommended for handling continuous data streams?

  A) Batch processing
  B) Stream processing
  C) Static analysis
  D) Direct disk access

**Correct Answer:** B
**Explanation:** Stream processing is designed for handling continuous data input, making it suitable for real-time scenarios.

### Activities
- Create a mini-project where students implement a Spark application utilizing data partitioning and caching techniques. Students can document their process and results.
- Have students refine their Spark jobs by optimizing configurations based on specific datasets they choose, analyzing the performance differences.

### Discussion Questions
- Discuss the challenges you face when optimizing Spark applications and share strategies that have worked for you.
- How can data partitioning influence the performance of Spark applications, and what are your thoughts on its implementation?

---

## Section 3: Overview of Spark

### Learning Objectives
- Identify the core features and advantages of Apache Spark.
- Explain the significance of in-memory processing and how it impacts data processing speed.
- Discuss the role of RDDs in Spark and their benefits for fault tolerance.

### Assessment Questions

**Question 1:** What is one advantage Spark has over Hadoop?

  A) Spark is not distributed
  B) Spark processes data in memory
  C) Hadoop supports real-time processing
  D) Spark is slower

**Correct Answer:** B
**Explanation:** Spark processes data in memory, which allows for faster data processing.

**Question 2:** Which programming languages does Spark support?

  A) Only Scala
  B) Java and C++
  C) Scala, Python, Java, and R
  D) Python and Ruby

**Correct Answer:** C
**Explanation:** Spark provides APIs in multiple languages, including Scala, Python, Java, and R, making it accessible to a wide range of developers.

**Question 3:** What does RDD stand for in Spark?

  A) Reliable Data Distribution
  B) Resilient Distributed Datasets
  C) Remote Data Directory
  D) Redundant Data Framework

**Correct Answer:** B
**Explanation:** RDD stands for Resilient Distributed Datasets, which is a fundamental data structure in Spark offering fault tolerance.

**Question 4:** Which of the following is NOT a primary processing model supported by Spark?

  A) Machine Learning
  B) Stream Processing
  C) Graph Processing
  D) Data Warehousing

**Correct Answer:** D
**Explanation:** Data warehousing is not a primary processing model in Spark. Instead, Spark focuses on processing models like machine learning, stream processing, and graph processing.

### Activities
- Create a simple Spark application using PySpark that reads a CSV file and performs a basic data query using Spark SQL.
- Compile a performance comparison chart showing the differences in processing speed between Spark and Hadoop for common data processing tasks.

### Discussion Questions
- How does Spark's unified engine simplify the development of data processing workflows?
- Discuss the implications of using in-memory processing for real-time data applications compared to traditional disk-based processing.

---

## Section 4: Key Spark Functionalities

### Learning Objectives
- Identify and explain core functionalities of Spark, including RDDs, DataFrames, and Spark SQL.
- Differentiate between RDDs, DataFrames, and Spark SQL in terms of usability and performance.
- Demonstrate the application of RDDs and DataFrames through coding activities.

### Assessment Questions

**Question 1:** What does RDD stand for in Spark?

  A) Resilient Distributed Data
  B) Reliable Data Distribution
  C) Resilient Distributed Datasets
  D) Rapid Data Distribution

**Correct Answer:** C
**Explanation:** RDD stands for Resilient Distributed Datasets, which are a core concept in Spark.

**Question 2:** Which of the following is a key advantage of using RDDs?

  A) They require significant disk space.
  B) They provide lower-level API for data manipulation.
  C) They are mutable collections.
  D) They cannot be used with SQL.

**Correct Answer:** B
**Explanation:** RDDs provide a lower-level API for data manipulation allowing for finer control over data processing.

**Question 3:** What is one of the primary benefits of DataFrames over RDDs?

  A) DataFrames are slower to execute.
  B) DataFrames do not allow for schema enforcement.
  C) DataFrames support optimization through the Catalyst optimizer.
  D) DataFrames are less user-friendly than RDDs.

**Correct Answer:** C
**Explanation:** DataFrames support optimization through the Catalyst optimizer which enhances query execution compared to RDDs.

**Question 4:** How can Spark SQL interact with data sources?

  A) Only with Hive databases.
  B) Only with structured data sources.
  C) It can interact with various data sources like Hive, Avro, and Parquet.
  D) It cannot read or write data from external sources.

**Correct Answer:** C
**Explanation:** Spark SQL can interact with various data sources, allowing for a flexible approach to data management.

### Activities
- Create a simple Spark application using RDDs, where you read a text file, process the data (e.g., count the occurrences of each word), and display the results.
- Use DataFrames to read data from a CSV file, apply transformations (like filtering and aggregation), and then write the transformed data back to a new CSV file.
- Implement a Spark SQL query that fetches records from a DataFrame based on specific conditions (such as age > 30) and returns the selected columns.

### Discussion Questions
- Discuss the advantages and disadvantages of using RDDs compared to DataFrames in Spark applications.
- How does Spark SQL enhance the capabilities of Spark in terms of data analysis?
- Reflect on a use case where using Spark would be more beneficial than using traditional data processing frameworks.

---

## Section 5: Data Pipeline Optimization

### Learning Objectives
- Understand the importance of optimizing data pipelines in Spark.
- Explore various techniques for optimizing data processing, including data partitioning, caching, broadcast variables, and efficient file formats.

### Assessment Questions

**Question 1:** Which technique is used to save intermediate results in memory?

  A) Data partitioning
  B) Caching and persistence
  C) Broadcast variables
  D) Data serialization

**Correct Answer:** B
**Explanation:** Caching and persistence are techniques used to store intermediate results in memory, allowing for faster access during subsequent operations.

**Question 2:** What is the benefit of using broadcast variables in Spark?

  A) To reduce storage space on disk
  B) To minimize network traffic by distributing small datasets across all nodes
  C) To speed up data partitioning operations
  D) To enforce data encryption

**Correct Answer:** B
**Explanation:** Broadcast variables allow small datasets to be stored in memory on all executor nodes, which minimizes network traffic during Spark jobs.

**Question 3:** Which of the following file formats is recommended for optimal performance in Spark?

  A) CSV
  B) JSON
  C) Parquet
  D) TXT

**Correct Answer:** C
**Explanation:** Parquet is a columnar storage file format that supports efficient compression and encoding schemes, making it a preferred choice for Spark processing.

**Question 4:** What does the `coalesce()` function do?

  A) Increases the number of partitions
  B) Decreases the number of partitions without a full shuffle
  C) Clears the cache
  D) Displays the structure of the DataFrame

**Correct Answer:** B
**Explanation:** The `coalesce()` function decreases the number of partitions without performing a full shuffle, helping to optimize workload distribution.

### Activities
- Design an optimized data pipeline for a fictional e-commerce dataset, including decisions on partitioning, caching, and transformations.
- Implement a sample Spark job using Python that demonstrates the use of partitioning, caching, and using efficient file formats.

### Discussion Questions
- Discuss a scenario where data partitioning might negatively impact performance and explain why.
- What challenges might arise when trying to optimize data pipelines for streaming data versus batch processing?

---

## Section 6: Advanced Transformations in Spark

### Learning Objectives
- Understand and apply Spark's advanced transformations effectively.
- Differentiate between map, filter, and reduce transformations and their uses in data processing.
- Analyze the efficiency of transformations through lazy execution and optimization techniques.

### Assessment Questions

**Question 1:** What does the 'map' transformation do in Spark?

  A) Applies a function to each element of the dataset
  B) Combines elements of an RDD using a function
  C) Filters elements based on a predicate
  D) Changes the format of RDD to DataFrame

**Correct Answer:** A
**Explanation:** The 'map' transformation applies a function to each element of the dataset, creating a new RDD.

**Question 2:** Which of the following statements is true about the 'filter' transformation?

  A) It reduces the size of an RDD to a single value.
  B) It returns a new RDD containing only elements satisfying a condition.
  C) It changes the data type of the elements in the RDD.
  D) It performs a cumulative sum over the RDD.

**Correct Answer:** B
**Explanation:** The 'filter' transformation creates a new RDD that contains only those elements that satisfy the specified predicate.

**Question 3:** What is a key benefit of Spark's lazy execution of transformations?

  A) It guarantees immediate results.
  B) It allows the Spark engine to optimize the execution plan.
  C) It saves all intermediate data to disk.
  D) It automatically partitions the data.

**Correct Answer:** B
**Explanation:** Lazy execution allows Spark to optimize the execution plan based on the entire data processing pipeline before any computations are performed.

**Question 4:** Which transformation would you use to combine the elements of an RDD into a single value?

  A) Map
  B) Filter
  C) Reduce
  D) Collect

**Correct Answer:** C
**Explanation:** The 'reduce' transformation is used to aggregate elements of an RDD using a specified function, yielding a single output value.

### Activities
- Select a dataset and implement the 'map', 'filter', and 'reduce' transformations to demonstrate data manipulation. Present your results and any insights derived from the transformations.

### Discussion Questions
- Discuss how the lazy execution of transformations impacts the performance of Spark jobs.
- What are the advantages and disadvantages of using 'reduce' compared to 'map' and 'filter'? Provide examples.
- In what scenarios would you use partitioning strategies to improve the performance of transformations in Spark?

---

## Section 7: Performance Tuning Strategies

### Learning Objectives
- Understand and apply different performance tuning strategies in Apache Spark.
- Identify the impacts of partitioning and caching on Spark job performance.
- Effectively configure Spark settings to optimize resource utilization.

### Assessment Questions

**Question 1:** Which of the following is a recommended strategy for partitioning in Spark?

  A) Use as many partitions as possible
  B) Aim for an optimal number of partitions based on your data size and cluster
  C) Never use partitions
  D) Only create one partition

**Correct Answer:** B
**Explanation:** Finding an optimal number of partitions is crucial for balancing performance and resource usage in Spark.

**Question 2:** What is the purpose of caching in Spark?

  A) To permanently store data on disk
  B) To avoid recomputation and speed up iterative processes
  C) To compress data for faster retrieval
  D) To divide data into smaller partitions

**Correct Answer:** B
**Explanation:** Caching is used to store intermediate results in memory, which prevents the need for recomputation and speeds up subsequent actions.

**Question 3:** Which configuration parameter can be adjusted to manage memory allocation for executors in Spark?

  A) spark.executor.cores
  B) spark.executor.memory
  C) spark.sql.shuffle.partitions
  D) spark.dynamicAllocation.enabled

**Correct Answer:** B
**Explanation:** The spark.executor.memory parameter allows you to allocate a specific amount of memory to each executor, affecting performance.

**Question 4:** What operation does the `coalesce()` function perform?

  A) Increases the number of partitions with a full shuffle
  B) Decreases the number of partitions without a full shuffle
  C) Combines data from different sources into a single DataFrame
  D) Splits a DataFrame into multiple smaller DataFrames

**Correct Answer:** B
**Explanation:** `coalesce()` is used to reduce the number of partitions while avoiding a full shuffle, making it efficient for certain operations.

### Activities
- Review a Spark job you have worked on previously. Identify potential areas for performance tuning based on partitioning, caching, and configuration settings. Propose at least three specific optimizations.

### Discussion Questions
- Discuss the circumstances under which you would prefer to use `repartition()` over `coalesce()`. What are the trade-offs?
- In what scenarios could caching lead to issues in memory management? How would you mitigate these issues?

---

## Section 8: Integrating Spark with Other Tools

### Learning Objectives
- Examine the ways Spark integrates with other data processing tools and data sources.
- Identify specific use cases for integrating Spark with Hadoop, Hive, Kafka, and NoSQL databases.
- Demonstrate the ability to write basic code that interacts with Spark's integration capabilities.

### Assessment Questions

**Question 1:** Which tool allows Spark to run on a distributed file system?

  A) Kafka
  B) Hive
  C) Hadoop
  D) MongoDB

**Correct Answer:** C
**Explanation:** Spark can run on top of Hadoop's distributed file system (HDFS), providing access to large datasets stored in Hadoop.

**Question 2:** What functionality does Spark SQL provide in relation to Hive?

  A) It replaces Hive.
  B) It allows SQL queries over Hive tables.
  C) It does not integrate with Hive.
  D) It only writes data to Hive.

**Correct Answer:** B
**Explanation:** Spark SQL can read from and write to Hive tables, enabling users to execute SQL queries directly against Hive datasets.

**Question 3:** Which of the following databases can Spark directly integrate with for real-time analytics?

  A) MySQL
  B) Cassandra
  C) SQL Server
  D) SQLite

**Correct Answer:** B
**Explanation:** Spark offers integration with NoSQL databases like Cassandra, allowing for real-time analytics.

**Question 4:** How can Spark consume streaming data?

  A) Through HTTP requests only.
  B) By integrating with Kafka topics.
  C) Only via batch processing.
  D) By using flat files.

**Correct Answer:** B
**Explanation:** Spark can consume streaming data from Kafka topics, which is essential for real-time data processing.

### Activities
- Develop a short report detailing how Spark integrates with either Hadoop or Hive, including code snippets and practical use cases.
- Create a simple Spark application that reads data from a Kafka topic and processes it, documenting the integration steps.

### Discussion Questions
- Discuss the advantages and disadvantages of using Spark in conjunction with Hadoop versus using it standalone.
- How does the ability to integrate with various tools enhance Spark's capabilities in data processing?
- What other data sources or tools do you think would benefit from integrating with Spark and why?

---

## Section 9: Real-World Case Studies

### Learning Objectives
- Identify and understand the application of advanced data processing techniques in real-world scenarios.
- Analyze the impact of these techniques on decision-making and organizational efficiency.

### Assessment Questions

**Question 1:** What was the primary benefit of using Apache Spark in the e-commerce case study?

  A) It is a traditional data processing tool.
  B) It provides real-time data analytics capabilities.
  C) It does not require a distributed computing environment.
  D) It exclusively supports SQL-based queries.

**Correct Answer:** B
**Explanation:** Apache Spark offers real-time data analytics capabilities, making it ideal for processing large volumes of customer interaction data.

**Question 2:** In the Smart City traffic management case study, what was a key outcome of the predictive analysis performed?

  A) Traffic congestion increased by 30%.
  B) Reduced travel time by optimizing traffic light schedules.
  C) No significant changes were observed in traffic patterns.
  D) Increased need for manual traffic management.

**Correct Answer:** B
**Explanation:** The predictive analysis enabled dynamic adjustments to traffic signals, successfully reducing traffic delays.

**Question 3:** Which machine learning technique was applied to predict health risks in the healthcare case study?

  A) K-means Clustering
  B) Decision Trees
  C) Linear Regression
  D) Support Vector Machines

**Correct Answer:** B
**Explanation:** Decision Trees were used to analyze historical data and identify at-risk patients effectively.

**Question 4:** What role does Natural Language Processing (NLP) play in the healthcare predictive analytics case study?

  A) To create graphics for patient data.
  B) To analyze structured data only.
  C) To extract insights from unstructured clinical notes.
  D) To predict traffic patterns.

**Correct Answer:** C
**Explanation:** NLP was utilized to process unstructured clinical notes, enhancing the overall dataset for predictive modeling.

### Activities
- Divide students into groups to select one of the case studies and prepare a brief presentation highlighting the data processing techniques used and their outcomes. Each group will then present their findings to the class.

### Discussion Questions
- What challenges might organizations face when implementing advanced data processing techniques in their operations?
- How can businesses ensure that the insights derived from data analysis are actionable and improve their strategies?

---

## Section 10: Ethics in Data Processing

### Learning Objectives
- Understand and articulate the key ethical considerations in data processing.
- Identify the requirements and implications of GDPR and HIPAA regulations.

### Assessment Questions

**Question 1:** Which regulation primarily protects the personal data of EU citizens?

  A) HIPAA
  B) GDPR
  C) CCPA
  D) FERPA

**Correct Answer:** B
**Explanation:** GDPR (General Data Protection Regulation) is specifically designed to protect the data rights of individuals within the European Union.

**Question 2:** What is a critical requirement under HIPAA?

  A) Data must always be stored in cloud servers.
  B) Organizations must implement safeguards to protect PHI.
  C) Information can be shared freely among healthcare providers.
  D) Patients have no control over their health data.

**Correct Answer:** B
**Explanation:** HIPAA mandates covered entities to implement administrative, physical, and technical safeguards to ensure the protection of Protected Health Information (PHI).

**Question 3:** Which of the following is NOT considered a right under GDPR?

  A) Right to Access
  B) Right to Data Portability
  C) Right to Manipulate Data
  D) Right to Erasure

**Correct Answer:** C
**Explanation:** The Right to Manipulate Data is not a recognized right under GDPR. GDPR focuses on the control and protection of personal data, not manipulation.

**Question 4:** What is the maximum fine an organization can face under GDPR for non-compliance?

  A) €5 million
  B) €20 million or 4% of annual global turnover
  C) €1 million
  D) None of the above

**Correct Answer:** B
**Explanation:** GDPR imposes heavy fines that can reach up to €20 million or 4% of the organization's annual global turnover, whichever is higher.

### Activities
- Draft a compliance plan that outlines key measures an organization should implement to ensure ethical data processing according to GDPR and HIPAA. Include steps on obtaining user consent and ensuring data transparency.

### Discussion Questions
- Discuss the importance of transparency in data processing. How can organizations enhance transparency with their users?
- In your opinion, what ethical challenges do organizations face when implementing data processing practices? Provide examples.

---

## Section 11: Hands-On Exercise

### Learning Objectives
- Implement optimization techniques to enhance Spark data processing workflows.
- Utilize best practices for effective resource utilization in Spark.
- Analyze and interpret performance metrics from Spark jobs.

### Assessment Questions

**Question 1:** Which of the following is an important technique for optimizing Spark jobs?

  A) Data joining as often as possible
  B) Applying data source formats indiscriminately
  C) Data partitioning and caching
  D) Processing entire datasets in a single job

**Correct Answer:** C
**Explanation:** Data partitioning and caching significantly enhance performance by enabling parallel processing and quick data access.

**Question 2:** What is the benefit of using DataFrames over RDDs in Spark?

  A) DataFrames only support numeric data types.
  B) DataFrames provide built-in optimization features.
  C) DataFrames cannot handle complex data structures.
  D) DataFrames require less memory overall.

**Correct Answer:** B
**Explanation:** DataFrames utilize Catalyst Optimizer and Tungsten execution backend, which optimize queries and improve performance.

**Question 3:** What is the purpose of caching a DataFrame in Spark?

  A) To remove unneeded data
  B) To improve the performance of transformations by reducing read time
  C) To ensure data is stored permanently
  D) To convert DataFrames to RDDs

**Correct Answer:** B
**Explanation:** Caching a DataFrame keeps it in memory, which accelerates subsequent operations that require re-accessing this data.

**Question 4:** When should you consider partitioning your data in Spark?

  A) When you have a small dataset that fits in memory.
  B) When your dataset is too large for single node processing.
  C) Partitioning is not necessary if you just want to load data.
  D) Partitioning should always be avoided.

**Correct Answer:** B
**Explanation:** Partitioning is beneficial when dealing with large datasets as it allows for parallel processing across multiple nodes.

### Activities
- Work in small groups to optimize a given Spark job by applying data partitioning and caching techniques. Record the job's performance metrics before and after the optimizations.
- Present your group's findings and analysis to the class, focusing on significant optimizations made and the impact on performance.

### Discussion Questions
- What challenges do you face when optimizing Spark jobs, and how can these be overcome?
- Can you think of a scenario where caching might not be beneficial? Explain your reasoning.
- Discuss how monitoring performance metrics can lead to continuous improvement in Spark workflows.

---

## Section 12: Wrap-Up and Q&A

### Learning Objectives
- Summarize the key concepts discussed in the session related to advanced data processing techniques.
- Engage actively in clarifying questions to enhance understanding of the material.
- Demonstrate the ability to apply the discussed techniques and concepts in practical scenarios.

### Assessment Questions

**Question 1:** Which Spark feature enhances its performance over traditional data processing tools?

  A) In-memory processing
  B) Disk-based storage
  C) Sequential processing
  D) Non-distributed execution

**Correct Answer:** A
**Explanation:** Spark's in-memory processing significantly reduces the latency of data processing tasks compared to traditional disk-based processing methods.

**Question 2:** What is the primary benefit of using Parquet over CSV for data storage?

  A) Larger file size
  B) Columnar storage enhancing query performance
  C) Easier to read in Excel
  D) Compatibility with older systems

**Correct Answer:** B
**Explanation:** Parquet uses a columnar storage format that allows for efficient querying and better performance with large datasets versus row-based formats like CSV.

**Question 3:** Which Spark operation is used for optimizing the number of partitions in a dataset?

  A) reduceByKey
  B) filter
  C) coalesce
  D) map

**Correct Answer:** C
**Explanation:** The coalesce operation reduces the number of partitions in a Spark DataFrame, which can optimize certain data processing tasks.

**Question 4:** Which streaming process is used in Spark to handle real-time data?

  A) Hadoop Streaming
  B) Spark SQL
  C) Spark Streaming
  D) Batch Processing

**Correct Answer:** C
**Explanation:** Spark Streaming is designed for processing real-time data streams, enabling the handling of live data in an efficient manner.

### Activities
- Prepare a short presentation on the differences between batch processing and stream processing, including when each should be used.
- Conduct a hands-on exercise where students optimize a provided Spark workflow by adjusting parameters like memory allocation and partitioning.

### Discussion Questions
- What challenges did you encounter while working with the Spark Streaming demo?
- Can you provide an example of a scenario where you would prefer using a broadcast join?
- Reflecting on the week’s material, which technique do you think will be the most beneficial for your future projects?

---

