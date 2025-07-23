# Assessment: Slides Generation - Week 6: Data Manipulation with SQL

## Section 1: Introduction to Data Manipulation with SQL

### Learning Objectives
- Understand the role of SQL in data manipulation.
- Identify the importance of data management in business intelligence.
- Recognize different types of SQL commands and their uses.

### Assessment Questions

**Question 1:** What is SQL primarily used for?

  A) Data manipulation
  B) Data visualization
  C) Data storage
  D) Data encryption

**Correct Answer:** A
**Explanation:** SQL is used primarily for data manipulation within databases.

**Question 2:** Which SQL statement is used to modify existing records?

  A) SELECT
  B) INSERT
  C) UPDATE
  D) DELETE

**Correct Answer:** C
**Explanation:** The UPDATE statement is specifically used to modify existing records in a database.

**Question 3:** What does DDL stand for in SQL?

  A) Data Definition Language
  B) Data Development Language
  C) Data Description Language
  D) Data Delivery Language

**Correct Answer:** A
**Explanation:** DDL stands for Data Definition Language, which is used to manage the structure of database objects.

**Question 4:** To remove an employee record with the last name 'Doe', which SQL command would you use?

  A) DELETE FROM employees WHERE last_name = 'Doe'
  B) REMOVE FROM employees WHERE last_name = 'Doe'
  C) DROP FROM employees WHERE last_name = 'Doe'
  D) CLEAR FROM employees WHERE last_name = 'Doe'

**Correct Answer:** A
**Explanation:** The correct command to remove records from a table is the DELETE statement.

### Activities
- Create a simple SQL script that includes a SELECT statement to retrieve all columns from a table named 'products'. Then, write an INSERT statement to add a new product to 'products'.

### Discussion Questions
- Why do you think SQL is considered essential for data analysts and data scientists?
- How does the ability to manipulate data with SQL impact decision-making in businesses?

---

## Section 2: Understanding SQL Queries

### Learning Objectives
- Construct SQL queries using essential clauses.
- Interpret the result of a simple SQL query.
- Understand the purpose of the SELECT, FROM, and WHERE clauses in SQL.

### Assessment Questions

**Question 1:** Which SQL clause is used to filter records?

  A) SELECT
  B) FROM
  C) WHERE
  D) ORDER BY

**Correct Answer:** C
**Explanation:** The WHERE clause is used for filtering records in SQL queries.

**Question 2:** What does the SELECT clause do in an SQL query?

  A) Specifies the data source
  B) Defines the columns to retrieve
  C) Joins multiple tables
  D) Orders the results

**Correct Answer:** B
**Explanation:** The SELECT clause defines which columns to retrieve from the database.

**Question 3:** In SQL, what keyword is used to select all columns from a table?

  A) ALL
  B) *
  C) SELECT_ALL
  D) COLUMN

**Correct Answer:** B
**Explanation:** The asterisk (*) is used to select all columns from a table.

**Question 4:** Which SQL clause would you use to specify the table from which data should be retrieved?

  A) SELECT
  B) FILTER
  C) FROM
  D) WHERE

**Correct Answer:** C
**Explanation:** The FROM clause identifies the table from which to retrieve data.

### Activities
- Write a basic SQL query that retrieves the 'name' and 'email' of contacts from a 'customers' table where the status is 'active'.
- Create a SQL statement that selects all columns from a 'products' table where the price is greater than 100.

### Discussion Questions
- Why is it important to understand the structure of SQL queries?
- How do the SELECT, FROM, and WHERE clauses interact to form a complete query?
- Can you think of a scenario where using the WHERE clause could significantly impact your query results?

---

## Section 3: Types of SQL Queries

### Learning Objectives
- Differentiate between DDL, DML, DCL, and TCL.
- Identify appropriate SQL statements for data tasks.
- Understand the purpose of each SQL command in managing a database.

### Assessment Questions

**Question 1:** Which type of SQL statement is used to manipulate data?

  A) DDL
  B) DML
  C) DCL
  D) TCL

**Correct Answer:** B
**Explanation:** DML (Data Manipulation Language) is used for manipulating data.

**Question 2:** What command would you use to add a new column to an existing table?

  A) INSERT
  B) ALTER
  C) UPDATE
  D) DROP

**Correct Answer:** B
**Explanation:** The ALTER command is used to modify an existing database object, including adding new columns.

**Question 3:** Which SQL statement would you use to provide user access privileges?

  A) COMMIT
  B) GRANT
  C) ROLLBACK
  D) DELETE

**Correct Answer:** B
**Explanation:** The GRANT command is used to provide user access privileges to database objects.

**Question 4:** Which statement is correct about Transaction Control Language (TCL)?

  A) It is used to define the structure of database objects.
  B) It manages how transactions are processed in a database.
  C) It is used for granting permissions to users.
  D) It manipulates the data in a table.

**Correct Answer:** B
**Explanation:** TCL manages how transactions are processed and ensures data integrity.

### Activities
- 1. Create a DDL command that creates a new table named 'Departments' with at least two fields.
- 2. Write a DML command to insert a record into the 'Employees' table.
- 3. Use the GRANT command to give a user named 'AdminUser' the SELECT permission on the 'Employees' table.

### Discussion Questions
- Why is it important to separate SQL commands into DDL, DML, DCL, and TCL?
- How does using TCL commands like COMMIT and ROLLBACK help in maintaining data integrity in a database?
- In what scenarios would you prefer using DCL commands, and why is security vital in database management?

---

## Section 4: Joins in SQL

### Learning Objectives
- Explain the different types of SQL joins.
- Use SQL joins to combine data from multiple tables.
- Determine the appropriate type of join based on specific data retrieval requirements.

### Assessment Questions

**Question 1:** What type of join returns all records from both tables?

  A) INNER JOIN
  B) LEFT JOIN
  C) RIGHT JOIN
  D) FULL OUTER JOIN

**Correct Answer:** D
**Explanation:** FULL OUTER JOIN returns all records when there is a match in either left or right table records.

**Question 2:** Which join would you use to retrieve all employees regardless of whether they belong to a department?

  A) INNER JOIN
  B) LEFT JOIN
  C) RIGHT JOIN
  D) FULL OUTER JOIN

**Correct Answer:** B
**Explanation:** LEFT JOIN returns all records from the left table, including those employees not belonging to any departments, with NULLs for department data.

**Question 3:** What will the RIGHT JOIN return if there are no matching records in the left table?

  A) All records from the left table, NULL for right
  B) All records from the right table, NULL for left
  C) Only matching records from both tables
  D) No records at all

**Correct Answer:** B
**Explanation:** RIGHT JOIN returns all records from the right table, filling in NULL for the records from the left table which have no matches.

**Question 4:** If you want to combine the results of a LEFT JOIN and RIGHT JOIN, which SQL join would you use?

  A) INNER JOIN
  B) LEFT JOIN
  C) RIGHT JOIN
  D) FULL OUTER JOIN

**Correct Answer:** D
**Explanation:** FULL OUTER JOIN is used to combine results from both LEFT and RIGHT JOIN, returning all records with matching NULLs where no matches exist.

### Activities
- Write SQL queries using INNER JOIN, LEFT JOIN, RIGHT JOIN, and FULL OUTER JOIN with sample tables, and analyze the returned results.
- Create a scenario with two tables (e.g., Orders and Products) and explain which join type would be most appropriate for different reporting needs.

### Discussion Questions
- In what situations would you prefer using a LEFT JOIN over an INNER JOIN?
- How do NULL values impact the analysis of data when using different types of joins?
- Can you think of a practical example in your field where using FULL OUTER JOIN is crucial?

---

## Section 5: Using Joins Effectively

### Learning Objectives
- Identify scenarios for effective join usage.
- Implement joins under various conditions.
- Differentiate between the various join types and their outputs.

### Assessment Questions

**Question 1:** Which join only returns rows with matching values in both tables?

  A) LEFT JOIN
  B) INNER JOIN
  C) FULL OUTER JOIN
  D) CROSS JOIN

**Correct Answer:** B
**Explanation:** INNER JOIN returns rows when there is a match in both tables.

**Question 2:** What does a LEFT JOIN return if there are no matches in the right table?

  A) All rows from both tables
  B) Only rows from the right table
  C) Rows from the left table with NULLs for non-matching right table rows
  D) An error

**Correct Answer:** C
**Explanation:** LEFT JOIN returns all rows from the left table and NULL for non-matching rows from the right table.

**Question 3:** In which scenario would you use a FULL OUTER JOIN?

  A) To retrieve only matched records
  B) To get all records, regardless of matches
  C) To match rows using a common key
  D) To combine data without duplicates

**Correct Answer:** B
**Explanation:** FULL OUTER JOIN returns all rows from both tables, along with unmatched rows, which will have NULLs.

### Activities
- Using a provided dataset, users should execute several types of joins (INNER, LEFT, RIGHT, FULL) to answer the following questions: 1) Which customers have made orders? 2) Which customers have never made an order? 3) Display all orders, including those that may not have matching customers.

### Discussion Questions
- How might the choice of join type affect the results of your SQL query?
- In what real-world scenarios have you encountered the need for complex joins?
- Discuss the performance implications of using different types of joins in large datasets.

---

## Section 6: Aggregation Functions in SQL

### Learning Objectives
- Demonstrate the use of aggregation functions in SQL.
- Interpret and explain results from aggregate queries.

### Assessment Questions

**Question 1:** Which function would you use to find the total salaries of all employees?

  A) COUNT
  B) SUM
  C) AVG
  D) MIN

**Correct Answer:** B
**Explanation:** SUM is used to calculate the total of a numeric column, such as salaries.

**Question 2:** What does the COUNT function return when applied to a column?

  A) Total of numerical values
  B) Average of all values
  C) Maximum value in the column
  D) Number of non-null values

**Correct Answer:** D
**Explanation:** COUNT returns the number of non-null values in a specified column.

**Question 3:** If you need to find the earliest hire date for employees, which function would you use?

  A) MAX
  B) MIN
  C) AVG
  D) COUNT

**Correct Answer:** B
**Explanation:** MIN finds the minimum value in a column, which in this case is the earliest hire date.

**Question 4:** Which of the following aggregation functions will ignore NULL values?

  A) COUNT
  B) SUM
  C) AVG
  D) Both B and C

**Correct Answer:** D
**Explanation:** SUM and AVG ignore NULL values when calculating the total and average.

### Activities
- Write SQL queries that utilize each of the aggregation functions (COUNT, SUM, AVG, MIN, MAX) on a sample dataset. Ensure you group results appropriately where necessary.

### Discussion Questions
- What are some real-world scenarios where aggregation functions would be particularly useful?
- How could aggregation functions contribute to decision-making in a business context?

---

## Section 7: Group By and Having Clauses

### Learning Objectives
- Explain the function of GROUP BY and HAVING clauses.
- Utilize GROUP BY effectively to analyze data.
- Apply aggregate functions to summarize data within groups.
- Differentiate between the WHERE and HAVING clauses in SQL.

### Assessment Questions

**Question 1:** What is the purpose of the HAVING clause?

  A) Filter rows before grouping
  B) Filter grouped results
  C) Create a group
  D) Sort groups

**Correct Answer:** B
**Explanation:** The HAVING clause is used to filter records after aggregation.

**Question 2:** Which of the following SQL statements uses GROUP BY correctly?

  A) SELECT * FROM sales WHERE amount > 100 GROUP BY salesperson;
  B) SELECT salesperson, COUNT(*) FROM sales GROUP BY salesperson;
  C) SELECT salesperson FROM sales GROUP BY amount;
  D) SELECT amount FROM sales GROUP BY salesperson;

**Correct Answer:** B
**Explanation:** Option B correctly uses GROUP BY to aggregate data by 'salesperson'.

**Question 3:** What must always follow a GROUP BY clause?

  A) A WHERE clause
  B) An ORDER BY clause
  C) An aggregate function
  D) A JOIN clause

**Correct Answer:** C
**Explanation:** An aggregate function is typically used to summarize the data in the groups defined by GROUP BY.

**Question 4:** When is it appropriate to use a HAVING clause?

  A) To restrict which rows are returned based on conditions before aggregation.
  B) To restrict which groups are returned based on conditions after aggregation.
  C) To sort the results of a query.
  D) To create new groups based on conditions.

**Correct Answer:** B
**Explanation:** The HAVING clause is specifically used to filter grouped results after aggregation.

### Activities
- Write a SQL query that retrieves the average sales amount for each product_id from the sales table and filters out products with an average sales amount less than $150.
- Create a dataset and apply GROUP BY and HAVING to demonstrate the impact of different aggregate functions on data grouping.

### Discussion Questions
- What are some practical scenarios where GROUP BY and HAVING could be particularly useful in business analytics?
- How do different aggregate functions alter the data grouping and the resulting output?
- Can you think of a case in which using HAVING might lead to performance issues? Why might that happen?

---

## Section 8: Subqueries and Nested Queries

### Learning Objectives
- Describe the purpose and functionality of subqueries.
- Implement subqueries within SQL statements.
- Differentiate between single-row and multi-row subqueries.

### Assessment Questions

**Question 1:** What is a subquery?

  A) A query that retrieves all rows
  B) A query nested inside another query
  C) A join between two tables
  D) A single SQL statement

**Correct Answer:** B
**Explanation:** A subquery is a query nested inside another query.

**Question 2:** Which of the following is true regarding multi-row subqueries?

  A) They can return only one column.
  B) They can only be used with the EXISTS operator.
  C) They can return multiple columns.
  D) They typically use operators like IN and ANY.

**Correct Answer:** D
**Explanation:** Multi-row subqueries can return multiple rows and are typically used with operators such as IN and ANY.

**Question 3:** What must surround a subquery in SQL?

  A) Brackets
  B) Quotation marks
  C) Parentheses
  D) Curly braces

**Correct Answer:** C
**Explanation:** Subqueries must be enclosed in parentheses to be properly executed.

**Question 4:** Which SQL statement demonstrates a practical use of a subquery?

  A) SELECT * FROM products;
  B) SELECT product_name FROM products WHERE price > (SELECT AVG(price) FROM products);
  C) SELECT * FROM employees JOIN departments ON employees.department_id = departments.id;
  D) SELECT COUNT(*) FROM orders;

**Correct Answer:** B
**Explanation:** Option B uses a subquery to find products priced higher than the average price of all products.

### Activities
- Write a SQL statement that retrieves the names of all employees whose salaries are greater than the average salary within their respective departments by using a subquery.

### Discussion Questions
- How might the use of subqueries affect the performance of a SQL query?
- What are scenarios where using a subquery is preferred over a join operation?
- Can subqueries be used in other programming languages or contexts outside of SQL? Discuss.

---

## Section 9: Practical Examples of Data Manipulation

### Learning Objectives
- Analyze real-world applications of SQL to manipulate and query data.
- Demonstrate the ability to construct SQL statements for various data operations.

### Assessment Questions

**Question 1:** Which SQL statement is used to add new records to a database?

  A) SELECT
  B) DELETE
  C) INSERT
  D) UPDATE

**Correct Answer:** C
**Explanation:** The INSERT statement is used to add new records into a table.

**Question 2:** What does the SELECT statement in SQL primarily do?

  A) Allows data entry
  B) Deletes records
  C) Queries data from one or more tables
  D) Updates existing records

**Correct Answer:** C
**Explanation:** The SELECT statement is used to query data from one or more tables, allowing users to retrieve specific information.

**Question 3:** In the context of financial reporting, which SQL command would you use to combine data from two tables?

  A) SELECT
  B) JOIN
  C) GROUP BY
  D) ORDER BY

**Correct Answer:** B
**Explanation:** The JOIN command is used to combine rows from two or more tables based on a related column between them.

**Question 4:** How would you find customers who have placed more than 3 orders?

  A) By using the DELETE statement
  B) By using the SELECT statement with GROUP BY and HAVING
  C) By using the INSERT statement
  D) By using the UPDATE statement

**Correct Answer:** B
**Explanation:** You would use the SELECT statement with GROUP BY to aggregate the order counts and HAVING to filter those greater than 3.

### Activities
- Create a SQL query that calculates the total number of products sold from a sales table and groups them by category.
- Develop a scenario for a fictional company's CRM system and write SQL queries to extract meaningful insights about customer behavior.

### Discussion Questions
- What challenges might an organization face when implementing SQL for data manipulation?
- Can you think of other practical applications of SQL in various industries? Provide examples.

---

## Section 10: Common Errors in SQL

### Learning Objectives
- Identify common errors encountered in SQL.
- Understand best practices for troubleshooting SQL queries.
- Apply debugging techniques to correct SQL queries.

### Assessment Questions

**Question 1:** What is a common mistake in SQL syntax?

  A) Missing commas between columns
  B) Using correct keywords
  C) Properly naming tables
  D) Formatting SQL correctly

**Correct Answer:** A
**Explanation:** Missing commas between columns is a frequent syntax error.

**Question 2:** Which of the following represents a logical error in SQL?

  A) SELECT * FROM users;
  B) SELECT * FROM orders WHERE order_date = '2022-01-01';
  C) SELECT name FROM customers;
  D) SELECT price FROM products WHERE price < 20;

**Correct Answer:** B
**Explanation:** Stating a specific date instead of a range represents a logical error if the intention was to retrieve orders from the entire month.

**Question 3:** What happens when you have ambiguous column references in your SQL query?

  A) The query runs faster
  B) The system cannot determine which column you are referring to
  C) It automatically corrects the error
  D) The query will be ignored

**Correct Answer:** B
**Explanation:** Ambiguous column references can cause the SQL engine to throw an error because it can't ascertain which table to pull the column from.

**Question 4:** What should you verify to avoid type mismatch errors in SQL?

  A) That keywords are in uppercase
  B) That values match expected data types
  C) That the database is properly indexed
  D) That column aliases are used

**Correct Answer:** B
**Explanation:** Ensuring that values in queries match the defined data types is crucial to avoid type mismatch errors.

### Activities
- Review the following SQL queries and identify the errors. Rewrite them to correct any mistakes: 1. SELECT * FROM employees WHERE id = '123'; 2. SELECT name FROM products, categories WHERE id = category_id;

### Discussion Questions
- Can you describe a situation where you encountered a logical error in a query? How did you resolve it?
- What strategies do you find most effective when debugging SQL queries?
- Why is it important to differentiate between syntax and logical errors in SQL?

---

## Section 11: Best Practices for SQL Query Optimization

### Learning Objectives
- Describe techniques for SQL query optimization.
- Apply best practices for writing efficient SQL queries.
- Analyze execution plans for performance bottlenecks.

### Assessment Questions

**Question 1:** What is a common method for optimizing SQL queries?

  A) Using SELECT *
  B) Indexing columns frequently queried
  C) Avoiding joins
  D) Writing complex subqueries

**Correct Answer:** B
**Explanation:** Indexing columns can significantly improve the performance of SQL queries.

**Question 2:** Which of the following statements should be avoided to improve query performance?

  A) Using SELECT with specific columns
  B) Using DISTINCT when not necessary
  C) Filtering with WHERE clauses
  D) Joining on indexed columns

**Correct Answer:** B
**Explanation:** Using DISTINCT unnecessarily can add overhead and slow down query performance.

**Question 3:** What is a benefit of using INNER JOIN over OUTER JOIN?

  A) INNER JOIN returns all records
  B) INNER JOIN performs better since it only returns matching rows
  C) OUTER JOIN does not support indexing
  D) INNER JOIN is easier to read

**Correct Answer:** B
**Explanation:** INNER JOIN generally performs better because it only returns matching rows, reducing overhead.

**Question 4:** What can be used to monitor the performance of SQL queries?

  A) SQL optimization commands
  B) Execution plans
  C) Data visualization tools
  D) Query formatting

**Correct Answer:** B
**Explanation:** Analyzing execution plans can help identify bottlenecks in SQL queries and improve performance.

### Activities
- Take a sample SQL query and identify areas where optimization can be applied. Suggest changes to improve performance.
- Use a database tool to review the execution plan for a query of your choice and summarize the findings.

### Discussion Questions
- What challenges have you encountered while optimizing SQL queries?
- How do you determine when a query needs optimization?

---

## Section 12: Conclusion: Mastering SQL for Data Manipulation

### Learning Objectives
- Summarize important concepts learned about SQL, including key commands and their functions.
- Recognize the impact of SQL on data manipulation and analysis in various industries.

### Assessment Questions

**Question 1:** Which SQL command is used to retrieve data from a database?

  A) INSERT
  B) UPDATE
  C) SELECT
  D) DELETE

**Correct Answer:** C
**Explanation:** The SELECT statement is used to query data from a database.

**Question 2:** What is the primary role of SQL in data science?

  A) Data visualization
  B) Data storage
  C) Data manipulation and retrieval
  D) Data warehousing

**Correct Answer:** C
**Explanation:** SQL is essential for data manipulation and retrieval in relational databases, enabling data professionals to conduct analyses.

**Question 3:** Why should you avoid using SELECT * in SQL queries?

  A) It is slower as it retrieves all columns which may not be necessary.
  B) It is not allowed in SQL syntax.
  C) It limits the amount of data returned.
  D) It is the only way to retrieve data.

**Correct Answer:** A
**Explanation:** Using SELECT * can retrieve unnecessary columns and slow down query performance. It's better to specify only the columns needed.

**Question 4:** How can indexes improve SQL query performance?

  A) By storing data in a more accessible format.
  B) By reducing the need for SQL commands.
  C) By speeding up data retrieval from tables.
  D) By creating backups of the database.

**Correct Answer:** C
**Explanation:** Indexes are used to speed up the retrieval of rows from a database table, making queries run faster.

### Activities
- Develop a SQL query to analyze sales data for a specific time frame and present your findings.
- Create a database schema for a hypothetical e-commerce platform and write SQL commands to manage the data.

### Discussion Questions
- What challenges have you faced when using SQL for data manipulation, and how did you overcome them?
- In what ways do you think SQL can be integrated with other programming languages for more advanced data analysis?

---

