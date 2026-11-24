# Week 5: Generative AI for Data Analysis

## 1. Introduction

During Week 5 of the training program **“Next-Gen Data Exploration with ML and Generative AI,”** we explored how Generative Artificial Intelligence (GenAI) can be used to simplify and improve data analysis workflows.

Generative AI helps users interact with data using natural language, generate SQL queries and Python code, understand complex datasets, and automate analytical tasks. It makes data analysis more accessible by reducing the need to write every query or program manually.

This week focused on understanding the role of Generative AI in data workflows, prompt engineering, converting natural language into SQL/Python, and using AI-assisted analytics.

## 2. Objectives

The main objectives of Week 5 were:

- Understand the role of Generative AI in data analysis.
- Explore how GenAI can improve data workflows.
- Learn the fundamentals of prompt engineering.
- Understand how natural language can be converted into SQL queries.
- Explore the generation of Python code using natural language instructions.
- Learn how AI-assisted analytics can support data exploration and decision-making.

## 3. Topics Covered

### 3.1 Generative AI in Data Workflows

Generative AI is a type of artificial intelligence that can generate new content, such as text, code, summaries, and analytical explanations, based on user instructions.

In data analysis, Generative AI can assist with several tasks:

1. **Data Understanding:** Explaining dataset columns, data types, and relationships.
2. **Data Cleaning:** Suggesting methods to handle missing values and duplicate records.
3. **Code Generation:** Generating Python code for data processing and analysis.
4. **Query Generation:** Creating SQL queries from natural language instructions.
5. **Data Summarization:** Generating meaningful summaries from analytical results.
6. **Visualization Assistance:** Suggesting suitable charts and generating plotting code.

**Example:**

A user can ask:

"Analyze the sales dataset and identify the top five products based on total revenue."

A Generative AI tool can help generate the Python code or SQL query required to perform this analysis.

### 3.2 Prompt Engineering

Prompt engineering is the process of designing clear and effective instructions (prompts) to obtain accurate and useful responses from Generative AI models.

A prompt describes the task that the AI model needs to perform.

**Importance of Prompt Engineering:**

- Improves the quality of AI-generated responses.
- Helps obtain more relevant and accurate results.
- Reduces ambiguity in instructions.
- Helps generate structured outputs.
- Makes AI tools more effective for data analysis.

**Elements of a Good Prompt:**

1. **Role:** Define the role of the AI model.
2. **Task:** Clearly describe what needs to be done.
3. **Context:** Provide relevant information about the dataset or problem.
4. **Constraints:** Mention specific requirements or limitations.
5. **Output Format:** Specify how the response should be presented.

**Example of a Basic Prompt:**

"Analyze this dataset."

**Improved Prompt:**

"Act as a data analyst. Analyze the uploaded sales dataset using Python. Identify missing values, calculate total revenue, find the top five products, and summarize the findings in simple language."

The improved prompt provides more specific instructions and helps generate a more useful response.

### 3.3 Natural Language to SQL (NL to SQL)

Natural Language to SQL refers to the process of converting human-language questions into SQL queries.

It allows users to retrieve information from databases without manually writing SQL statements.

**Example:**

Natural Language Question:

"Display all students who scored more than 80 marks."

Generated SQL Query:

```sql
SELECT *
FROM students
WHERE marks > 80;
```

**Another Example:**

Natural Language Question:

"Find the average salary of employees."

Generated SQL Query:

```sql
SELECT AVG(salary)
FROM employees;
```

**Applications of NL to SQL:**

- Retrieving records from databases.
- Generating reports.
- Filtering and sorting data.
- Performing calculations and aggregations.
- Making database analysis easier for non-technical users.

### 3.4 Natural Language to Python

Generative AI can convert natural language instructions into Python code.

This helps users perform data analysis without writing every line of code manually.

**Example:**

Natural Language Instruction:

"Create a bar chart showing the sales of different products."

AI-generated Python Code:

```python
import matplotlib.pyplot as plt

products = ["Laptop", "Mobile", "Tablet", "Headphones"]
sales = [50000, 75000, 30000, 20000]

plt.bar(products, sales)
plt.xlabel("Products")
plt.ylabel("Sales")
plt.title("Product Sales Analysis")
plt.show()
```

**Applications:**

- Data cleaning and preprocessing.
- Exploratory Data Analysis (EDA).
- Statistical calculations.
- Data visualization.
- Machine learning implementation.
- Generating analytical reports.

### 3.5 AI-Assisted Analytics

AI-assisted analytics refers to the use of artificial intelligence tools to support data exploration, analysis, interpretation, and decision-making.

It helps analysts understand datasets and discover useful insights more efficiently.

**Major Applications:**

1. **Automated Data Analysis:** AI can assist in identifying patterns and trends.
2. **Data Visualization:** AI can suggest suitable charts and graphs.
3. **Predictive Insights:** AI can help interpret predictions generated by machine learning models.
4. **Report Generation:** AI can summarize analytical findings.
5. **Anomaly Detection:** AI can assist in identifying unusual observations.
6. **Decision Support:** AI-generated insights can help users make informed decisions.

**Example:**

A company can use AI-assisted analytics to examine monthly sales data, identify high-performing products, detect sales declines, and generate a summary of business performance.

## 4. Practical Implementation

### Practical 1: Using Generative AI for Data Analysis

**Objective:** To understand how Generative AI can assist in analyzing a dataset.

**Prompt:**

"Act as a data analyst. Explain the structure of a dataset, identify missing values, suggest suitable cleaning methods, and provide a summary of important observations."

### Practical 2: Generating SQL Queries Using Natural Language

**Objective:** To convert natural language questions into SQL queries.

**Input:**

"Find all employees whose salary is greater than 50,000."

**Generated SQL:**

```sql
SELECT *
FROM employees
WHERE salary > 50000;
```

### Practical 3: Generating Python Code Using AI

**Objective:** To generate Python code for basic data analysis.

**Prompt:**

"Write Python code using Pandas to load a CSV file, display the first five rows, check missing values, and calculate basic statistical information."

**Generated Python Code:**

```python
import pandas as pd

# Load dataset
df = pd.read_csv("data.csv")

# Display first five rows
print(df.head())

# Check missing values
print(df.isnull().sum())

# Display statistical summary
print(df.describe())
```

## 5. Hands-on Practice Tasks

The following tasks can be performed to strengthen the understanding of Generative AI in data analysis:

1. Use a Generative AI tool to explain a sample dataset.
2. Write effective prompts for data cleaning and data analysis.
3. Convert at least five natural language questions into SQL queries.
4. Generate Python code for data preprocessing using AI.
5. Create a visualization using AI-generated Python code.
6. Ask an AI tool to summarize the findings of a dataset.
7. Compare AI-generated code with manually written code and verify the results.

## 6. Learning Outcomes

After completing Week 5, I gained an understanding of:

- The role of Generative AI in modern data workflows.
- The importance of prompt engineering.
- Converting natural language instructions into SQL queries.
- Generating Python code using AI tools.
- Using AI-assisted analytics for data exploration.
- Applying Generative AI to simplify analytical tasks.

## 7. Key Takeaways

- Generative AI makes data analysis more accessible and efficient.
- Prompt engineering plays an important role in obtaining useful AI responses.
- Natural Language to SQL allows users to interact with databases using simple questions.
- AI-generated Python code can reduce repetitive coding work.
- AI-assisted analytics supports data interpretation and decision-making.
- AI-generated outputs should always be checked and validated before use.

## 8. Conclusion

Week 5 provided practical knowledge of how Generative AI can be integrated into data analysis workflows. We explored prompt engineering, natural language to SQL/Python, and AI-assisted analytics.

The concepts learned during this week demonstrated how AI tools can simplify complex analytical tasks, improve productivity, and make data-driven insights easier to understand. This knowledge provides a foundation for exploring advanced machine learning and Generative AI systems in the next week.
