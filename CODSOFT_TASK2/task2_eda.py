import pandas as pd

# Load Dataset
file_path = r"D:\CODSOFT INTERNSHIP\CODSOFT_TASK2\retail_sales_eda.csv"

df = pd.read_csv(file_path)

print("Dataset loaded successfully!\n")

# Dataset Shape
print("Dataset Shape:")
print(df.shape)

# Column Names
print("\nColumn Names:")
print(df.columns.tolist())

# First 5 Rows
print("\nFirst 5 Rows:")
print(df.head())

# Data Types
print("\nData Types:")
print(df.dtypes)

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate Records
print("\nDuplicate Records:")
print(df.duplicated().sum())


# Step 2: Descriptive Statistics

print("\nDescriptive Statistics:")
print(df.describe())

print("\nNumerical Columns:")
print(df.select_dtypes(include="number").columns.tolist())

print("\nCategorical Columns:")
print(df.select_dtypes(include="object").columns.tolist())


# Step 3: Trends, Distributions and Relationships

# Convert Order_Date to datetime
df["Order_Date"] = pd.to_datetime(df["Order_Date"])

print("\nDate Conversion Completed!")

# Sales by Region
print("\nTotal Sales by Region:")
print(df.groupby("Region")["Sales"].sum().sort_values(ascending=False))

# Sales by Category
print("\nTotal Sales by Category:")
print(df.groupby("Category")["Sales"].sum().sort_values(ascending=False))

# Sales by Product
print("\nTotal Sales by Product:")
print(df.groupby("Product")["Sales"].sum().sort_values(ascending=False))

# Profit by Category
print("\nTotal Profit by Category:")
print(df.groupby("Category")["Profit"].sum().sort_values(ascending=False))

# Customer Type Analysis
print("\nSales by Customer Type:")
print(df.groupby("Customer_Type")["Sales"].sum().sort_values(ascending=False))

# Correlation Matrix
print("\nCorrelation Matrix:")
print(df[["Sales", "Quantity", "Discount", "Profit"]].corr())



# Step 4: Outlier Detection using IQR Method

numerical_columns = ["Sales", "Quantity", "Discount", "Profit"]

print("\nOutlier Detection:")
print("-" * 50)

for column in numerical_columns:
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[
        (df[column] < lower_bound) |
        (df[column] > upper_bound)
    ]

    print(f"\nColumn: {column}")
    print(f"Q1: {Q1:.2f}")
    print(f"Q3: {Q3:.2f}")
    print(f"IQR: {IQR:.2f}")
    print(f"Lower Bound: {lower_bound:.2f}")
    print(f"Upper Bound: {upper_bound:.2f}")
    print(f"Number of Outliers: {len(outliers)}")

print("\nOutlier Detection Completed Successfully!")




# Step 5: Box Plot for Outlier Visualization

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))

df[["Sales", "Quantity", "Discount", "Profit"]].boxplot()

plt.title("Box Plot of Numerical Variables")
plt.ylabel("Values")
plt.grid(True)

plt.savefig("box_plot_numerical_variables.png")
plt.show()

print("\nBox Plot created successfully!")


# Step 6: Business Questions and Answers

print("\n" + "=" * 60)
print("BUSINESS QUESTIONS & ANSWERS")
print("=" * 60)

# Question 1
best_region = df.groupby("Region")["Sales"].sum().idxmax()
best_region_sales = df.groupby("Region")["Sales"].sum().max()

print("\nQ1. Which region generated the highest sales?")
print(f"Answer: {best_region} generated the highest sales of {best_region_sales:.2f}")

# Question 2
best_category = df.groupby("Category")["Sales"].sum().idxmax()
best_category_sales = df.groupby("Category")["Sales"].sum().max()

print("\nQ2. Which product category generated the highest sales?")
print(f"Answer: {best_category} generated the highest sales of {best_category_sales:.2f}")

# Question 3
best_product = df.groupby("Product")["Sales"].sum().idxmax()
best_product_sales = df.groupby("Product")["Sales"].sum().max()

print("\nQ3. Which product generated the highest sales?")
print(f"Answer: {best_product} generated the highest sales of {best_product_sales:.2f}")

# Question 4
best_customer = df.groupby("Customer_Type")["Sales"].sum().idxmax()
best_customer_sales = df.groupby("Customer_Type")["Sales"].sum().max()

print("\nQ4. Which customer type generated the highest sales?")
print(f"Answer: {best_customer} customers generated the highest sales of {best_customer_sales:.2f}")

# Question 5
best_profit_category = df.groupby("Category")["Profit"].sum().idxmax()
best_profit = df.groupby("Category")["Profit"].sum().max()

print("\nQ5. Which category generated the highest total profit?")
print(f"Answer: {best_profit_category} generated the highest profit of {best_profit:.2f}")

# Question 6
average_sales = df["Sales"].mean()

print("\nQ6. What is the average sales value per order?")
print(f"Answer: The average sales per order is {average_sales:.2f}")

print("\nBusiness Questions Analysis Completed Successfully!")



# Step 7: EDA Findings Report

total_sales = df["Sales"].sum()
total_profit = df["Profit"].sum()
average_sales = df["Sales"].mean()

best_region = df.groupby("Region")["Sales"].sum().idxmax()
best_category = df.groupby("Category")["Sales"].sum().idxmax()
best_product = df.groupby("Product")["Sales"].sum().idxmax()
best_customer = df.groupby("Customer_Type")["Sales"].sum().idxmax()
best_profit_category = df.groupby("Category")["Profit"].sum().idxmax()

report = f"""
CODSOFT TASK 2 - EDA FINDINGS REPORT
====================================

Dataset Overview
----------------
Total Records: {len(df)}
Total Columns: {len(df.columns)}
Total Sales: {total_sales:.2f}
Total Profit: {total_profit:.2f}
Average Sales per Order: {average_sales:.2f}

Key Findings
------------
1. West region generated the highest sales:
   {df.groupby("Region")["Sales"].sum().max():.2f}

2. Technology was the highest-selling category:
   {df.groupby("Category")["Sales"].sum().max():.2f}

3. Phone was the highest-selling product:
   {df.groupby("Product")["Sales"].sum().max():.2f}

4. Corporate customers generated the highest sales:
   {df.groupby("Customer_Type")["Sales"].sum().max():.2f}

5. Office Supplies generated the highest total profit:
   {df.groupby("Category")["Profit"].sum().max():.2f}

6. Average sales value per order:
   {average_sales:.2f}

Outlier Findings
----------------
- Sales contained 2 outliers.
- Profit contained 2 outliers.
- Quantity contained 0 outliers.
- Discount contained 0 outliers.

Correlation Findings
--------------------
- Sales and Profit correlation: {df["Sales"].corr(df["Profit"]):.4f}
- Sales and Quantity correlation: {df["Sales"].corr(df["Quantity"]):.4f}
- Sales and Discount correlation: {df["Sales"].corr(df["Discount"]):.4f}

Conclusion
----------
The EDA identified important patterns in sales, profit, products,
regions, and customer types. West was the strongest region,
Technology was the leading category, and Corporate customers
generated the highest sales.

Note:
The dataset is a synthetic dataset created for EDA practice.
Some unusual values were intentionally included to demonstrate
outlier detection.
"""

with open("eda_findings_report.txt", "w") as file:
    file.write(report)

print("\nEDA Findings Report created successfully!")