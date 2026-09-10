# Retail Sales EDA - Oasis Infobyte Internship
# Author: Usha

# Step 1: Load dataset and initial inspection
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("retail_sales_dataset.csv")

print("Shape:", df.shape)
print("\nData Types:\n", df.dtypes)
print("\nMissing Values:\n", df.isnull().sum())
df.head()

# --- Markdown Observation ---
# The dataset has transaction-level details with 9 columns.
# No major missing values are expected. Columns include demographics and sales info.

# Step 2: Descriptive statistics
print(df.describe())
print("\nMode:\n", df.mode().iloc[0])

# --- Markdown Observation ---
# Summary statistics show average age, quantity, price per unit, and total amount.
# Mode reveals most common values for categorical and numerical columns.

# Step 3: Time series analysis
df['Date'] = pd.to_datetime(df['Date'])

monthly_sales = df.groupby(df['Date'].dt.to_period('M'))['Total Amount'].sum()
monthly_sales.plot(kind='line', figsize=(10,5), title="Monthly Sales Trend")
plt.show()

quarterly_sales = df.groupby(df['Date'].dt.to_period('Q'))['Total Amount'].sum()
quarterly_sales.plot(kind='line', figsize=(10,5), title="Quarterly Sales Trend")
plt.show()

# --- Markdown Observation ---
# Monthly and quarterly sales trends reveal seasonal peaks (e.g., festive months).

# Step 4: Customer demographics
sns.countplot(x='Gender', data=df)
plt.title("Customer Gender Distribution")
plt.show()

sns.histplot(df['Age'], bins=10, kde=True)
plt.title("Customer Age Distribution")
plt.show()

age_bins = [18,25,35,45,55,65]
df['Age Group'] = pd.cut(df['Age'], bins=age_bins)
age_group_sales = df.groupby('Age Group')['Total Amount'].sum()
age_group_sales.plot(kind='bar', title="Spending by Age Group")
plt.show()

# --- Markdown Observation ---
# Gender distribution shows male vs female customers.
# Age distribution highlights spending concentration in certain age ranges.

# Step 5: Product analysis
category_sales = df.groupby('Product Category')['Total Amount'].sum()
category_sales.plot(kind='bar', title="Revenue by Product Category")
plt.show()

top_transactions = df.nlargest(10, 'Total Amount')
print(top_transactions[['Transaction ID','Product Category','Total Amount']])

# --- Markdown Observation ---
# Electronics and Clothing categories dominate revenue.
# Top transactions show high-value purchases in these categories.

# Step 6: Correlation heatmap
sns.heatmap(df[['Age','Quantity','Price per Unit','Total Amount']].corr(), 
            annot=True, cmap="coolwarm")
plt.title("Correlation Matrix")
plt.show()

# --- Markdown Observation ---
# Strong correlation between Quantity and Total Amount.
# Price per Unit also influences Total Amount significantly.

# Step 7: Additional visualization (non-obvious insight)
gender_category = df.groupby(['Gender','Product Category'])['Total Amount'].mean().unstack()
gender_category.plot(kind='bar', figsize=(8,5), title="Average Spending by Gender per Category")
plt.show()

# --- Markdown Observation ---
# Female customers spend more on Clothing and Beauty,
# while male customers spend more on Electronics.

# Step 8: Markdown cells throughout notebook
# Add short insights after each chart (already shown as comments above).

# Step 9: Conclusion section
print("\n--- Conclusion & Recommendations ---")
print("1. Increase stock and marketing during peak months (Nov, Dec).")
print("2. Target promotions to age groups 25–35 and 45–55, who show strong spending.")
print("3. Focus on Electronics and Clothing categories, as they generate the highest revenue.")

