# ============================================================
# OIBSIP DATA ANALYTICS - LEVEL 2 TASK 3
# DATA CLEANING
# Dataset: Dirty Retail Store Sales
# ============================================================


# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv("retail_store_sales.csv")

print("Dataset loaded successfully!")
print("Shape:", df.shape)


# ============================================================
# 3. INITIAL DATA INSPECTION
# ============================================================

print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nBasic information:")
df.info()


# ============================================================
# 4. DATA QUALITY REPORT - BEFORE CLEANING
# ============================================================

print("\n" + "="*60)
print("DATA QUALITY REPORT - BEFORE CLEANING")
print("="*60)

# Missing values
missing_values = df.isnull().sum()

print("\nMissing values per column:")
print(missing_values)

# Missing percentage
missing_percentage = (df.isnull().sum() / len(df)) * 100

print("\nMissing percentage per column:")
print(missing_percentage.round(2))

# Duplicate rows
duplicate_count = df.duplicated().sum()

print("\nDuplicate rows:", duplicate_count)

# Numerical summary
print("\nNumerical summary:")
print(df.describe())


# ============================================================
# 5. STORE BEFORE-CLEANING STATISTICS
# ============================================================

before_rows = len(df)
before_duplicates = df.duplicated().sum()
before_missing = df.isnull().sum().sum()

before_stats = {
    "Rows": before_rows,
    "Duplicate Rows": before_duplicates,
    "Total Missing Values": before_missing
}

print("\nBefore-cleaning statistics:")
print(before_stats)


# ============================================================
# 6. CHECK UNIQUE VALUES IN CATEGORICAL COLUMNS
# ============================================================

categorical_columns = [
    "Category",
    "Item",
    "Payment Method",
    "Location",
    "Discount Applied"
]

for column in categorical_columns:
    print("\n" + "-"*50)
    print(column)
    print("-"*50)
    print(df[column].value_counts(dropna=False).head(20))


# ============================================================
# 7. CHECK NUMERICAL VALUE RANGES
# ============================================================

numeric_columns = [
    "Price Per Unit",
    "Quantity",
    "Total Spent"
]

print("\n" + "="*60)
print("NUMERICAL VALUE RANGE CHECK")
print("="*60)

for column in numeric_columns:
    print(f"\n{column}")
    print("Minimum:", df[column].min())
    print("Maximum:", df[column].max())
    print("Mean:", round(df[column].mean(), 2))
    print("Median:", df[column].median())


# ============================================================
# 8. HANDLE MISSING VALUES
# ============================================================

print("\n" + "="*60)
print("HANDLING MISSING VALUES")
print("="*60)

# ------------------------------------------------------------
# Item
# ------------------------------------------------------------
# Item is categorical. Missing item names cannot be safely
# guessed, so they are replaced with "Unknown".

df["Item"] = df["Item"].fillna("Unknown")


# ------------------------------------------------------------
# Price Per Unit
# ------------------------------------------------------------
# Price is numerical and may contain extreme values.
# Median is used because it is less affected by outliers.

df["Price Per Unit"] = df["Price Per Unit"].fillna(
    df["Price Per Unit"].median()
)


# ------------------------------------------------------------
# Quantity
# ------------------------------------------------------------
# Quantity is numerical, so median is used.

df["Quantity"] = df["Quantity"].fillna(
    df["Quantity"].median()
)


# ------------------------------------------------------------
# Total Spent
# ------------------------------------------------------------
# Total Spent should logically equal:
# Quantity × Price Per Unit
#
# Therefore, instead of blindly filling with the median,
# we calculate it from the cleaned transaction values.

df["Total Spent"] = df["Quantity"] * df["Price Per Unit"]


# ------------------------------------------------------------
# Discount Applied
# ------------------------------------------------------------
# This is a Boolean/categorical column.
# Missing values are replaced using the most frequent value.

discount_mode = df["Discount Applied"].mode()[0]

df["Discount Applied"] = df["Discount Applied"].fillna(
    discount_mode
)


print("\nMissing values after handling:")
print(df.isnull().sum())


# ============================================================
# 9. REMOVE DUPLICATE ROWS
# ============================================================

duplicates_before = df.duplicated().sum()

df = df.drop_duplicates()

duplicates_after = df.duplicated().sum()

print("\nDuplicate rows before removal:", duplicates_before)
print("Duplicate rows after removal:", duplicates_after)


# ============================================================
# 10. STANDARDIZE TEXT DATA
# ============================================================

print("\n" + "="*60)
print("STANDARDIZING TEXT DATA")
print("="*60)

# Remove unnecessary leading/trailing spaces
text_columns = [
    "Transaction ID",
    "Customer ID",
    "Category",
    "Item",
    "Payment Method",
    "Location"
]

for column in text_columns:
    df[column] = df[column].astype(str).str.strip()


# Standardize common text formatting
df["Category"] = df["Category"].str.title()
df["Payment Method"] = df["Payment Method"].str.title()
df["Location"] = df["Location"].str.title()


# Standardize Item formatting
df["Item"] = df["Item"].str.upper()


print("\nText standardization completed.")


# ============================================================
# 11. CONVERT TRANSACTION DATE TO DATETIME
# ============================================================

print("\n" + "="*60)
print("DATE STANDARDIZATION")
print("="*60)

df["Transaction Date"] = pd.to_datetime(
    df["Transaction Date"],
    errors="coerce"
)

print("\nTransaction Date datatype:")
print(df["Transaction Date"].dtype)

print("\nDate range:")
print("Minimum date:", df["Transaction Date"].min())
print("Maximum date:", df["Transaction Date"].max())


# ============================================================
# 12. HANDLE INVALID DATES
# ============================================================

invalid_dates = df["Transaction Date"].isnull().sum()

print("\nInvalid dates found:", invalid_dates)

# Since invalid dates cannot be reliably reconstructed,
# remove only rows where date conversion failed.

if invalid_dates > 0:
    df = df.dropna(subset=["Transaction Date"])

print("Rows after invalid date handling:", len(df))


# ============================================================
# 13. CORRECT DATA TYPES
# ============================================================

print("\n" + "="*60)
print("DATA TYPE CORRECTION")
print("="*60)

# IDs should be strings
df["Transaction ID"] = df["Transaction ID"].astype(str)
df["Customer ID"] = df["Customer ID"].astype(str)

# Numeric columns
df["Price Per Unit"] = pd.to_numeric(
    df["Price Per Unit"],
    errors="coerce"
)

df["Quantity"] = pd.to_numeric(
    df["Quantity"],
    errors="coerce"
)

df["Total Spent"] = pd.to_numeric(
    df["Total Spent"],
    errors="coerce"
)

# Discount as boolean
df["Discount Applied"] = (
    df["Discount Applied"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({
        "true": True,
        "false": False
    })
)

# Fill any values that could not be converted
df["Discount Applied"] = df["Discount Applied"].fillna(
    df["Discount Applied"].mode()[0]
)

print("\nCorrected data types:")
print(df.dtypes)


# ============================================================
# 14. CHECK LOGICAL DATA ERRORS
# ============================================================

print("\n" + "="*60)
print("LOGICAL DATA VALIDATION")
print("="*60)

# Quantity should be positive
invalid_quantity = (df["Quantity"] <= 0).sum()

print("Invalid quantity values:", invalid_quantity)

# Price should be positive
invalid_price = (df["Price Per Unit"] <= 0).sum()

print("Invalid price values:", invalid_price)

# Total Spent should be positive
invalid_total = (df["Total Spent"] <= 0).sum()

print("Invalid total spent values:", invalid_total)


# Remove invalid records if present
df = df[df["Quantity"] > 0]
df = df[df["Price Per Unit"] > 0]
df = df[df["Total Spent"] > 0]


# ============================================================
# 15. VALIDATE TOTAL SPENT
# ============================================================

print("\n" + "="*60)
print("TOTAL SPENT VALIDATION")
print("="*60)

# Expected total
df["Calculated Total"] = (
    df["Quantity"] * df["Price Per Unit"]
)

# Difference
df["Total Difference"] = (
    df["Total Spent"] - df["Calculated Total"]
)

mismatch_count = (
    df["Total Difference"].abs() > 0.01
).sum()

print("Transactions with Total Spent mismatch:",
      mismatch_count)


# Since Total Spent should be based on Quantity × Price,
# update it using the calculated value.

df["Total Spent"] = df["Calculated Total"]

# Remove temporary columns
df.drop(
    columns=["Calculated Total", "Total Difference"],
    inplace=True
)

print("Total Spent validation completed.")


# ============================================================
# 16. OUTLIER DETECTION USING IQR
# ============================================================

print("\n" + "="*60)
print("OUTLIER DETECTION USING IQR")
print("="*60)

def detect_outliers_iqr(data, column):
    
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    
    IQR = Q3 - Q1
    
    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR
    
    outliers = data[
        (data[column] < lower_limit) |
        (data[column] > upper_limit)
    ]
    
    print(f"\nColumn: {column}")
    print("Q1:", Q1)
    print("Q3:", Q3)
    print("IQR:", IQR)
    print("Lower limit:", lower_limit)
    print("Upper limit:", upper_limit)
    print("Number of outliers:", len(outliers))
    
    return lower_limit, upper_limit, outliers


outlier_limits = {}

for column in numeric_columns:
    
    lower, upper, outliers = detect_outliers_iqr(
        df,
        column
    )
    
    outlier_limits[column] = (lower, upper)


# ============================================================
# 17. VISUALIZE OUTLIERS BEFORE HANDLING
# ============================================================

for column in numeric_columns:
    
    plt.figure(figsize=(8, 5))
    
    plt.boxplot(df[column])
    
    plt.title(f"Boxplot of {column}")
    plt.ylabel(column)
    
    plt.show()


# ============================================================
# 18. HANDLE OUTLIERS
# ============================================================

print("\n" + "="*60)
print("OUTLIER HANDLING")
print("="*60)

# We will cap extreme Total Spent values using the IQR limits
# rather than deleting valid customer transactions.
#
# Quantity and Price Per Unit have no meaningful IQR outliers
# in this dataset.

for column in numeric_columns:
    
    lower, upper = outlier_limits[column]
    
    outlier_count = (
        (df[column] < lower) |
        (df[column] > upper)
    ).sum()
    
    if outlier_count > 0:
        
        print(
            f"{column}: {outlier_count} outliers detected."
        )
        
        # Cap instead of deleting
        df[column] = df[column].clip(
            lower=lower,
            upper=upper
        )
        
    else:
        
        print(
            f"{column}: No IQR outliers detected."
        )


# ============================================================
# 19. FINAL MISSING VALUE CHECK
# ============================================================

print("\n" + "="*60)
print("FINAL MISSING VALUE CHECK")
print("="*60)

print(df.isnull().sum())


# ============================================================
# 20. FINAL DUPLICATE CHECK
# ============================================================

print("\nFinal duplicate rows:")
print(df.duplicated().sum())


# ============================================================
# 21. BEFORE VS AFTER CLEANING SUMMARY
# ============================================================

after_rows = len(df)
after_duplicates = df.duplicated().sum()
after_missing = df.isnull().sum().sum()

comparison = pd.DataFrame({
    "Metric": [
        "Number of Rows",
        "Duplicate Rows",
        "Total Missing Values"
    ],
    
    "Before Cleaning": [
        before_rows,
        before_duplicates,
        before_missing
    ],
    
    "After Cleaning": [
        after_rows,
        after_duplicates,
        after_missing
    ]
})

print("\n" + "="*60)
print("BEFORE VS AFTER CLEANING")
print("="*60)

print(comparison)


# ============================================================
# 22. DATA TYPE SUMMARY
# ============================================================

dtype_summary = pd.DataFrame({
    "Column": df.columns,
    "Data Type": df.dtypes.astype(str).values
})

print("\n" + "="*60)
print("FINAL DATA TYPES")
print("="*60)

print(dtype_summary)


# ============================================================
# 23. FINAL DATASET INFORMATION
# ============================================================

print("\n" + "="*60)
print("FINAL DATASET INFORMATION")
print("="*60)

print("Rows:", df.shape[0])
print("Columns:", df.shape[1])

print("\nColumns:")
print(df.columns.tolist())

print("\nData types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())


# ============================================================
# 24. FINAL DESCRIPTIVE STATISTICS
# ============================================================

print("\n" + "="*60)
print("FINAL DESCRIPTIVE STATISTICS")
print("="*60)

print(df.describe())


# ============================================================
# 25. CLEANED DATA PREVIEW
# ============================================================

print("\nFirst 10 rows of cleaned dataset:")
print(df.head(10))


# ============================================================
# 26. FINAL VALIDATION CHECKS
# ============================================================

print("\n" + "="*60)
print("FINAL VALIDATION")
print("="*60)

# Check for missing values
assert df.isnull().sum().sum() == 0

# Check for duplicates
assert df.duplicated().sum() == 0

# Check positive quantity
assert (df["Quantity"] > 0).all()

# Check positive price
assert (df["Price Per Unit"] > 0).all()

# Check positive total
assert (df["Total Spent"] > 0).all()

# Check date datatype
assert pd.api.types.is_datetime64_any_dtype(
    df["Transaction Date"]
)

# Check ID datatypes
assert df["Transaction ID"].dtype == "object"
assert df["Customer ID"].dtype == "object"

print("✓ No missing values")
print("✓ No duplicate rows")
print("✓ Quantity values are valid")
print("✓ Price values are valid")
print("✓ Total Spent values are valid")
print("✓ Transaction Date is datetime")
print("✓ IDs are stored as strings")

print("\nAll validation checks passed!")


# ============================================================
# 27. SAVE CLEANED DATASET
# ============================================================

output_file = "cleaned_retail_sales.csv"

df.to_csv(
    output_file,
    index=False
)

print("\nCleaned dataset saved successfully as:")
print(output_file)


# ============================================================
# 28. FINAL PROJECT SUMMARY
# ============================================================

print("\n" + "="*60)
print("DATA CLEANING PROJECT SUMMARY")
print("="*60)

print(f"Original rows: {before_rows}")
print(f"Final rows: {len(df)}")

print(f"Original missing values: {before_missing}")
print(f"Final missing values: {df.isnull().sum().sum()}")

print(f"Original duplicate rows: {before_duplicates}")
print(f"Final duplicate rows: {df.duplicated().sum()}")

print("\nCleaning operations completed:")
print("1. Dataset inspection")
print("2. Data quality report")
print("3. Missing value handling")
print("4. Duplicate removal")
print("5. Text standardization")
print("6. Date conversion")
print("7. Data type correction")
print("8. Logical validation")
print("9. Total Spent validation")
print("10. IQR-based outlier detection")
print("11. Outlier handling")
print("12. Before vs After comparison")
print("13. Final validation")
print("14. Cleaned CSV export")

print("\nPROJECT COMPLETED SUCCESSFULLY!")
