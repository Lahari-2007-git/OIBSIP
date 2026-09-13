# OASIS INFOBYTE INTERNSHIP
# TASK 2: CUSTOMER SEGMENTATION ANALYSIS

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ---------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------

df = pd.read_csv("Online Retail.csv")

print("First 5 rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns)

print("\nData Types:")
print(df.dtypes)


# ---------------------------------------------------------
# 2. DATA CLEANING
# ---------------------------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Remove rows without CustomerID
df = df.dropna(subset=["CustomerID"])

# Remove duplicate rows
df = df.drop_duplicates()

# Remove cancelled invoices
df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]

# Keep only positive quantity and price
df = df[df["Quantity"] > 0]
df = df[df["UnitPrice"] > 0]

# Convert CustomerID to integer
df["CustomerID"] = df["CustomerID"].astype(int)

# Convert InvoiceDate correctly
df["InvoiceDate"] = pd.to_datetime(
    df["InvoiceDate"],
    dayfirst=True
)

# Create TotalAmount
df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]

print("\nCleaned Dataset Shape:")
print(df.shape)

# ---------------------------------------------------------
# 3. BASIC BUSINESS ANALYSIS
# ---------------------------------------------------------

total_sales = df["TotalAmount"].sum()
total_orders = df["InvoiceNo"].nunique()
total_customers = df["CustomerID"].nunique()
average_order_value = total_sales / total_orders

print("\n----- BUSINESS OVERVIEW -----")
print("Total Sales:", round(total_sales, 2))
print("Total Orders:", total_orders)
print("Total Customers:", total_customers)
print("Average Order Value:", round(average_order_value, 2))


# ---------------------------------------------------------
# 4. MONTHLY SALES TREND
# ---------------------------------------------------------

monthly_sales = (
    df.set_index("InvoiceDate")
      .resample("M")["TotalAmount"]
      .sum()
)

plt.figure(figsize=(10, 5))
plt.plot(monthly_sales.index, monthly_sales.values, marker="o")
plt.title("Monthly Sales Trend")
plt.xlabel("Month")
plt.ylabel("Sales")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 5. TOP PRODUCTS
# ---------------------------------------------------------

top_products = (
    df.groupby("Description")["TotalAmount"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
)

print("\nTop 10 Products by Sales:")
print(top_products)

plt.figure(figsize=(10, 5))
top_products.sort_values().plot(kind="barh")
plt.title("Top 10 Products by Sales")
plt.xlabel("Sales")
plt.ylabel("Product")
plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 6. TOP COUNTRIES
# ---------------------------------------------------------

top_countries = (
    df.groupby("Country")["TotalAmount"]
      .sum()
      .sort_values(ascending=False)
      .head(10)
)

print("\nTop 10 Countries by Sales:")
print(top_countries)


# ---------------------------------------------------------
# 7. CREATE RFM ANALYSIS
# ---------------------------------------------------------

# Reference date
analysis_date = df["InvoiceDate"].max() + pd.Timedelta(days=1)

rfm = df.groupby("CustomerID").agg({

    "InvoiceDate": lambda x: (analysis_date - x.max()).days,

    "InvoiceNo": "nunique",

    "TotalAmount": "sum"
})


# Rename columns
rfm.columns = ["Recency", "Frequency", "Monetary"]


print("\n----- RFM DATA -----")
print(rfm.head())


# ---------------------------------------------------------
# 8. DESCRIPTIVE STATISTICS
# ---------------------------------------------------------

print("\nRFM Descriptive Statistics:")
print(rfm.describe())

print("\nAverage Purchase Value:")
print(round(rfm["Monetary"].mean(), 2))

print("\nAverage Purchase Frequency:")
print(round(rfm["Frequency"].mean(), 2))

print("\nAverage Customer Lifetime Value:")
print(round(rfm["Monetary"].mean(), 2))


# ---------------------------------------------------------
# 9. RFM DISTRIBUTIONS
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))
sns.histplot(rfm["Recency"], bins=30, kde=True)
plt.title("Recency Distribution")
plt.xlabel("Days Since Last Purchase")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(rfm["Frequency"], bins=30, kde=True)
plt.title("Frequency Distribution")
plt.xlabel("Number of Orders")
plt.show()

plt.figure(figsize=(8, 5))
sns.histplot(rfm["Monetary"], bins=30, kde=True)
plt.title("Monetary Distribution")
plt.xlabel("Total Spending")
plt.show()


# ---------------------------------------------------------
# 10. HANDLE SKEWNESS
# ---------------------------------------------------------

rfm_log = np.log1p(rfm)

print("\nLog Transformed RFM:")
print(rfm_log.head())


# ---------------------------------------------------------
# 11. STANDARDIZE FEATURES
# ---------------------------------------------------------

scaler = StandardScaler()

rfm_scaled = scaler.fit_transform(rfm_log)

print("\nScaled RFM Data:")
print(rfm_scaled[:5])


# ---------------------------------------------------------
# 12. ELBOW METHOD
# ---------------------------------------------------------

inertia = []

for k in range(2, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    kmeans.fit(rfm_scaled)

    inertia.append(kmeans.inertia_)


plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), inertia, marker="o")

plt.title("Elbow Method for Optimal K")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.xticks(range(2, 11))
plt.grid(True)

plt.show()

# ---------------------------------------------------------
# 13. SILHOUETTE SCORE
# ---------------------------------------------------------

silhouette_scores = []

for k in range(2, 11):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )

    labels = kmeans.fit_predict(rfm_scaled)

    score = silhouette_score(rfm_scaled, labels)

    silhouette_scores.append(score)


print("\nSilhouette Scores:")

for k, score in zip(range(2, 11), silhouette_scores):

    print("K =", k, "Score =", round(score, 3))


plt.figure(figsize=(8, 5))
plt.plot(range(2, 11), silhouette_scores, marker="o")
plt.title("Silhouette Score")
plt.xlabel("Number of Clusters")
plt.ylabel("Silhouette Score")
plt.xticks(range(2, 11))
plt.grid(True)
plt.show()


# ---------------------------------------------------------
# 14. APPLY K-MEANS
# ---------------------------------------------------------

# Start with 4 clusters
optimal_k = 4

kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)


print("\nCluster Assignment:")
print(rfm.head())


# ---------------------------------------------------------
# 15. NUMBER OF CUSTOMERS IN EACH CLUSTER
# ---------------------------------------------------------

cluster_counts = rfm["Cluster"].value_counts().sort_index()

print("\nCustomers in Each Cluster:")
print(cluster_counts)


plt.figure(figsize=(8, 5))
cluster_counts.plot(kind="bar")

plt.title("Number of Customers in Each Cluster")
plt.xlabel("Cluster")
plt.ylabel("Number of Customers")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 16. CLUSTER PROFILE
# ---------------------------------------------------------

cluster_profile = rfm.groupby("Cluster")[[
    "Recency",
    "Frequency",
    "Monetary"
]].mean()

cluster_profile["Customer_Count"] = rfm.groupby("Cluster").size()

cluster_profile["Percentage"] = (
    cluster_profile["Customer_Count"]
    / len(rfm) * 100
)

print("\n----- CLUSTER PROFILE -----")
print(cluster_profile.round(2))


# ---------------------------------------------------------
# 17. HEATMAP
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.heatmap(
    cluster_profile[["Recency", "Frequency", "Monetary"]],
    annot=True,
    fmt=".1f"
)

plt.title("Cluster Profile")
plt.xlabel("RFM Features")
plt.ylabel("Cluster")

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 18. SCATTER PLOT 1
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=rfm,
    x="Recency",
    y="Frequency",
    hue="Cluster",
    palette="Set2"
)

plt.title("Recency vs Frequency")
plt.xlabel("Recency")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 19. SCATTER PLOT 2
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=rfm,
    x="Frequency",
    y="Monetary",
    hue="Cluster",
    palette="Set2"
)

plt.title("Frequency vs Monetary")
plt.xlabel("Frequency")
plt.ylabel("Monetary")

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 20. SCATTER PLOT 3
# ---------------------------------------------------------

plt.figure(figsize=(8, 5))

sns.scatterplot(
    data=rfm,
    x="Recency",
    y="Monetary",
    hue="Cluster",
    palette="Set2"
)

plt.title("Recency vs Monetary")
plt.xlabel("Recency")
plt.ylabel("Monetary")

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 21. SORT CLUSTERS BY MONETARY VALUE
# ---------------------------------------------------------

sorted_clusters = cluster_profile.sort_values(
    "Monetary",
    ascending=False
)

print("\nClusters Sorted by Customer Value:")
print(sorted_clusters.round(2))


# ---------------------------------------------------------
# 22. CUSTOMER SEGMENT INTERPRETATION
# ---------------------------------------------------------

median_recency = rfm["Recency"].median()
median_frequency = rfm["Frequency"].median()
median_monetary = rfm["Monetary"].median()


def identify_segment(row):

    if (
        row["Recency"] <= median_recency
        and row["Frequency"] >= median_frequency
        and row["Monetary"] >= median_monetary
    ):
        return "High Value Customers"

    elif (
        row["Recency"] <= median_recency
        and row["Frequency"] >= median_frequency
    ):
        return "Loyal Customers"

    elif row["Recency"] > median_recency:
        return "At Risk Customers"

    else:
        return "New / Low Value Customers"


rfm["Customer_Type"] = rfm.apply(
    identify_segment,
    axis=1
)


print("\nCustomer Segment Distribution:")
print(rfm["Customer_Type"].value_counts())


# ---------------------------------------------------------
# 23. CUSTOMER TYPE BAR CHART
# ---------------------------------------------------------

customer_types = rfm["Customer_Type"].value_counts()

plt.figure(figsize=(9, 5))

customer_types.plot(kind="bar")

plt.title("Customer Segments")
plt.xlabel("Customer Type")
plt.ylabel("Number of Customers")
plt.xticks(rotation=20)

plt.tight_layout()
plt.show()


# ---------------------------------------------------------
# 24. MARKETING RECOMMENDATIONS
# ---------------------------------------------------------

print("\n----- MARKETING RECOMMENDATIONS -----")

print("""
1. High Value Customers:
   - Provide exclusive offers and loyalty rewards.
   - Encourage premium and repeat purchases.

2. Loyal Customers:
   - Give loyalty points and personalized recommendations.
   - Introduce referral programs.

3. At Risk Customers:
   - Send re-engagement emails and special discounts.
   - Offer limited-time promotions to bring them back.

4. New / Low Value Customers:
   - Provide welcome offers.
   - Recommend popular and affordable products.
""")


# ---------------------------------------------------------
# 25. FINAL CLUSTER SUMMARY
# ---------------------------------------------------------

final_summary = rfm.groupby("Customer_Type").agg({
    "Recency": "mean",
    "Frequency": "mean",
    "Monetary": "mean"
})

# Add customer count separately
customer_count = rfm["Customer_Type"].value_counts()

final_summary["Customer_Count"] = customer_count

print("\n----- FINAL CUSTOMER SEGMENT SUMMARY -----")
print(final_summary.round(2))


# ---------------------------------------------------------
# 26. SAVE CUSTOMER SEGMENTS
# ---------------------------------------------------------

rfm.to_csv(
    "customer_segments.csv",
    index=True
)

print("\nCustomer segmentation file saved as:")
print("customer_segments.csv")


# ---------------------------------------------------------
# 27. FINAL PROJECT INSIGHTS
# ---------------------------------------------------------

print("""
=========================================================
CUSTOMER SEGMENTATION ANALYSIS - FINAL INSIGHTS
=========================================================

• RFM analysis was used to understand customer behaviour.

• K-Means clustering divided customers into distinct groups.

• Recency shows how recently customers purchased.

• Frequency shows how often customers purchased.

• Monetary shows how much customers spent.

• High-value customers should receive loyalty rewards.

• Loyal customers can be targeted with personalized offers.

• At-risk customers should receive re-engagement campaigns.

• New or low-value customers can be encouraged with welcome
  offers and product recommendations.

""")
