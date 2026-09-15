# ============================================================
# OIBSIP - TASK 3: FRAUD DETECTION
# Dataset: creditcard.csv
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

from imblearn.over_sampling import SMOTE

import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 2. LOAD DATASET
# ============================================================

df = pd.read_csv("creditcard.csv")

print("Dataset Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nColumn Names:")
print(df.columns.tolist())

print("\nDataset Information:")
print(df.info())


# ============================================================
# 3. CHECK MISSING VALUES AND DUPLICATES
# ============================================================

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())


# ============================================================
# 4. BASIC STATISTICS
# ============================================================

print("\nStatistical Summary:")
print(df.describe())


# ============================================================
# 5. ANALYZE CLASS IMBALANCE
# ============================================================

class_counts = df["Class"].value_counts()

print("\nClass Distribution:")
print(class_counts)

fraud_count = class_counts[1]
legitimate_count = class_counts[0]
total_transactions = len(df)

fraud_percentage = (fraud_count / total_transactions) * 100
legitimate_percentage = (legitimate_count / total_transactions) * 100

print("\nTotal Transactions:", total_transactions)
print("Legitimate Transactions:", legitimate_count)
print("Fraudulent Transactions:", fraud_count)

print(f"\nFraud Percentage: {fraud_percentage:.4f}%")
print(f"Legitimate Percentage: {legitimate_percentage:.4f}%")


# ============================================================
# 6. CLASS DISTRIBUTION VISUALIZATION
# ============================================================

plt.figure(figsize=(7, 5))

sns.countplot(x="Class", data=df)

plt.title("Fraud vs Legitimate Transactions")
plt.xlabel("Transaction Class (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Number of Transactions")

plt.show()


# ============================================================
# 7. WHY ACCURACY IS MISLEADING
# ============================================================

print("\n" + "=" * 60)
print("WHY STANDARD ACCURACY IS MISLEADING")
print("=" * 60)

print("""
The dataset is highly imbalanced.

Only a very small percentage of transactions are fraudulent.
Therefore, a model that predicts almost every transaction as
legitimate could achieve very high accuracy while failing to
detect most fraudulent transactions.

For fraud detection, Precision, Recall, F1-score and ROC-AUC/
PR-AUC are more informative than accuracy alone.

Recall is especially important when missing a fraudulent
transaction can cause financial loss.
""")


# ============================================================
# 8. TRANSACTION AMOUNT ANALYSIS
# ============================================================

print("\nAverage Transaction Amount:")
print(df.groupby("Class")["Amount"].mean())

print("\nMedian Transaction Amount:")
print(df.groupby("Class")["Amount"].median())


# Boxplot
plt.figure(figsize=(8, 5))

sns.boxplot(x="Class", y="Amount", data=df)

plt.title("Transaction Amount: Fraud vs Legitimate")
plt.xlabel("Class (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Transaction Amount")

plt.show()


# Log transformation for better visualization
df["Log_Amount"] = np.log1p(df["Amount"])

plt.figure(figsize=(8, 5))

sns.boxplot(x="Class", y="Log_Amount", data=df)

plt.title("Log-Scaled Transaction Amount: Fraud vs Legitimate")
plt.xlabel("Class (0 = Legitimate, 1 = Fraud)")
plt.ylabel("Log(Transaction Amount + 1)")

plt.show()


# ============================================================
# 9. TIME-OF-DAY ANALYSIS
# ============================================================

# Time is measured in seconds from the first transaction.
# Convert seconds into hour of day.

df["Hour"] = ((df["Time"] // 3600) % 24).astype(int)

print("\nFraud Transactions by Hour:")
print(df[df["Class"] == 1]["Hour"].value_counts().sort_index())


# Fraud transactions by hour
hourly_fraud = df.groupby("Hour")["Class"].sum()

plt.figure(figsize=(10, 5))

plt.plot(hourly_fraud.index, hourly_fraud.values, marker="o")

plt.title("Fraudulent Transactions by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Fraudulent Transactions")
plt.xticks(range(24))

plt.grid(True)

plt.show()


# ============================================================
# 10. FRAUD RATE BY HOUR
# ============================================================

hourly_stats = df.groupby("Hour")["Class"].agg(
    ["count", "sum"]
)

hourly_stats["Fraud_Rate"] = (
    hourly_stats["sum"] / hourly_stats["count"]
) * 100

print("\nFraud Rate by Hour:")
print(hourly_stats)


plt.figure(figsize=(10, 5))

plt.plot(
    hourly_stats.index,
    hourly_stats["Fraud_Rate"],
    marker="o"
)

plt.title("Fraud Rate by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Fraud Rate (%)")
plt.xticks(range(24))

plt.grid(True)

plt.show()


# ============================================================
# 11. PREPARE FEATURES AND TARGET
# ============================================================

# Remove columns created only for EDA
X = df.drop(columns=["Class", "Log_Amount", "Hour"])

y = df["Class"]

print("\nFeature Shape:", X.shape)
print("Target Shape:", y.shape)


# ============================================================
# 12. TRAIN-TEST SPLIT WITH STRATIFICATION
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Set Shape:", X_train.shape)
print("Testing Set Shape:", X_test.shape)

print("\nTraining Class Distribution:")
print(y_train.value_counts())

print("\nTesting Class Distribution:")
print(y_test.value_counts())


# ============================================================
# 13. FEATURE SCALING
# ============================================================

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ============================================================
# 14. APPLY SMOTE TO TRAINING DATA ONLY
# ============================================================

print("\nBefore SMOTE:")
print(pd.Series(y_train).value_counts())

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train_scaled,
    y_train
)

print("\nAfter SMOTE:")
print(pd.Series(y_train_smote).value_counts())


# ============================================================
# 15. LOGISTIC REGRESSION MODEL
# ============================================================

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(
    X_train_smote,
    y_train_smote
)

logistic_pred = logistic_model.predict(X_test_scaled)
logistic_prob = logistic_model.predict_proba(X_test_scaled)[:, 1]


# ============================================================
# 16. RANDOM FOREST MODEL
# ============================================================

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

random_forest_model.fit(
    X_train_smote,
    y_train_smote
)

rf_pred = random_forest_model.predict(X_test_scaled)
rf_prob = random_forest_model.predict_proba(X_test_scaled)[:, 1]


# ============================================================
# 17. MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(model_name, y_true, y_pred, y_prob):

    accuracy = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_true,
        y_prob
    )

    pr_auc = average_precision_score(
        y_true,
        y_prob
    )

    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)

    print(f"Accuracy : {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-Score : {f1:.4f}")
    print(f"ROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_true,
            y_pred,
            target_names=["Legitimate", "Fraud"],
            zero_division=0
        )
    )

    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "PR-AUC": pr_auc
    }


# ============================================================
# 18. EVALUATE BOTH MODELS
# ============================================================

logistic_results = evaluate_model(
    "Logistic Regression",
    y_test,
    logistic_pred,
    logistic_prob
)

rf_results = evaluate_model(
    "Random Forest",
    y_test,
    rf_pred,
    rf_prob
)


# ============================================================
# 19. MODEL COMPARISON
# ============================================================

results = pd.DataFrame([
    logistic_results,
    rf_results
])

print("\nModel Comparison:")
print(results.round(4))


# ============================================================
# 20. MODEL COMPARISON CHART
# ============================================================

metrics = [
    "Precision",
    "Recall",
    "F1-Score",
    "ROC-AUC",
    "PR-AUC"
]

comparison = results.set_index("Model")[metrics]

comparison.plot(
    kind="bar",
    figsize=(11, 6)
)

plt.title("Fraud Detection Model Comparison")
plt.ylabel("Score")
plt.xlabel("Model")
plt.xticks(rotation=0)
plt.ylim(0, 1.05)

plt.legend()
plt.grid(axis="y")

plt.show()


# ============================================================
# 21. CONFUSION MATRIX - LOGISTIC REGRESSION
# ============================================================

cm_logistic = confusion_matrix(
    y_test,
    logistic_pred
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm_logistic,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Legitimate", "Fraud"],
    yticklabels=["Legitimate", "Fraud"]
)

plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


# ============================================================
# 22. CONFUSION MATRIX - RANDOM FOREST
# ============================================================

cm_rf = confusion_matrix(
    y_test,
    rf_pred
)

plt.figure(figsize=(6, 5))

sns.heatmap(
    cm_rf,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["Legitimate", "Fraud"],
    yticklabels=["Legitimate", "Fraud"]
)

plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.show()


# ============================================================
# 23. ROC CURVE
# ============================================================

fpr_logistic, tpr_logistic, _ = roc_curve(
    y_test,
    logistic_prob
)

fpr_rf, tpr_rf, _ = roc_curve(
    y_test,
    rf_prob
)

auc_logistic = roc_auc_score(
    y_test,
    logistic_prob
)

auc_rf = roc_auc_score(
    y_test,
    rf_prob
)


plt.figure(figsize=(8, 6))

plt.plot(
    fpr_logistic,
    tpr_logistic,
    label=f"Logistic Regression (AUC = {auc_logistic:.4f})"
)

plt.plot(
    fpr_rf,
    tpr_rf,
    label=f"Random Forest (AUC = {auc_rf:.4f})"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Random Classifier"
)

plt.title("ROC Curve - Fraud Detection")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 24. PRECISION-RECALL CURVE
# ============================================================

precision_logistic, recall_logistic, _ = precision_recall_curve(
    y_test,
    logistic_prob
)

precision_rf, recall_rf, _ = precision_recall_curve(
    y_test,
    rf_prob
)

pr_auc_logistic = average_precision_score(
    y_test,
    logistic_prob
)

pr_auc_rf = average_precision_score(
    y_test,
    rf_prob
)


plt.figure(figsize=(8, 6))

plt.plot(
    recall_logistic,
    precision_logistic,
    label=f"Logistic Regression (PR-AUC = {pr_auc_logistic:.4f})"
)

plt.plot(
    recall_rf,
    precision_rf,
    label=f"Random Forest (PR-AUC = {pr_auc_rf:.4f})"
)

plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")

plt.legend()
plt.grid(True)

plt.show()


# ============================================================
# 25. FEATURE IMPORTANCE - RANDOM FOREST
# ============================================================

feature_importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": random_forest_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
    by="Importance",
    ascending=False
)

print("\nTop 15 Important Features - Random Forest:")
print(feature_importance.head(15))


plt.figure(figsize=(10, 7))

sns.barplot(
    data=feature_importance.head(15),
    x="Importance",
    y="Feature"
)

plt.title("Top 15 Feature Importances - Random Forest")
plt.xlabel("Importance")
plt.ylabel("Feature")

plt.show()


# ============================================================
# 26. LOGISTIC REGRESSION COEFFICIENT ANALYSIS
# ============================================================

coefficients = pd.DataFrame({
    "Feature": X.columns,
    "Coefficient": logistic_model.coef_[0]
})

coefficients["Absolute_Coefficient"] = (
    coefficients["Coefficient"].abs()
)

coefficients = coefficients.sort_values(
    by="Absolute_Coefficient",
    ascending=False
)

print("\nTop 15 Logistic Regression Features:")
print(coefficients.head(15))


plt.figure(figsize=(10, 7))

sns.barplot(
    data=coefficients.head(15),
    x="Coefficient",
    y="Feature"
)

plt.title("Top Logistic Regression Coefficients")
plt.xlabel("Coefficient")
plt.ylabel("Feature")

plt.show()


# ============================================================
# 27. WHICH METRIC MATTERS MOST?
# ============================================================

print("\n" + "=" * 60)
print("PRECISION VS RECALL")
print("=" * 60)

print("""
Precision:
Of all transactions predicted as fraud, how many were
actually fraudulent?

Recall:
Of all actual fraudulent transactions, how many were
successfully detected?

For fraud detection, Recall is often very important because
missing a fraudulent transaction can result in financial loss.

However, very high Recall may produce more false positives.
Therefore, the practical goal is to achieve a good balance
between Recall and Precision.

F1-Score combines Precision and Recall into a single metric.

Because this dataset is highly imbalanced, PR-AUC is also
particularly useful for evaluating fraud detection performance.
""")


# ============================================================
# 28. IDENTIFY BEST MODEL
# ============================================================

best_model = results.loc[
    results["F1-Score"].idxmax()
]

print("\n" + "=" * 60)
print("BEST MODEL BASED ON F1-SCORE")
print("=" * 60)

print("Model:", best_model["Model"])
print("Precision:", round(best_model["Precision"], 4))
print("Recall:", round(best_model["Recall"], 4))
print("F1-Score:", round(best_model["F1-Score"], 4))
print("ROC-AUC:", round(best_model["ROC-AUC"], 4))
print("PR-AUC:", round(best_model["PR-AUC"], 4))


# ============================================================
# 29. SAMPLE FRAUD PREDICTION
# ============================================================

# Use the Random Forest model for demonstration

sample_transaction = X_test.iloc[[0]]

sample_scaled = scaler.transform(
    sample_transaction
)

prediction = random_forest_model.predict(
    sample_scaled
)[0]

probability = random_forest_model.predict_proba(
    sample_scaled
)[0][1]

print("\n" + "=" * 60)
print("SAMPLE TRANSACTION PREDICTION")
print("=" * 60)

if prediction == 1:
    print("Prediction: FRAUDULENT TRANSACTION")
else:
    print("Prediction: LEGITIMATE TRANSACTION")

print(f"Fraud Probability: {probability:.4f}")


# ============================================================
# 30. SCALABILITY DISCUSSION
# ============================================================

print("\n" + "=" * 60)
print("SCALABILITY: 1 MILLION TRANSACTIONS PER HOUR")
print("=" * 60)

print("""
For a system processing 1 million transactions per hour,
a batch-only approach may not be sufficient.

A production fraud detection system could use:

1. Real-time or streaming data processing.
2. A trained model deployed as an API/service.
3. Batch processing for historical analysis.
4. Feature engineering pipelines.
5. Distributed processing technologies such as Spark.
6. Model monitoring for data drift and fraud-pattern changes.
7. Threshold tuning based on the business cost of false
   positives and false negatives.

The model should be trained offline and then used for
fast predictions on incoming transactions.

SMOTE should be used during model training rather than
during real-time prediction.
""")


# ============================================================
# 31. FINAL CONCLUSION
# ============================================================

print("\n" + "=" * 60)
print("PROJECT CONCLUSION")
print("=" * 60)

print("""
This project developed a machine learning pipeline for
detecting fraudulent financial transactions.

The dataset was highly imbalanced, with fraudulent
transactions representing only a small percentage of all
transactions.

SMOTE was applied only to the training data to address
class imbalance while keeping the test data representative
of real-world transaction distributions.

Logistic Regression and Random Forest were trained and
evaluated using Precision, Recall, F1-Score, ROC-AUC and
PR-AUC.

Recall is especially important in fraud detection because
failure to identify fraudulent transactions can lead to
financial losses.

The final model can be further improved through threshold
tuning, advanced feature engineering, hyperparameter
optimization and real-time deployment.
""")


# ============================================================
# 32. SAVE RESULTS
# ============================================================

results.to_csv(
    "fraud_model_comparison.csv",
    index=False
)

feature_importance.to_csv(
    "fraud_feature_importance.csv",
    index=False
)

print("\nResult files saved successfully:")
print("1. fraud_model_comparison.csv")
print("2. fraud_feature_importance.csv")

print("\nFraud Detection Project Completed Successfully!")
