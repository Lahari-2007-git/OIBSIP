# ============================================================
# OIBSIP DATA ANALYTICS - LEVEL 1 TASK 4
# SENTIMENT ANALYSIS
# ============================================================

# ============================================================
# 1. IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np
import re
import string
import matplotlib.pyplot as plt

from collections import Counter

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from wordcloud import WordCloud


# ============================================================
# 2. DOWNLOAD NLTK RESOURCES
# ============================================================

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")


# ============================================================
# 3. LOAD DATASET
# ============================================================

df = pd.read_csv("Twitter_Data.csv")

print("Dataset loaded successfully!")

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nFirst 5 rows:")
print(df.head())


# ============================================================
# 4. INITIAL DATA INSPECTION
# ============================================================

print("\n" + "=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print(df.info())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:")
print(df.duplicated().sum())

print("\nData types:")
print(df.dtypes)


# ============================================================
# 5. REMOVE MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("HANDLING MISSING VALUES")
print("=" * 60)

print("Missing values before cleaning:")
print(df.isnull().sum())

# Remove rows where text or sentiment category is missing
df = df.dropna(subset=["clean_text", "category"])

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nDataset shape after removing missing values:")
print(df.shape)


# ============================================================
# 6. REMOVE DUPLICATES
# ============================================================

duplicates_before = df.duplicated().sum()

df = df.drop_duplicates()

duplicates_after = df.duplicated().sum()

print("\nDuplicates before removal:", duplicates_before)
print("Duplicates after removal:", duplicates_after)


# ============================================================
# 7. CONVERT SENTIMENT LABELS
# ============================================================

# Convert category to integer
df["category"] = df["category"].astype(int)

# Map numerical labels to sentiment names
sentiment_mapping = {
    -1: "Negative",
     0: "Neutral",
     1: "Positive"
}

df["sentiment"] = df["category"].map(sentiment_mapping)

print("\nSentiment labels:")
print(df["sentiment"].value_counts())


# ============================================================
# 8. CHECK CLASS DISTRIBUTION
# ============================================================

sentiment_counts = df["sentiment"].value_counts()

print("\n" + "=" * 60)
print("SENTIMENT CLASS DISTRIBUTION")
print("=" * 60)

print(sentiment_counts)

print("\nSentiment percentages:")
print(
    (sentiment_counts / len(df) * 100).round(2)
)


# ============================================================
# 9. SENTIMENT DISTRIBUTION VISUALIZATION
# ============================================================

plt.figure(figsize=(8, 5))

sentiment_counts.reindex(
    ["Positive", "Neutral", "Negative"]
).plot(
    kind="bar"
)

plt.title("Sentiment Distribution")
plt.xlabel("Sentiment")
plt.ylabel("Number of Tweets")
plt.xticks(rotation=0)

plt.tight_layout()
plt.show()


# ============================================================
# 10. TEXT PREPROCESSING
# ============================================================

stop_words = set(stopwords.words("english"))


def preprocess_text(text):
    
    # Convert to string
    text = str(text)
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)
    
    # Remove mentions
    text = re.sub(r"@\w+", "", text)
    
    # Remove hashtags symbol but keep the word
    text = re.sub(r"#", "", text)
    
    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )
    
    # Remove numbers
    text = re.sub(r"\d+", "", text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Stopword removal
    tokens = [
        word for word in tokens
        if word not in stop_words
    ]
    
    # Keep only alphabetic words
    tokens = [
        word for word in tokens
        if word.isalpha()
    ]
    
    return " ".join(tokens)


print("\n" + "=" * 60)
print("TEXT PREPROCESSING")
print("=" * 60)

# Apply preprocessing
df["processed_text"] = df["clean_text"].apply(
    preprocess_text
)

print("\nOriginal text:")
print(df["clean_text"].iloc[0])

print("\nProcessed text:")
print(df["processed_text"].iloc[0])


# ============================================================
# 11. REMOVE EMPTY TEXT AFTER PREPROCESSING
# ============================================================

empty_text_count = (
    df["processed_text"].str.strip() == ""
).sum()

print("\nEmpty texts after preprocessing:",
      empty_text_count)

df = df[
    df["processed_text"].str.strip() != ""
]


# ============================================================
# 12. DEFINE FEATURES AND TARGET
# ============================================================

X = df["processed_text"]
y = df["category"]


# ============================================================
# 13. TRAIN / TEST SPLIT - 80/20
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 60)
print("TRAIN / TEST SPLIT")
print("=" * 60)

print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 14. TF-IDF FEATURE EXTRACTION
# ============================================================

"""
TF-IDF (Term Frequency-Inverse Document Frequency)
converts text into numerical features that machine
learning algorithms can understand.

It gives higher importance to words that are frequent
in a particular document but less common across the
entire collection of documents.
"""

tfidf = TfidfVectorizer(
    max_features=30000,
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)

X_train_tfidf = tfidf.fit_transform(X_train)

X_test_tfidf = tfidf.transform(X_test)

print("\n" + "=" * 60)
print("TF-IDF FEATURE EXTRACTION")
print("=" * 60)

print("Training TF-IDF shape:",
      X_train_tfidf.shape)

print("Testing TF-IDF shape:",
      X_test_tfidf.shape)


# ============================================================
# 15. MODEL 1 - NAIVE BAYES
# ============================================================

print("\n" + "=" * 60)
print("MODEL 1 - NAIVE BAYES")
print("=" * 60)

nb_model = MultinomialNB()

nb_model.fit(
    X_train_tfidf,
    y_train
)

nb_predictions = nb_model.predict(
    X_test_tfidf
)

print("Naive Bayes training completed!")


# ============================================================
# 16. MODEL 2 - LOGISTIC REGRESSION
# ============================================================

print("\n" + "=" * 60)
print("MODEL 2 - LOGISTIC REGRESSION")
print("=" * 60)

lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr_model.fit(
    X_train_tfidf,
    y_train
)

lr_predictions = lr_model.predict(
    X_test_tfidf
)

print("Logistic Regression training completed!")


# ============================================================
# 17. EVALUATION FUNCTION
# ============================================================

def evaluate_model(model_name, y_true, y_pred):
    
    accuracy = accuracy_score(
        y_true,
        y_pred
    )
    
    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )
    
    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )
    
    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )
    
    print("\n" + "=" * 60)
    print(model_name)
    print("=" * 60)
    
    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))
    
    print("\nClassification Report:")
    
    print(
        classification_report(
            y_true,
            y_pred,
            labels=[-1, 0, 1],
            target_names=[
                "Negative",
                "Neutral",
                "Positive"
            ],
            zero_division=0
        )
    )
    
    return {
        "Model": model_name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }


# ============================================================
# 18. EVALUATE BOTH MODELS
# ============================================================

nb_results = evaluate_model(
    "Naive Bayes",
    y_test,
    nb_predictions
)

lr_results = evaluate_model(
    "Logistic Regression",
    y_test,
    lr_predictions
)


# ============================================================
# 19. MODEL COMPARISON
# ============================================================

results = pd.DataFrame([
    nb_results,
    lr_results
])

print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    results.round(4)
)


# ============================================================
# 20. MODEL COMPARISON BAR CHART
# ============================================================

results_plot = results.set_index("Model")

results_plot[
    ["Accuracy", "Precision", "Recall", "F1 Score"]
].plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Model Performance Comparison")
plt.xlabel("Model")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)

plt.legend(
    title="Metrics"
)

plt.tight_layout()
plt.show()


# ============================================================
# 21. CONFUSION MATRIX - NAIVE BAYES
# ============================================================

nb_cm = confusion_matrix(
    y_test,
    nb_predictions,
    labels=[-1, 0, 1]
)

plt.figure(figsize=(7, 5))

plt.imshow(nb_cm)

plt.title("Confusion Matrix - Naive Bayes")
plt.xlabel("Predicted Sentiment")
plt.ylabel("Actual Sentiment")

plt.xticks(
    [0, 1, 2],
    ["Negative", "Neutral", "Positive"]
)

plt.yticks(
    [0, 1, 2],
    ["Negative", "Neutral", "Positive"]
)

for i in range(3):
    for j in range(3):
        plt.text(
            j,
            i,
            nb_cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()
plt.show()


# ============================================================
# 22. CONFUSION MATRIX - LOGISTIC REGRESSION
# ============================================================

lr_cm = confusion_matrix(
    y_test,
    lr_predictions,
    labels=[-1, 0, 1]
)

plt.figure(figsize=(7, 5))

plt.imshow(lr_cm)

plt.title("Confusion Matrix - Logistic Regression")
plt.xlabel("Predicted Sentiment")
plt.ylabel("Actual Sentiment")

plt.xticks(
    [0, 1, 2],
    ["Negative", "Neutral", "Positive"]
)

plt.yticks(
    [0, 1, 2],
    ["Negative", "Neutral", "Positive"]
)

for i in range(3):
    for j in range(3):
        plt.text(
            j,
            i,
            lr_cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()
plt.show()


# ============================================================
# 23. SELECT BEST MODEL
# ============================================================

best_model_name = results.loc[
    results["F1 Score"].idxmax(),
    "Model"
]

print("\n" + "=" * 60)
print("BEST MODEL")
print("=" * 60)

print(
    "Best performing model:",
    best_model_name
)


# ============================================================
# 24. WORD CLOUD - POSITIVE SENTIMENT
# ============================================================

positive_text = " ".join(
    df.loc[
        df["category"] == 1,
        "processed_text"
    ]
)

if positive_text.strip():
    
    positive_wordcloud = WordCloud(
        width=900,
        height=500,
        background_color="white",
        max_words=100
    ).generate(positive_text)
    
    plt.figure(figsize=(10, 6))
    
    plt.imshow(
        positive_wordcloud,
        interpolation="bilinear"
    )
    
    plt.axis("off")
    
    plt.title("Positive Sentiment WordCloud")
    
    plt.show()


# ============================================================
# 25. WORD CLOUD - NEGATIVE SENTIMENT
# ============================================================

negative_text = " ".join(
    df.loc[
        df["category"] == -1,
        "processed_text"
    ]
)

if negative_text.strip():
    
    negative_wordcloud = WordCloud(
        width=900,
        height=500,
        background_color="white",
        max_words=100
    ).generate(negative_text)
    
    plt.figure(figsize=(10, 6))
    
    plt.imshow(
        negative_wordcloud,
        interpolation="bilinear"
    )
    
    plt.axis("off")
    
    plt.title("Negative Sentiment WordCloud")
    
    plt.show()


# ============================================================
# 26. WORD CLOUD - NEUTRAL SENTIMENT
# ============================================================

neutral_text = " ".join(
    df.loc[
        df["category"] == 0,
        "processed_text"
    ]
)

if neutral_text.strip():
    
    neutral_wordcloud = WordCloud(
        width=900,
        height=500,
        background_color="white",
        max_words=100
    ).generate(neutral_text)
    
    plt.figure(figsize=(10, 6))
    
    plt.imshow(
        neutral_wordcloud,
        interpolation="bilinear"
    )
    
    plt.axis("off")
    
    plt.title("Neutral Sentiment WordCloud")
    
    plt.show()


# ============================================================
# 27. ERROR ANALYSIS - LOGISTIC REGRESSION
# ============================================================

error_analysis = pd.DataFrame({
    "Text": X_test.values,
    "Actual": y_test.values,
    "Predicted": lr_predictions
})

misclassified = error_analysis[
    error_analysis["Actual"] !=
    error_analysis["Predicted"]
]

print("\n" + "=" * 60)
print("ERROR ANALYSIS")
print("=" * 60)

print(
    "Total misclassified examples:",
    len(misclassified)
)


# Display 5 misclassified examples
print("\nFive misclassified examples:\n")

for i, row in misclassified.head(5).iterrows():
    
    actual_label = sentiment_mapping[
        row["Actual"]
    ]
    
    predicted_label = sentiment_mapping[
        row["Predicted"]
    ]
    
    print("-" * 60)
    
    print("Text:")
    print(row["Text"])
    
    print("\nActual Sentiment:",
          actual_label)
    
    print("Predicted Sentiment:",
          predicted_label)


# ============================================================
# 28. ERROR ANALYSIS - DISCUSSION
# ============================================================

print("\n" + "=" * 60)
print("ERROR ANALYSIS DISCUSSION")
print("=" * 60)

print("""
Possible reasons for misclassification include:

1. Tweets can contain sarcasm or irony, which is difficult
   for traditional machine learning models to understand.

2. Some tweets may contain mixed positive and negative
   opinions.

3. Short tweets provide limited contextual information.

4. Slang, abbreviations, emojis and informal language can
   make sentiment classification difficult.

5. Neutral and positive/negative expressions can sometimes
   use similar words.

6. TF-IDF mainly represents word importance and does not
   fully understand the deeper meaning or context of a tweet.
""")


# ============================================================
# 29. SAMPLE PREDICTION FUNCTION
# ============================================================

def predict_sentiment(text):
    
    processed = preprocess_text(text)
    
    transformed = tfidf.transform(
        [processed]
    )
    
    prediction = lr_model.predict(
        transformed
    )[0]
    
    return sentiment_mapping[prediction]


# ============================================================
# 30. TEST THE MODEL WITH SAMPLE TEXT
# ============================================================

print("\n" + "=" * 60)
print("SAMPLE SENTIMENT PREDICTIONS")
print("=" * 60)

sample_texts = [
    "I really enjoyed this amazing experience!",
    "This is the worst service I have ever received.",
    "The event will be held tomorrow."
]

for text in sample_texts:
    
    prediction = predict_sentiment(text)
    
    print("\nText:", text)
    print("Predicted Sentiment:", prediction)


# ============================================================
# 31. FINAL MODEL PERFORMANCE
# ============================================================

print("\n" + "=" * 60)
print("FINAL MODEL PERFORMANCE")
print("=" * 60)

for _, row in results.iterrows():
    
    print("\nModel:", row["Model"])
    print("Accuracy :", round(row["Accuracy"], 4))
    print("Precision:", round(row["Precision"], 4))
    print("Recall   :", round(row["Recall"], 4))
    print("F1 Score :", round(row["F1 Score"], 4))


# ============================================================
# 32. FINAL PROJECT INSIGHTS
# ============================================================

print("\n" + "=" * 60)
print("FINAL PROJECT INSIGHTS")
print("=" * 60)

print("""
1. The dataset contains positive, negative and neutral
   sentiment classes.

2. Text preprocessing was performed using lowercase
   conversion, punctuation removal, tokenization and
   stopword removal.

3. TF-IDF was used to convert text into numerical features.

4. Two machine learning classifiers were trained:
   Naive Bayes and Logistic Regression.

5. Both models were evaluated using accuracy, precision,
   recall and F1-score.

6. Confusion matrices were used to understand classification
   errors.

7. WordClouds were generated for positive, negative and
   neutral sentiment classes.

8. Misclassified tweets were examined during error analysis.

9. The model can be useful for analysing customer feedback,
   social media opinions, product reviews and public opinion.
""")


# ============================================================
# 33. SAVE RESULTS
# ============================================================

results.to_csv(
    "sentiment_model_results.csv",
    index=False
)

print("\nModel results saved as:")
print("sentiment_model_results.csv")


# ============================================================
# 34. SAVE PROCESSED DATASET
# ============================================================

df.to_csv(
    "processed_twitter_sentiment.csv",
    index=False
)

print("\nProcessed dataset saved as:")
print("processed_twitter_sentiment.csv")


# ============================================================
# 35. PROJECT COMPLETION
# ============================================================

print("\n" + "=" * 60)
print("SENTIMENT ANALYSIS PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 60)
