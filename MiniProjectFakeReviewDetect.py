import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Sample Dataset
data = {
    "review": [
        "This product is amazing",
        "Worst product ever",
        "I love this item",
        "Fake quality and bad service",
        "Excellent product and fast delivery",
        "Totally useless item",
        "Very happy with the purchase",
        "Do not buy this product"
    ],
    "label": [
        "Real",
        "Fake",
        "Real",
        "Fake",
        "Real",
        "Fake",
        "Real",
        "Fake"
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

print("Dataset:\n")
print(df)

# Features and Labels
X = df["review"]
y = df["label"]

# Text Vectorization
vectorizer = CountVectorizer()

X_vector = vectorizer.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_vector,
    y,
    test_size=0.3,
    random_state=42
)

# Model Training
model = MultinomialNB()

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
print("\nAccuracy:\n")
print(accuracy_score(y_test, y_pred))

# Classification Report
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Test Review
sample_review = ["This product is excellent"]

sample_vector = vectorizer.transform(sample_review)

prediction = model.predict(sample_vector)

print("\nTest Review Prediction:\n")
print("Review:", sample_review[0])
print("Prediction:", prediction[0])