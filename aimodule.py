import json
import pandas as pd
import re # library for regex
# Load the JSON dataset
with open('after-change3.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# Convert to a DataFrame (assuming each entry has 'text' and 'label')
df = pd.DataFrame(data)


# data pre-prossessing
# nothing here


# 20 80 split for train, test
from sklearn.model_selection import train_test_split
X = df['text']  # feature
y = df['sentiment']  # target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")

#text to numerical features
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train).toarray()
X_test_tfidf = vectorizer.transform(X_test).toarray()

# Check the shape of the transformed data
print(f"Training data shape: {X_train_tfidf.shape}")
print(f"Testing data shape: {X_test_tfidf.shape}")


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Initialize the model
model = LogisticRegression(max_iter=1000, random_state=42)

# Train the model on the TF-IDF features and training labels
model.fit(X_train_tfidf, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Display detailed evaluation metrics
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# Define a function to preprocess, vectorize, and predict
def predict_sentiment(text):    
    # Convert the text to TF-IDF features
    text_tfidf = vectorizer.transform([text]).toarray()
    
    # Predict the sentiment
    prediction = model.predict(text_tfidf)
    
    return prediction[0]  # Return the predicted label

if __name__ == "__main__":
    custom_text = "" 
    result = predict_sentiment(custom_text)

    # Output the result
    print(f"Predicted Sentiment: {result}")

