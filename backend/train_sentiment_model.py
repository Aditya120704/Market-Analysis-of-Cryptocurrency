import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import os

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'training.1600000.processed.noemoticon.csv')
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

# Load data
# The dataset has no header, so we define column names
# Column 0: sentiment (0 = negative, 4 = positive)
# Column 5: text
df = pd.read_csv(DATA_PATH, encoding='ISO-8859-1', engine='python', header=None)
df = df[[0, 5]]
df.columns = ['sentiment', 'text']

# Sample a smaller dataset for faster training during development
# For production, consider using the full dataset or a larger sample
df_sample = df.sample(n=100000, random_state=42) # Adjust sample size as needed

# Preprocessing: Convert sentiment 4 to 1 for binary classification (positive)
df_sample['sentiment'] = df_sample['sentiment'].replace(4, 1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(df_sample['text'], df_sample['sentiment'], test_size=0.2, random_state=42)

# Train TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000) # Limit features to 5000
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Train Logistic Regression Model
model = LogisticRegression(max_iter=1000) # Increase max_iter for convergence
model.fit(X_train_vec, y_train)

# Evaluate model
y_pred = model.predict(X_test_vec)
accuracy = accuracy_score(y_test, y_pred)
print(f"Sentiment Model Accuracy: {accuracy:.4f}")

# Save model and vectorizer
joblib.dump(model, os.path.join(MODELS_DIR, 'sentiment_model.pkl'))
joblib.dump(vectorizer, os.path.join(MODELS_DIR, 'vectorizer.pkl'))
print("Sentiment model and vectorizer saved successfully.")