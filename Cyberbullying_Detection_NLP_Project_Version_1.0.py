import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib # For saving/loading the model and vectorizer
import os

# Ensure NLTK data is downloaded (run once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError: # Changed to LookupError
    nltk.download('stopwords')
    print("NLTK stopwords downloaded.")
try:
    nltk.data.find('stemmers/porter')
except LookupError: # Changed to LookupError
    nltk.download('porter') # Corrected NLTK download for porter stemmer
    print("NLTK porter stemmer downloaded.")
# Ensure punkt tokenizer is downloaded for word_tokenize
try:
    nltk.data.find('tokenizers/punkt')
except LookupError: # Changed to LookupError
    nltk.download('punkt')
    print("NLTK punkt tokenizer downloaded.")


# --- Configuration for persistence ---
MODEL_FILE = 'cyberbullying_model.joblib'
VECTORIZER_FILE = 'tfidf_vectorizer.joblib'

# --- 1. Data Collection (Dummy Data for Demonstration) ---
# In a real project, you would load a dataset from a file (e.g., CSV).
# Example of a real dataset structure: 'text' column for content, 'is_cyberbullying' (0 or 1) for label.
def load_dummy_data():
    """
    Creates a small, dummy dataset for demonstration purposes.
    In a real scenario, you'd load data from a CSV, JSON, etc.
    """
    data = {
        'text': [
            "You are so stupid and ugly, no one likes you!",
            "Having a great day with friends!",
            "Stop bothering me, you're a loser.",
            "The weather is nice today.",
            "I will make your life a living hell.",
            "Let's grab some coffee later.",
            "You're pathetic and deserve nothing good.",
            "Just finished my workout.",
            "Why are you always so annoying? Go away!",
            "What a beautiful sunset tonight.",
            "You should just disappear, nobody cares about you.",
            "Enjoying a quiet evening at home.",
            "Your ideas are worthless.",
            "Thinking about a new project.",
            "I hate your guts, you're a waste of space."
        ],
        'is_cyberbullying': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1] # 1 for bullying, 0 for not bullying
    }
    df = pd.DataFrame(data)
    print("Dummy dataset loaded successfully.")
    print(df.head())
    return df

# --- 2. Data Preprocessing ---
def preprocess_text(text):
    """
    Cleans and preprocesses text data:
    - Lowercases text
    - Removes punctuation
    - Removes numbers
    - Removes stopwords
    - Applies stemming
    """
    # Lowercase
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Tokenize
    words = nltk.word_tokenize(text)
    # Remove stopwords and stem
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    processed_words = [stemmer.stem(word) for word in words if word not in stop_words]
    return ' '.join(processed_words)

# --- Main Execution ---
if __name__ == "__main__":
    trained_model = None
    tfidf_vectorizer = None

    # Check if a trained model and vectorizer exist
    if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
        print(f"Loading existing model from {MODEL_FILE} and vectorizer from {VECTORIZER_FILE}...")
        try:
            trained_model = joblib.load(MODEL_FILE)
            tfidf_vectorizer = joblib.load(VECTORIZER_FILE)
            print("Model and vectorizer loaded successfully. Skipping training.")
        except Exception as e:
            print(f"Error loading saved files: {e}. Proceeding with fresh training.")
            trained_model = None # Reset to trigger retraining
            tfidf_vectorizer = None
    else:
        print("No existing model or vectorizer found. Proceeding with data loading and training.")

    # Load data (or create dummy data if no model exists)
    df = load_dummy_data()

    if df.empty:
        print("No data available to train or predict. Exiting.")
        exit()

    # Apply preprocessing
    print("Preprocessing text data...")
    df['cleaned_text'] = df['text'].apply(preprocess_text)
    print("Text preprocessing complete.")
    print(df[['text', 'cleaned_text']].head())

    # --- 3. Feature Extraction (TF-IDF) ---
    # Only fit and transform if a vectorizer was not loaded
    if tfidf_vectorizer is None:
        print("Fitting TF-IDF Vectorizer and transforming text...")
        tfidf_vectorizer = TfidfVectorizer(max_features=5000) # Limit features to 5000 for simplicity
        X = tfidf_vectorizer.fit_transform(df['cleaned_text'])
        print("TF-IDF Vectorizer fitted and text transformed.")
        # Save the fitted vectorizer
        joblib.dump(tfidf_vectorizer, VECTORIZER_FILE)
        print(f"TF-IDF Vectorizer saved to {VECTORIZER_FILE}.")
    else:
        print("Transforming text using loaded TF-IDF Vectorizer...")
        X = tfidf_vectorizer.transform(df['cleaned_text'])
        print("Text transformed using loaded vectorizer.")

    y = df['is_cyberbullying']

    # --- 4. Model Training ---
    # Only train if a model was not loaded
    if trained_model is None:
        print("Splitting data into training and testing sets...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        print(f"Training set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

        print("Training Logistic Regression model...")
        trained_model = LogisticRegression(max_iter=1000) # Increased max_iter for convergence
        trained_model.fit(X_train, y_train)
        print("Model trained successfully.")

        # Save the trained model
        joblib.dump(trained_model, MODEL_FILE)
        print(f"Trained model saved to {MODEL_FILE}.")

        # --- 5. Evaluation ---
        print("\n--- Model Evaluation ---")
        y_pred = trained_model.predict(X_test)
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("------------------------\n")
    else:
        print("\nSkipping model training as an existing model was loaded.")
        # If a model was loaded, we can still evaluate it on the full data or new test data if available
        # For this example, we'll assume the loaded model is ready for prediction.


    # --- 6. Prediction on New Text ---
    def predict_cyberbullying(text_to_predict, model, vectorizer):
        """
        Predicts whether a given text is cyberbullying.
        """
        print(f"\nPredicting for new text: '{text_to_predict}'")
        cleaned_text = preprocess_text(text_to_predict)
        text_vectorized = vectorizer.transform([cleaned_text])
        prediction = model.predict(text_vectorized)
        probability = model.predict_proba(text_vectorized)[:, 1][0] # Probability of being class 1 (bullying)
        
        if prediction[0] == 1:
            print(f"Prediction: This text is likely cyberbullying. (Probability: {probability:.2f})")
        else:
            print(f"Prediction: This text is NOT cyberbullying. (Probability: {1-probability:.2f})")
        return prediction[0], probability

    # Example predictions
    predict_cyberbullying("You are so annoying, I hate you!", trained_model, tfidf_vectorizer)
    predict_cyberbullying("What a beautiful day it is today.", trained_model, tfidf_vectorizer)
    predict_cyberbullying("I'm going to find you and make you regret this.", trained_model, tfidf_vectorizer)
    predict_cyberbullying("Could you please pass me the salt?", trained_model, tfidf_vectorizer)

