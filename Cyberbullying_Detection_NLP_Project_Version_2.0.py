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
except LookupError:
    nltk.download('stopwords')
    print("NLTK stopwords downloaded.")
try:
    nltk.data.find('stemmers/porter')
except LookupError:
    nltk.download('porter')
    print("NLTK porter stemmer downloaded.")
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
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

    This dataset includes more diverse examples inspired by definitions of bullying.
    """
    data = {
        'text': [
            "You are so stupid and ugly, no one likes you!", # Bullying: Name-calling, undermining
            "Having a great day with friends!", # Not bullying
            "Stop bothering me, you're a loser.", # Bullying: Name-calling, undermining
            "The weather is nice today.", # Not bullying
            "I will make your life a living hell.", # Bullying: Threatening
            "Let's grab some coffee later.", # Not bullying
            "You're pathetic and deserve nothing good.", # Bullying: Undermining
            "Just finished my workout.", # Not bullying
            "Why are you always so annoying? Go away!", # Bullying: Name-calling, pushing away
            "What a beautiful sunset tonight.", # Not bullying
            "You should just disappear, nobody cares about you.", # Bullying: Undermining, social exclusion
            "Enjoying a quiet evening at home.", # Not bullying
            "Your ideas are worthless.", # Bullying: Undermining
            "Thinking about a new project.", # Not bullying
            "I hate your guts, you're a waste of space.", # Bullying: Name-calling, undermining

            # Additional examples inspired by bullying definitions (name-calling, spreading rumors, threatening)
            "haha u r so dumb lol", # Bullying: Slang, name-calling
            "This is amazing! 🎉 So happy for you!", # Not bullying: Positive, emoji
            "Shut up, you deserve all the bad things. 😠", # Bullying: Aggressive, emoji
            "I'm feeling great 😊", # Not bullying: Positive, emoji
            "gtfo, nobody wants you here", # Bullying: Acronym, social exclusion
            "You're the worst person ever smh", # Bullying: Name-calling, acronym
            "Did you hear the rumor about them? It's really bad.", # Bullying: Spreading rumors
            "I'm going to tell everyone your secret if you don't do what I say.", # Bullying: Threatening, coercion
            "You are a total fraud, everyone knows it now.", # Bullying: Undermining, spreading rumors
            "Just had a lovely time at the park.", # Not bullying
            "Your clothes are so ugly, nobody wants to hang out with you.", # Bullying: Undermining, social exclusion
            "Reading a good book right now. 📚", # Not bullying
            "Don't even try, you'll fail anyway. ", # Bullying: Undermining
            "This game is so much fun!", # Not bullying
            "I heard you cheated on the test. Everyone's talking about it." # Bullying: Spreading rumors
        ],
        'is_cyberbullying': [
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
            1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1
        ] # 1 for bullying, 0 for not bullying
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
    - Handles basic emojis and common internet slang/acronyms
    """
    # Lowercase
    text = text.lower()

    # Define a simple emoji removal pattern (basic, more complex needed for full coverage)
    # This pattern removes common emoji ranges.
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # emoticons
        "\U0001F300-\U0001F5FF"  # symbols & pictographs
        "\U0001F680-\U0001F6FF"  # transport & map symbols
        "\U0001F1E0-\U0001F1FF"  # flags (iOS)
        "\U00002702-\U000027B0"
        "\U000024C2-\U0001F251"
        "]+", flags=re.UNICODE
    )
    text = emoji_pattern.sub(r'', text)

    # Basic slang/acronym replacement (can be expanded)
    slang_replacements = {
        'lol': 'laughing out loud',
        'brb': 'be right back',
        'lmao': 'laughing my ass off',
        'rofl': 'rolling on floor laughing',
        'aka': 'also known as',
        'gtfo': 'get out', # Example of an aggressive one
        'smh': 'shaking my head',
        'ikr': 'i know right',
        'idk': 'i do not know',
        'ttyl': 'talk to you later',
        'afk': 'away from keyboard'
    }
    words = text.split()
    words = [slang_replacements.get(word, word) for word in words]
    text = ' '.join(words)

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
        # Added zero_division=0 to suppress warning about 0 precision/recall
        print(classification_report(y_test, y_pred, zero_division=0)) 
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
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
    predict_cyberbullying("You are a total loser. get lost!!", trained_model, tfidf_vectorizer)
    predict_cyberbullying("Feeling good today. 😊 What about you?", trained_model, tfidf_vectorizer)
    predict_cyberbullying("Why are you always so dumb? smh", trained_model, tfidf_vectorizer)
    predict_cyberbullying("Hey, how are you doing? Hope you're well.", trained_model, tfidf_vectorizer)
    predict_cyberbullying("I heard you're spreading rumors about me.", trained_model, tfidf_vectorizer)
    predict_cyberbullying("Stop bothering me, or else.", trained_model, tfidf_vectorizer)
    predict_cyberbullying("Let's go grab a bite to eat.", trained_model, tfidf_vectorizer)
    predict_cyberbullying("You should just disappear, nobody cares about you.", trained_model, tfidf_vectorizer)
    predict_cyberbullying("Your ideas are worthless and stupid.", trained_model, tfidf_vectorizer)
    predict_cyberbullying("I love this new song, it's amazing!", trained_model, tfidf_vectorizer)