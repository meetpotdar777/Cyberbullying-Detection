import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer # Keeping PorterStemmer import for reference, but will use WordNetLemmatizer
from nltk.stem import WordNetLemmatizer # New import for lemmatization
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
except LookupError: # This might still trigger if 'porter' is specifically checked, but we're moving to wordnet
    nltk.download('porter') # Keep for robustness if other parts depend on it, or remove if strictly only WordNet is needed
    print("NLTK porter stemmer downloaded.")
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    print("NLTK punkt tokenizer downloaded.")
try: # New NLTK download for WordNet Lemmatizer
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    print("NLTK WordNet downloaded for lemmatization.")


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
            "You are so stupid and ugly, no one likes you!", # Bullying: Name-calling, undermining (1)
            "Having a great day with friends!", # Not bullying (0)
            "Stop bothering me, you're a loser.", # Bullying: Name-calling, undermining (1)
            "The weather is nice today.", # Not bullying (0)
            "I will make your life a living hell.", # Bullying: Threatening (1)
            "Let's grab some coffee later.", # Not bullying (0)
            "You're pathetic and deserve nothing good.", # Bullying: Undermining (1)
            "Just finished my workout.", # Not bullying (0)
            "Why are you always so annoying? Go away!", # Bullying: Name-calling, pushing away (1)
            "What a beautiful sunset tonight.", # Not bullying (0)
            "You should just disappear, nobody cares about you.", # Bullying: Undermining, social exclusion (1)
            "Enjoying a quiet evening at home.", # Not bullying (0)
            "Your ideas are worthless.", # Bullying: Undermining (1)
            "Thinking about a new project.", # Not bullying (0)
            "I hate your guts, you're a waste of space.", # Bullying: Name-calling, undermining (1)

            # Additional examples inspired by bullying definitions (name-calling, spreading rumors, threatening)
            "haha u r so dumb lol", # Bullying: Slang, name-calling (1)
            "This is amazing! 🎉 So happy for you!", # Not bullying: Positive, emoji (0)
            "Shut up, you deserve all the bad things. 😠", # Bullying: Aggressive, emoji (1)
            "I'm feeling great 😊", # Not bullying: Positive, emoji (0)
            "gtfo, nobody wants you here", # Bullying: Acronym, social exclusion (1)
            "You're the worst person ever smh", # Bullying: Name-calling, acronym (1)
            "Did you hear the rumor about them? It's really bad.", # Bullying: Spreading rumors (1)
            "I'm going to tell everyone your secret if you don't do what I say.", # Bullying: Threatening, coercion (1)
            "You are a total fraud, everyone knows it now.", # Bullying: Undermining, spreading rumors (1)
            "Just had a lovely time at the park.", # Not bullying (0)
            "Your clothes are so ugly, nobody wants to hang out with you.", # Bullying: Undermining, social exclusion (1)
            "Reading a good book right now. 📚", # Not bullying (0)
            "Don't even try, you'll fail anyway. ", # Bullying: Undermining (1)
            "This game is so much fun!", # Not bullying (0)
            "I heard you cheated on the test. Everyone's talking about it.", # Bullying: Spreading rumors (1)
            "Your mom is so fat, lol. Everyone laughs at her.", # Bullying: Name-calling, targeting family (1)
            "Heard you got fired. Couldn't happen to a nicer person. #karma", # Bullying: Undermining, schadenfreude (1)
            "Why do you even bother posting? You're terrible at everything. @loser", # Bullying: Undermining, mention (1)
            "I will hunt you down.", # Bullying: Threatening (1)
            "You're a clown. Stop embarrassing yourself. #pathetic", # Bullying: Name-calling, undermining, hashtag (1)
            "Having a wonderful day with my family. So blessed! 🙏", # Not bullying: Positive, emoji (0)
            "Just chilling at home, watching movies. 🎬�", # Not bullying: Neutral, emoji (0)
            "Can't believe how rude some people are online. It's disgusting. 😡", # Not bullying: Expressing negative sentiment about bullying (0)
            "I totally agree with what you said on Twitter! So insightful. 👍", # Not bullying: Positive, mention (0)
            "Learning new Python skills today! Feeling productive. #coding", # Not bullying: Positive, hashtag (0)
            "Hope you have a fantastic weekend! See you soon. 👋", # Not bullying: Positive, emoji (0)
            "You'll never amount to anything. Just give up already.", # Bullying: Undermining (1)
            "This is the worst project ever. You clearly have no idea what you're doing. 😂", # Bullying: Undermining, sarcastic emoji (1)
            "Such a beautiful sunset! Feeling very peaceful. 🌄", # Not bullying: Positive, emoji (0)
            "You deserve all the hate you get. Go away!", # Bullying: Aggressive, social exclusion (1)
            "I'm so excited for my vacation next month! ✈️🏖️", # Not bullying: Positive, emoji (0)
            "Heard you failed the exam. Not surprised given how lazy you are.", # Bullying: Undermining, spreading rumors (1)
            "What's up, fam? vibin' hard today. #goodvibes", # Not bullying: Slang, positive (0)
            "You're such a creep. Stay away from me.", # Bullying: Name-calling, aggressive (1)
            "Working hard on my new coding challenge. Wish me luck!", # Not bullying (0)
            "Your opinion doesn't matter. Just shut up.", # Bullying: Undermining, aggressive (1)
            "Enjoying a quiet coffee this morning. ☕", # Not bullying (0)
            "I'm going to ruin your reputation online.", # Bullying: Threatening (1)
            "Always happy to help out where I can! 😊", # Not bullying (0)
            "You're just jealous because you'll never be as good as me." # Bullying: Taunting (1)
        ],
        'is_cyberbullying': [
            1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,  # 15 entries
            1, 0, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1,  # 15 entries
            1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1,  # 15 entries (corrected to match new text items)
            0, 1, 0, 1, 0, 1, 0, 1, 0, 1 # Remaining 10 entries for last 10 text items
        ] # Total 55 text entries and 55 labels
    }
    
    # Debug prints to ensure lists have same length before DataFrame creation
    print(f"DEBUG: Length of 'text' list: {len(data['text'])}")
    print(f"DEBUG: Length of 'is_cyberbullying' list: {len(data['is_cyberbullying'])}")

    df = pd.DataFrame(data)
    print("Dummy dataset loaded successfully.")
    print(df.head())
    return df

# --- 2. Data Preprocessing ---
def preprocess_text(text):
    """
    Cleans and preprocesses text data:
    - Lowercases text
    - Removes URLs, mentions, hashtags
    - Expands contractions
    - Handles basic emojis and common internet slang/acronyms
    - Removes punctuation
    - Removes numbers
    - Removes stopwords
    - Applies lemmatization (using WordNet)
    - Removes extra whitespaces
    """
    # Lowercase
    text = text.lower()

    # Remove URLs (e.g., http://, https://, www.)
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove mentions (@username)
    text = re.sub(r'@\w+', '', text)
    # Remove hashtags (#hashtag)
    text = re.sub(r'#\w+', '', text)

    # Basic contraction expansion (can be significantly expanded)
    contractions = {
        "ain't": "am not", "aren't": "are not", "can't": "cannot", "can't've": "cannot have",
        "could've": "could have", "couldn't": "could not", "didn't": "did not",
        "doesn't": "does not", "don't": "do not", "hadn't": "had not", "hasn't": "has not",
        "haven't": "have not", "he'd": "he would", "he'll": "he will", "he's": "he is",
        "how'd": "how did", "how'll": "how will", "how's": "how is", "i'd": "i would",
        "i'll": "i will", "i'm": "i am", "i've": "i have", "isn't": "is not",
        "it'd": "it had", "it'll": "it will", "it's": "it is", "let's": "let us",
        "mightn't": "might not", "mustn't": "must not", "shan't": "shall not",
        "she'd": "she would", "she'll": "she will", "she's": "she is", "shouldn't": "should not",
        "that's": "that is", "there's": "there is", "they'd": "they would", "they'll": "they will",
        "they're": "they are", "they've": "they have", "we'd": "we would", "we're": "we are",
        "we've": "we have", "weren't": "were not", "what'll": "what will", "what're": "what are",
        "what's": "what is", "what've": "what have", "where'd": "where did", "where's": "where is",
        "who'll": "who will", "who's": "who is", "won't": "will not", "wouldn't": "would not",
        "you'd": "you would", "you'll": "you will", "you're": "you are", "you've": "you have"
    }
    words = text.split()
    words = [contractions[word] if word in contractions else word for word in words]
    text = ' '.join(words)


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
        "\U0001F900-\U0001F9FF"  # supplemental symbols and pictographs
        "\U0001FA00-\U0001FA6F"  # Chess Symbols (less common for emojis)
        "\U00002600-\U000026FF"  # Miscellaneous Symbols
        "\U00002500-\U000025FF"  # Geometric Shapes
        "]+", flags=re.UNICODE
    )
    text = emoji_pattern.sub(r'', text)

    # Basic slang/acronym replacement (can be expanded)
    slang_replacements = {
        'lol': 'laughing out loud', 'brb': 'be right back', 'lmao': 'laughing my ass off',
        'rofl': 'rolling on floor laughing', 'aka': 'also known as', 'gtfo': 'get out',
        'smh': 'shaking my head', 'ikr': 'i know right', 'idk': 'i do not know',
        'ttyl': 'talk to you later', 'afk': 'away from keyboard', 'irl': 'in real life',
        'fomo': 'fear of missing out', 'ftw': 'for the win', 'imo': 'in my opinion',
        'omg': 'oh my god', 'wtf': 'what the heck', 'nvm': 'never mind',
        'rn': 'right now', 'tmi': 'too much information', 'tbt': 'throwback thursday',
        'yolo': 'you only live once', 'bff': 'best friends forever', 'thx': 'thanks',
        'np': 'no problem', 'pls': 'please', 'ur': 'you are', 'r': 'are', 'u': 'you',
        'cuz': 'because', 'fr': 'for real', 'wyd': 'what you doing', 'hmu': 'hit me up'
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
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_words = [word for word in words if word not in stop_words]
    
    # Apply Lemmatization (replaces stemming)
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_words]

    # Remove extra whitespaces (e.g., multiple spaces into a single space)
    return re.sub(r'\s+', ' ', ' '.join(lemmatized_words)).strip()

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
    predict_cyberbullying("Are you feeling okay today? Anything bothering you?", trained_model, tfidf_vectorizer) # New neutral
    predict_cyberbullying("Your outfit is hideous, you look like a clown.", trained_model, tfidf_vectorizer) # New bullying
    predict_cyberbullying("Just enjoyed a nice walk in the park with my dog.", trained_model, tfidf_vectorizer) # New neutral
    predict_cyberbullying("You're a disgrace to your family. Everyone says so.", trained_model, tfidf_vectorizer) # New bullying