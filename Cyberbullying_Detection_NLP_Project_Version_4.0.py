import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib # For saving/loading the model and vectorizer
import os
import random # For generating more diverse dummy data

# --- Conceptual Imports for Advanced Understanding (Uncomment if using in a full environment) ---
# For pre-trained word embeddings (e.g., Word2Vec, GloVe)
# import gensim.models # For Word2Vec, FastText
# from gensim.models import KeyedVectors # To load pre-trained vectors

# For deep learning models (e.g., LSTMs, Transformers)
# import tensorflow as tf
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout

# For advanced Transformer-based models (like BERT, RoBERTa, etc.)
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# import torch # if using PyTorch


# Ensure NLTK data is downloaded (run once)
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    print("NLTK stopwords downloaded.")
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    print("NLTK punkt tokenizer downloaded.")
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')
    print("NLTK WordNet downloaded for lemmatization.")


# --- Configuration for persistence ---
MODEL_FILE = 'cyberbullying_model.joblib'
VECTORIZER_FILE = 'tfidf_vectorizer.joblib'

# --- 1. Data Collection ---

# Option 1: Load a real-world dataset (RECOMMENDED for better performance)
# In a real project, you would uncomment and use this function.
# Look for cyberbullying datasets on Kaggle (e.g., "Cyberbullying Tweet Dataset").
def load_real_data_from_csv(file_path, text_column='tweet_text', label_column='cyberbullying_label'):
    """
    Loads a real-world dataset from a CSV file.
    Assumes the CSV has a 'text_column' for the content and a 'label_column' (0 for not bullying, 1 for bullying).
    Adjust column names as per your dataset.

    Args:
        file_path (str): The path to your CSV dataset file.
        text_column (str): The name of the column containing the text.
        label_column (str): The name of the column containing the labels (0 or 1).

    Returns:
        pd.DataFrame: DataFrame with 'text' and 'is_cyberbullying' columns, or None if file not found.
    """
    if not os.path.exists(file_path):
        print(f"Error: Dataset file not found at '{file_path}'. Please provide a valid path.")
        return None
    try:
        df = pd.read_csv(file_path)
        df = df[[text_column, label_column]].rename(columns={text_column: 'text', label_column: 'is_cyberbullying'})
        # Ensure labels are binary (0 or 1) and text is string
        df['is_cyberbullying'] = df['is_cyberbullying'].astype(int)
        df['text'] = df['text'].astype(str)
        print(f"Real dataset loaded successfully from {file_path}.")
        print(f"Dataset shape: {df.shape}")
        print(f"Label distribution:\n{df['is_cyberbullying'].value_counts()}")
        return df
    except Exception as e:
        print(f"Error loading real dataset from CSV: {e}")
        return None

# Option 2: Generate a larger, synthetic dummy dataset (for demonstration without a real file)
def generate_synthetic_data(num_samples=500):
    """
    Generates a larger, synthetic dummy dataset for demonstration purposes.
    This simulates having more data, but real-world data is always preferred.
    """
    bullying_templates = [
        "You are so {} and {}, no one likes you!",
        "Stop bothering me, you're a {}.",
        "I will make your life a living {}.",
        "You're {} and deserve nothing good.",
        "Why are you always so {}? Go away!",
        "You should just disappear, nobody cares about you {}.",
        "Your ideas are {}.",
        "I hate your {}, you're a waste of space.",
        "haha u r so {} lol",
        "Shut up, you deserve all the bad things. 😠",
        "gtfo, nobody wants you here {}",
        "You're the worst person ever smh {}",
        "Did you hear the rumor about them? It's really {}.",
        "I'm going to tell everyone your secret if you don't do what I say.",
        "You are a total {}, everyone knows it now.",
        "Your {} are so {}, nobody wants to hang out with you.",
        "Don't even try, you'll {} anyway. ",
        "I heard you cheated on the test. Everyone's talking about it {}.",
        "Your mom is so {}, lol. Everyone laughs at her.",
        "Heard you got {}. Couldn't happen to a nicer person. #karma",
        "Why do you even bother posting? You're terrible at everything. @loser {}",
        "I will hunt you down.",
        "You're a {}. Stop embarrassing yourself. #pathetic {}",
        "You'll never amount to anything. Just {} up already.",
        "This is the worst project ever. You clearly have no idea what you're doing. 😂",
        "You deserve all the hate you get. Go away! {}",
        "Heard you failed the exam. Not surprised given how {} you are.",
        "You're such a {}. Stay away from me.",
        "Your {} doesn't matter. Just shut up.",
        "I'm going to {} your reputation online.",
        "You're just jealous because you'll never be as good as me."
    ]
    bullying_words = {
        'adj': ['stupid', 'ugly', 'loser', 'pathetic', 'annoying', 'worthless', 'dumb', 'fat', 'hideous', 'lazy', 'creep'],
        'noun': ['loser', 'hell', 'guts', 'fraud', 'clown', 'disgrace'],
        'verb': ['disappear', 'fail', 'quit', 'ruin'],
        'emoji': ['😠', '😡', '😂'],
        'filler': ['lol', 'smh']
    }

    non_bullying_templates = [
        "Having a {} day with friends!",
        "The weather is {} today.",
        "Let's grab some coffee {}.",
        "Just finished my {}.",
        "What a {} sunset tonight.",
        "Enjoying a quiet evening at {}.",
        "Thinking about a new {}.",
        "This is amazing! 🎉 So happy for you!",
        "I'm feeling great 😊",
        "Just had a {} time at the park.",
        "Reading a good book right now. 📚",
        "This game is so much {}!",
        "Having a wonderful day with my family. So blessed! 🙏",
        "Just chilling at home, watching movies. 🎬🍿",
        "Can't believe how rude some people are online. It's disgusting. 😡",
        "I totally agree with what you said on Twitter! So insightful. �",
        "Learning new Python skills today! Feeling {}. #coding",
        "Hope you have a fantastic weekend! See you soon. 👋",
        "Such a {} sunset! Feeling very peaceful. 🌄",
        "I'm so excited for my vacation next month! ✈️🏖️",
        "What's up, fam? vibin' hard today. #goodvibes",
        "Working hard on my new coding challenge. Wish me luck!",
        "Enjoying a quiet coffee this morning. ☕",
        "Always happy to help out where I can! 😊"
    ]
    non_bullying_words = {
        'adj': ['great', 'nice', 'beautiful', 'lovely', 'good', 'happy', 'productive', 'fantastic', 'peaceful'],
        'time': ['later', 'soon', 'tonight', 'morning'],
        'noun': ['workout', 'sunset', 'home', 'project', 'time', 'book', 'fun', 'family', 'weekend', 'vacation', 'coffee'],
        'emoji': ['😊', '🎉', '🙏', '🎬', '🍿', '👍', '👋', '🌄', '✈️', '🏖️', '☕'],
        'filler': ['so', 'very']
    }

    texts = []
    labels = []

    # Generate bullying samples
    for _ in range(num_samples // 2):
        template = random.choice(bullying_templates)
        if '{}' in template:
            filled_template = template.format(
                random.choice(bullying_words.get('adj', [''])),
                random.choice(bullying_words.get('noun', [''])),
                random.choice(bullying_words.get('verb', [''])),
                random.choice(bullying_words.get('emoji', [''])),
                random.choice(bullying_words.get('filler', [''])),
                random.choice(bullying_words.get('noun', [''])), # for {} in "gtfo, nobody wants you here {}"
                random.choice(bullying_words.get('filler', [''])), # for {} in "You're the worst person ever smh {}"
                random.choice(bullying_words.get('adj', [''])), # for {} in "Did you hear the rumor about them? It's really {}."
                random.choice(bullying_words.get('noun', [''])), # for {} in "You are a total {}, everyone knows it now."
                random.choice(bullying_words.get('verb', [''])), # for {} in "Don't even try, you'll {} anyway."
                random.choice(bullying_words.get('filler', [''])), # for {} in "I heard you cheated on the test. Everyone's talking about it {}."
                random.choice(bullying_words.get('adj', [''])), # for {} in "Your mom is so {}, lol. Everyone laughs at her."
                random.choice(bullying_words.get('verb', [''])), # for {} in "Heard you got {}. Couldn't happen to a nicer person. #karma"
                random.choice(bullying_words.get('filler', [''])), # for {} in "Why do you even bother posting? You're terrible at everything. @loser {}"
                random.choice(bullying_words.get('emoji', [''])), # for {} in "You're a clown. Stop embarrassing yourself. #pathetic {}"
                random.choice(bullying_words.get('verb', [''])), # for {} in "You'll never amount to anything. Just {} up already."
                random.choice(bullying_words.get('emoji', [''])), # for {} in "You deserve all the hate you get. Go away! {}"
                random.choice(bullying_words.get('adj', [''])), # for {} in "Heard you failed the exam. Not surprised given how {} you are."
                random.choice(bullying_words.get('adj', [''])), # for {} in "You're such a {}. Stay away from me."
                random.choice(bullying_words.get('noun', [''])), # for {} in "Your {} doesn't matter. Just shut up."
                random.choice(bullying_words.get('verb', [''])) # for {} in "I'm going to {} your reputation online."
            )
            # Simple cleanup for potential multiple spaces from empty random choices
            filled_template = re.sub(r'\s+', ' ', filled_template).strip()
            texts.append(filled_template)
        else:
            texts.append(template) # Template without placeholders
        labels.append(1)

    # Generate non-bullying samples
    for _ in range(num_samples // 2):
        template = random.choice(non_bullying_templates)
        if '{}' in template:
            filled_template = template.format(
                random.choice(non_bullying_words.get('adj', [''])),
                random.choice(non_bullying_words.get('time', [''])),
                random.choice(non_bullying_words.get('noun', [''])),
                random.choice(non_bullying_words.get('emoji', [''])),
                random.choice(non_bullying_words.get('filler', ['']))
            )
            filled_template = re.sub(r'\s+', ' ', filled_template).strip()
            texts.append(filled_template)
        else:
            texts.append(template) # Template without placeholders
        labels.append(0)

    # Shuffle the data
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)

    df = pd.DataFrame({'text': list(texts), 'is_cyberbullying': list(labels)})
    print(f"Synthetic dataset generated with {len(df)} samples.")
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

    # --- Data Loading Strategy ---
    # To use a real dataset, uncomment the 'real_data_path' and 'df = load_real_data_from_csv(...)' lines
    # and comment out 'df = generate_synthetic_data()'.
    # Ensure your CSV has a 'text' column and 'is_cyberbullying' column (or adjust column names in the function).
    # real_data_path = 'path/to/your/cyberbullying_dataset.csv' # <<<--- IMPORTANT: Update this path!
    # df = load_real_data_from_csv(real_data_path, text_column='tweet_text', label_column='cyberbullying_label')

    # For demonstration, we use a larger synthetic dataset if no real data is loaded/found
    # If df is None (e.g., real data path was invalid or file not found), then generate synthetic data.
    # Otherwise, assume df was loaded from real data.
    # Set num_samples to a higher value like 500, 1000, or more for better simulation
    df = generate_synthetic_data(num_samples=500) # Increased dummy data size for better simulation

    if df is None or df.empty:
        print("No data available to train or predict after attempting to load real or generate synthetic data. Exiting.")
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
        # Using stratify=y to ensure a balanced split in terms of bullying/non-bullying samples
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
        # Added zero_division=0 to suppress warning about 0 precision/recall for minority classes
        print("Classification Report:")
        print(classification_report(y_test, y_pred, zero_division=0)) 
        print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        print("------------------------\n")
    else:
        print("\nSkipping model training as an existing model was loaded.")
        # If a model was loaded, we can still evaluate it on the full data or new test data if available.
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

    # Example predictions (can be customized)
    print("\n--- Example Predictions on New Text ---")
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
    predict_cyberbullying("Are you feeling okay today? Anything bothering you?", trained_model, tfidf_vectorizer)
    predict_cyberbullying("Your outfit is hideous, you look like a clown.", trained_model, tfidf_vectorizer)
    predict_cyberbullying("Just enjoyed a nice walk in the park with my dog.", trained_model, tfidf_vectorizer)
    predict_cyberbullying("You're a disgrace to your family. Everyone says so.", trained_model, tfidf_vectorizer)