# 🛡️ Cyberbullying Detection System

An end-to-end Natural Language Processing (NLP) and Machine Learning framework designed to automatically detect, classify, and mitigate cyberbullying and toxic behavior across online platforms and social media feeds.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Framework-Flask_/_FastAPI-lightgrey.svg?style=for-the-badge&logo=flask)](https://flask.palletsprojects.com/)
[![Machine Learning](https://img.shields.io/badge/ML%2FDL-Scikit--learn%20%2F%20TensorFlow-orange.svg?style=for-the-badge&logo=tensorflow)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

---

## 📌 Overview

With the rapid expansion of digital social ecosystems, online harassment and cyberbullying have become pervasive threats to digital well-being. This project leverages state-of-the-art text classification methodologies and pipeline workflows to analyze incoming textual streams, evaluate sentiment nuances, and flag harmful interactions in real time. 

The architecture handles everything from raw dataset text preprocessing (tokenization, lemmatization, stop-word filtering) to vector embeddings and classification evaluation.

---

## ⚡ Key Features

- **Text Clean-up Pipeline**: Automated text standardization filtering out URLs, special characters, emojis, and HTML tags.
- **Multimodal Word Embeddings**: Supports multiple text representations including TF-IDF vectorization and custom word embeddings.
- **Robust Model Classifier**: Utilizes high-precision classification models (e.g., Logistic Regression, Naive Bayes, or deep recurrent architectures like LSTM) optimized for imbalanced text datasets.
- **Interactive UI Dashboard**: Includes a user-friendly web interface built with a lightweight Python web framework (Flask/FastAPI) to handle text classification on the fly.
- **CSV Batch Analysis**: Supports uploading larger data logs (such as CSV exports) to execute broad content moderation sweeps.

---

## 🛠️ Architecture & Tech Stack

- **Language**: Python 3.8+
- **NLP Libraries**: `NLTK`, `Gensim`, `Scikit-learn`
- **Deep Learning Frameworks**: `TensorFlow` / `Keras`
- **Backend Deployment**: `Flask` / `FastAPI`
- **Data Manipulation**: `Pandas`, `NumPy`

---

## 🚀 Getting Started

### 📋 Prerequisites

Ensure you have Python installed alongside package management tools like `pip` or `Anaconda`.

```bash
# Verify installations
python --version
pip --version

```

### ⚙️ Installation & Setup

1. **Clone the Repository**:
```bash
git clone [https://github.com/meetpotdar777/Cyberbullying-Detection.git](https://github.com/meetpotdar777/Cyberbullying-Detection.git)
cd Cyberbullying-Detection

```


2. **Set Up a Virtual Environment (Recommended)**:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate

```


3. **Install Required Packages**:
```bash
pip install -r requirements.txt

```



### 🖥️ Running the Application

To launch the training module or run the interactive web inference app locally:

1. **Train/Evaluate the Model** (if updating your local parameters):
```bash
python train.py

```


2. **Boot Up the Web Interface**:
```bash
python app.py

```


3. Open your favorite web browser and navigate to the local host address provided in the terminal console output:
```text
[http://127.0.0.1:5000/](http://127.0.0.1:5000/)

```



---

## 📊 Evaluation & Metrics

The baseline models are optimized against standard NLP classification metrics to ensure minimal false positives while keeping recall rates high for severe instances of toxicity:

| Metric | Target Score |
| --- | --- |
| **Precision** | ~72%+ |
| **Recall** | ~69%+ |
| **F1-Score** | ~70%+ |

---

## 📂 Project Directory Structure

```text
├── datasets/             # Labeled datasets used for model training
├── models/               # Saved weights, serialization parameters, or custom embedding dictionaries
├── templates/            # Frontend layout HTML/CSS for web interface
├── app.py                # Main web application driver script
├── train.py              # Training script and model evaluation pipeline
├── requirements.txt      # List of dependencies
└── README.md             # Project documentation

```

---

## 🤝 Contributing

Contributions to improve model accuracy, optimize feature extraction pipelines, or enhance the dashboard UI are welcome!

1. **Fork** the project repository.
2. **Create** your feature branch (`git checkout -b feature/AmazingFeature`).
3. **Commit** your refinements (`git commit -m 'Add some AmazingFeature'`).
4. **Push** to the origin branch (`git push origin feature/AmazingFeature`).
5. Open a **Pull Request**.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---
