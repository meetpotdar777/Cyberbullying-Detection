# 🛡️ Cyberbullying Detection System

An iterative, production-scalable Natural Language Processing (NLP) framework engineered to classify and mitigate toxic text patterns and online harassment. The repository tracks structural engineering advancements from raw baseline tokenization up to automated synthetic text factories and modular data adapters.

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg?style=for-the-badge&logo=python)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML%2FDL-Scikit--learn%20%2F%20NLTK-orange.svg?style=for-the-badge&logo=scikit-learn)](https://scikit-learn.org/)
[![NLP - NLTK](https://img.shields.io/badge/NLP-NLTK-green?style=for-the-badge&logo=analyzer)](https://www.nltk.org/)
[![Data Pipeline - Pandas](https://img.shields.io/badge/Pipeline-Pandas-150458?style=for-the-badge&logo=pandas)](https://pandas.pydata.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](https://opensource.org/licenses/MIT)

---

## 📌 Engine Version Roadmap

The architecture scales through four progressive operational phases:

### 🔹 NLP Version 1.0 (Baseline Tokenizer)
* Sets fallback hooks downloading fundamental NLTK assets (`stopwords`, `porter`, `punkt`).
* Alphanumeric text stripping, lower-case balancing, and crude stop-word pruning.
* Baseline representation via single-layer **TF-IDF Vectorization** routed to a **Logistic Regression** classifier.

### 🔹 NLP Version 2.0 (Slang Expansion Engine)
* Appends custom Unicode regex filters isolating complex social web pictographs/emojis.
* Map-transforms common internet acronyms and toxic shorthand strings (e.g., `lol`, `gtfo`, `smh`) to clear semantic forms.

### 🔹 NLP Version 3.0 (Semantic Normalization)
* Upgrades text stems to **WordNet Lemmatization** pipelines to maintain contextual baseline dictionary structures.
* Strips explicit hyper-links (`http`/`https`), handle signatures (`@username`), and trailing index queries (`#hashtag`).
* Map-expands structural contraction formats (e.g., `can't've` $\rightarrow$ `cannot have`).

### 🔹 NLP Version 4.0 (Synthetic Factories & Data Ingestion Adapters)
* **Real-World CSV Loader (`load_real_data_from_csv`)**: Integrates production data loaders featuring strict column restructuring, binary target classification alignment, and runtime file safety diagnostics.
* **Algorithmic Text Factory (`generate_synthetic_data`)**: Implements parametric string synthesis templates to dynamically construct balanced mock matrices (500+ items) combining adjectives, action verbs, and aggressive internet colloquialisms.
* **Conceptual Blueprints**: Outlines immediate architectural staging configurations for Deep Learning migrations including Distributed Static Embeddings (`Gensim Word2Vec`), Recurrent Networks (`TensorFlow LSTM`), and Contextual Transformers (`HuggingFace BERT`).

---

## 🛠️ Tech Stack & Dependencies

- **Core Language Engine**: Python 3.8+
- **Data Architecture Tools**: `Pandas`, `NumPy`
- **Text Processing & ML Frameworks**: `Scikit-learn`, `NLTK`, `Joblib`
- **Underlying Conceptual Models**: Logistic Regression, Term Frequency-Inverse Document Frequency (TF-IDF)

---

## 🛠️ Project Directory Layout

```text
├── .vscode/                     # Editor workplace workspace settings
├── Cyberbullying_Detection_NLP_Project_Version_1.0.py   # Baseline engine
├── Cyberbullying_Detection_NLP_Project_Version_2.0.py   # Slang expansion layer
├── Cyberbullying_Detection_NLP_Project_Version_3.0.py   # Lemmatizer optimization
├── Cyberbullying_Detection_NLP_Project_Version_4.0.py   # Synthetics & CSV data pipeline
├── cyberbullying_model.joblib   # Serialized Logistic Regression weights
└── tfidf_vectorizer.joblib      # Persisted TF-IDF vocabulary metrics

```

---

## 🚀 Getting Started

### 📋 Prerequisites

Verify your local environment has Python and package management managers configured:

```bash
python --version
pip --version

```

### ⚙️ Installation & Workspace Setup

1. **Clone the Repository**:
```bash
git clone [https://github.com/meetpotdar777/Cyberbullying-Detection.git](https://github.com/meetpotdar777/Cyberbullying-Detection.git)
cd Cyberbullying-Detection
```

2. **Set Up a Virtual Environment**:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate

```

3. **Install Package Requirements**:
```bash
pip install pandas numpy scikit-learn joblib nltk

```

---

## 🖥️ Running the Application Pipelines

Each script can execute standalone. The underlying codebase incorporates smart data checkpoint lookups to eliminate repetitive text vector re-computations.

```bash
# Execute the latest ingestion pipeline featuring synthetic template generation:
python Cyberbullying_Detection_NLP_Project_Version_4.0.py

```

---

### 💾 Model Serialization Strategy

Checkpointing operations prevent computational feature recalculation overhead across multiple execution processes:

```python
# Automated system checkpoint lookup logic
if os.path.exists(MODEL_FILE) and os.path.exists(VECTORIZER_FILE):
    trained_model = joblib.load(MODEL_FILE)
    tfidf_vectorizer = joblib.load(VECTORIZER_FILE)

```

---

## 📊 Pipeline Ingestion & Processing Workflow

1. **Ingestion Channel** $\rightarrow$ Triggers real-world CSV file imports or routes tasks to the `random`-seeded text array synthesis pool.
2. **Standardization Scrape** $\rightarrow$ Purges strict web links, active `@user` callouts, and clean-strips `#` tags.
3. **Lexical Mapping** $\rightarrow$ Runs dictionary passes expanding common contractions, normalizes net-slang, and strips targeted unicode emoji arrays.
4. **Semantic Root Normalization** $\rightarrow$ Tokenizes string characters into singular arrays and runs structural **WordNet Lemmatization** passes.
5. **Feature Mapping** $\rightarrow$ Vectorizes processed content via a 5,000 max-feature `TfidfVectorizer` sparse layout.
6. **Classification & Inference** $\rightarrow$ Trains stratified splits on Logistic Regression (using a calibrated $max\_iter=1000$ bound) and yields live prediction probability scores.

---

## 🤝 Contributing

Contributions to scale classification accuracy, append comprehensive slang dictionaries, or integrate advanced transformer layers are welcome!

1. **Fork** the project repository.
2. **Create** your feature branch (`git checkout -b feature/AmazingFeature`).
3. **Commit** your refinements (`git commit -m 'Add some AmazingFeature'`).
4. **Push** to the origin branch (`git push origin feature/AmazingFeature`).
5. Open a **Pull Request**.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.

---
