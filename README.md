# 🍽️ Zomato Sentiment Analysis & Review Clustering

## 📌 Project Overview

Online food delivery platforms like **Zomato** receive thousands of customer reviews daily. These reviews contain valuable insights about food quality, service experience, delivery performance, and overall customer satisfaction. However, manually analyzing large volumes of textual feedback is inefficient and time-consuming.

This project builds an **end-to-end Machine Learning system** that automatically:

* Cleans and preprocesses customer reviews
* Classifies reviews into **Positive or Negative sentiment**
* Applies **Clustering (K-Means)** to identify hidden review patterns
* Visualizes insights for business decision-making
* Provides a **deployment-ready sentiment prediction function**

---

## 🎯 Problem Statement

To build a machine learning system that automatically analyzes Zomato restaurant reviews and classifies them into positive or negative sentiments using NLP techniques, while also identifying hidden patterns through clustering to support data-driven business decisions.

---

## 🛠️ Technologies Used

* **Python**
* **Pandas & NumPy**
* **Matplotlib & Seaborn**
* **NLTK**
* **Scikit-learn**
* **TF-IDF Vectorization**
* **Logistic Regression**
* **Naive Bayes**
* **Random Forest**
* **K-Means Clustering**
* **PCA (Dimensionality Reduction)**
* **Joblib (Model Saving)**

---

## 🔄 Project Workflow

### 1️⃣ Data Cleaning & Wrangling

* Removed duplicates
* Handled missing values
* Extracted numeric ratings
* Created sentiment labels
* Generated new features (Review Length)

### 2️⃣ Text Preprocessing (NLP)

* Lowercasing
* Removing punctuation, URLs, numbers
* Stopword removal
* Tokenization
* Lemmatization
* TF-IDF Vectorization

### 3️⃣ Machine Learning Models

Three models were implemented and compared:

| Model               | Purpose                       |
| ------------------- | ----------------------------- |
| Logistic Regression | Primary classification model  |
| Naive Bayes         | Probabilistic text classifier |
| Random Forest       | Ensemble model                |

The models were evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Cross-validation

After tuning, **Logistic Regression performed best** and was selected as the final model.

---

## 📊 Clustering (Unsupervised Learning)

* Applied **K-Means Clustering**
* Used **PCA** for visualization
* Identified natural grouping of customer feedback
* Enabled review segmentation for deeper business insights

---

## 📈 Business Impact

This system enables businesses to:

* Automatically monitor customer sentiment
* Detect dissatisfied customers early
* Identify recurring complaints
* Improve service quality
* Enhance customer retention
* Make data-driven decisions

Recall and F1-score were prioritized to ensure negative reviews are not missed.

---

## 🚀 Deployment Ready

* Final model saved using **Joblib**
* Includes a prediction function:

```python
predict_sentiment("The food was amazing!")
```

* Notebook runs fully without errors (one-go execution)
* Production-ready structure

---

## 📂 How to Run

```bash
1. Clone the repository
2. Install dependencies:
   pip install -r requirements.txt
3. Run the Jupyter Notebook
4. Use the prediction function to test new reviews
```

---

## 📦 Required Dependencies

```txt
numpy
pandas
matplotlib
seaborn
scikit-learn
nltk
wordcloud
joblib
```

---

## 🧠 Key Highlights

✔ End-to-End ML Pipeline
✔ NLP Preprocessing
✔ Model Comparison
✔ Hyperparameter Tuning
✔ Cross-Validation
✔ Clustering Integrated
✔ Business-Focused Evaluation
✔ Deployment Ready

---

## 👨‍💻 Author

Ponna Chaitanya
Machine Learning & NLP Project

---

# ⭐ Project Level

Final Year / Major Project
Domain: Machine Learning & Natural Language Processing
