🍽️ Zomato Sentiment Analysis with Clustering
📌 Project Overview

Online food delivery platforms such as Zomato receive thousands of customer reviews daily. These reviews contain valuable feedback regarding food quality, service, and overall customer experience. Manually analyzing such large volumes of textual data is inefficient and time-consuming.

This project automates the process of sentiment analysis using Natural Language Processing (NLP) and Machine Learning, while also integrating Unsupervised Learning (Clustering) to discover hidden patterns in customer reviews.

The system classifies restaurant reviews into Positive and Negative sentiments and groups similar reviews using clustering techniques.

🎯 Objectives

Perform Exploratory Data Analysis (EDA) on review data

Clean and preprocess textual data

Convert text into numerical features using TF-IDF

Build a sentiment classification model

Apply K-Means clustering to discover review patterns

Evaluate model and clustering performance

Implement a live sentiment prediction system

🛠️ Technologies Used

Python

Pandas & NumPy

Matplotlib & Seaborn

NLTK

Scikit-learn

TF-IDF Vectorization

Logistic Regression

K-Means Clustering

PCA (Principal Component Analysis)

📊 Project Workflow
1️⃣ Data Preprocessing

Handled missing values

Cleaned inconsistent rating formats

Converted ratings into sentiment labels

Removed punctuation, numbers, and stopwords

Applied lemmatization

2️⃣ Feature Engineering

Converted text into numerical format using TF-IDF Vectorization

3️⃣ Sentiment Classification

Used Logistic Regression

Split dataset into training and testing sets

Achieved ~87% accuracy

Evaluated using:

Accuracy

Precision

Recall

F1-Score

Confusion Matrix

4️⃣ Clustering (Unsupervised Learning)

Applied K-Means Clustering

Used Elbow Method to find optimal number of clusters

Calculated Silhouette Score for cluster validation

Used PCA for 2D cluster visualization

5️⃣ Live Prediction

Implemented a function that predicts sentiment for new user-input reviews.

📈 Results

Logistic Regression achieved strong classification performance (~87% accuracy).

Clustering revealed natural groupings of similar reviews.

PCA visualization provided clear cluster separation.

The system is deployment-ready for real-time sentiment analysis.

💡 Real-World Applications

Restaurant reputation monitoring

Customer satisfaction analysis

Business intelligence dashboards

Market research

Automated feedback systems

🚀 Future Improvements

Implement Deep Learning models (LSTM / BERT)

Hyperparameter tuning

Deploy using Streamlit or Flask

Integrate real-time review scraping

📂 How to Run
1. Clone the repository
2. Install required dependencies
3. Run the Python script
4. Use the prediction function to test new reviews

👨‍💻 Author

ponna chaitanya
Machine Learning & NLP Project
