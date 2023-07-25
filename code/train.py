import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn

import mlflow

# Set the MLflow tracking URI to the correct endpoint
mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Change this to your MLflow server URI

def train_model(input_file):
    # Read the preprocessed data from CSV
    df = pd.read_csv(input_file)

    # Prepare the data for model training
    X1 = df['preprocessed_sentence1']
    X2 = df['preprocessed_sentence2']
    y = np.ones(len(X1))  # Assume all pairs are similar (for demonstration purposes)

    # Split the data into training and testing sets
    X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(X1, X2, y, test_size=0.2, random_state=42)

    # Build the cosine similarity model using TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    X1_train_tfidf = tfidf_vectorizer.fit_transform(X1_train)
    X2_train_tfidf = tfidf_vectorizer.transform(X2_train)

    similarity_model = Pipeline([
        ('tfidf', tfidf_vectorizer),
        ('cosine_similarity', cosine_similarity)
    ])

    # Train the model (Note: For this example, the model doesn't require training)
    trained_model = similarity_model

    # Log hyperparameters and metrics with MLflow
    with mlflow.start_run():
        learning_rate = 0.001
        mlflow.log_param("learning_rate", learning_rate)

        # Log a dummy accuracy metric (for demonstration purposes)
        accuracy = 0.85
        mlflow.log_metric("accuracy", accuracy)

        # Save the trained model as an MLflow artifact
        mlflow.sklearn.log_model(trained_model, "similarity_model")

if __name__ == "__main__":
    input_file = "./data/processed/sentences_preprocessed.csv"
    train_model(input_file)
