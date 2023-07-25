import mlflow
import mlflow.sklearn
from sklearn.metrics.pairwise import cosine_similarity

mlflow.set_tracking_uri("http://127.0.0.1:5000")

def predict_similarity(sentence1, sentence2, model_path):
    # Load the trained similarity model
    loaded_model = mlflow.sklearn.load_model(model_path)

    # Preprocess the input sentences (Assuming the same preprocessing as during training)
    preprocessed_sentence1 = preprocess_sentence(sentence1)
    preprocessed_sentence2 = preprocess_sentence(sentence2)

    # Calculate TF-IDF vectors for the input sentences
    tfidf_vectorizer = loaded_model.named_steps['tfidf']
    tfidf_matrix = tfidf_vectorizer.transform([preprocessed_sentence1, preprocessed_sentence2])

    return cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0]

def preprocess_sentence(sentence):
    # Implement your preprocessing steps here (e.g., lowercasing, tokenization, removing stopwords)
    # Make sure to use the same preprocessing as during training.
    # For simplicity, we'll assume no preprocessing is needed in this example.
    return sentence

if __name__ == "__main__":
    # Example input sentences
    sentence1 = "The quick brown fox jumps over the lazy dog."
    sentence2 = "A lazy dog is lying under a tree."

    # Specify the path to the logged similarity model (change to your specific model path)
    model_path = "runs:/88a0f993536a4ca6a449b5ccec8cb857/similarity_model"

    # Predict the similarity score between the input sentences
    similarity_score = predict_similarity(sentence1, sentence2, model_path)
    print(f"Similarity Score: {similarity_score:.4f}")
