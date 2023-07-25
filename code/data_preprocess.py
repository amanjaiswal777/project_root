import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_sentence(sentence):
    # Lowercase the sentence
    sentence = sentence.lower()

    # Tokenize the sentence
    tokens = word_tokenize(sentence)

    # Remove punctuation and special characters
    table = str.maketrans('', '', string.punctuation)
    tokens = [token.translate(table) for token in tokens]

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]

    return ' '.join(tokens)

def preprocess_data(input_file, output_file):
    # Read the input CSV file containing sentence pairs
    df = pd.read_csv(input_file)

    # Preprocess the sentences and create new preprocessed columns
    df['preprocessed_sentence1'] = df['sentence1'].apply(preprocess_sentence)
    df['preprocessed_sentence2'] = df['sentence2'].apply(preprocess_sentence)

    # Save the preprocessed data to the output CSV file
    df.to_csv(output_file, index=False)

if __name__ == "__main__":
    input_file = "./data/raw/sentences.csv"
    output_file = "./data/processed/sentences_preprocessed.csv"
    preprocess_data(input_file, output_file)
