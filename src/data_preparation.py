import pandas as pd
import nltk
from nltk.stem import PorterStemmer

# Download necessary NLTK data and initialize stemmer
nltk.download("punkt")
stemmer = PorterStemmer()


def load_keywords(file_path: str) -> list:
    return pd.read_csv(file_path)["keyword"].tolist()
