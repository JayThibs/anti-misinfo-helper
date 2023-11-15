import pandas as pd
import nltk
from nltk.stem import PorterStemmer

# Download necessary NLTK data and initialize stemmer
nltk.download("punkt")
stemmer = PorterStemmer()


def load_keywords(file_path: str) -> list:
    try:
        return pd.read_csv(file_path)["Keyword"].tolist()
    except KeyError:
        print(f"Column 'keyword' not found in {file_path}")
        return []
