from typing import List
from fuzzywuzzy import process
from .data_preparation import stemmer


def preprocess_tweet_modified(tweet: str) -> List[str]:
    words = tweet.split()
    return [stemmer.stem(word.lower()) for word in words]


def find_similar_words_preprocessed_modified(
    tweet: str, word_list: List[str], threshold: int = 80
) -> List[str]:
    processed_words = preprocess_tweet_modified(tweet)
    return [
        match[0]
        for word in processed_words
        if (match := process.extractOne(word, word_list)) and match[1] >= threshold
    ]
