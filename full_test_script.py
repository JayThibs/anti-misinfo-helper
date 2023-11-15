import os
import openai
import numpy as np
import pandas as pd
from fuzzywuzzy import process
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from dotenv import load_dotenv
from typing import List, Tuple, Dict

# Set your OpenAI API key and system messages
openai.api_key = os.getenv("OPENAI_API_KEY")
system_message_for_review = "You are a highly advanced assistant tasked with evaluating tweets. Assess the content of the tweet and provide a score indicating the urgency of review. Use a nuanced approach to determine if the tweet definitely needs immediate review or does not need review."
system_message_for_assistance = "You are an intelligent assistant specialized in aiding community note writers. Synthesize accurate and relevant context from reliable sources for the tweet. The note should be clear, neutral, well-sourced, and directly address the content of the tweet, providing essential information."

# Download necessary NLTK data and initialize stemmer
nltk.download("punkt")
stemmer = PorterStemmer()

# Load keywords
keywords = pd.read_csv("keywords.csv")["keyword"].tolist()
contentious_issues = pd.read_csv("contentious_issues.csv")["issue"].tolist()


def preprocess_tweet_modified(tweet):
    """
    This function preprocesses the tweet by manually tokenizing and stemming the words.
    """
    # Manually tokenize the tweet by splitting on spaces
    words = tweet.split()

    # Stem each word
    processed_words = [stemmer.stem(word.lower()) for word in words]

    return processed_words


def find_similar_words_preprocessed_modified(tweet, word_list, threshold=80):
    """
    This function preprocesses a tweet and then finds similar words from the word_list.
    """
    # Preprocess the tweet
    processed_words = preprocess_tweet_modified(tweet)

    # Find similar words using fuzzy matching
    matched_words = []

    for word in processed_words:
        best_match = process.extractOne(word, word_list)

        if best_match and best_match[1] >= threshold:
            matched_words.append(best_match[0])

    return matched_words


def get_top_liked_tweets(tweets, y) -> List[Dict]:
    """
    Retrieve the top 'y' most-liked tweets.

    Args:
    tweets (List[Dict]): A list of tweet dictionaries with 'likes' and other metadata.
    y (int): Number of top tweets to retrieve based on likes.

    Returns:
    List[Dict]: A list of the top 'y' most-liked tweets.
    """
    # Sort tweets based on the number of likes
    sorted_tweets = sorted(tweets, key=lambda tweet: tweet["likes"], reverse=True)

    # Return the top 'y' tweets
    return sorted_tweets[:y]


def contains_keywords(tweet: str, keywords: List[str]) -> bool:
    """
    Check if a tweet contains any of the specified keywords.

    Args:
    tweet (str): The tweet text to be analyzed.
    keywords (List[str]): A list of keywords to check against the tweet.

    Returns:
    bool: True if the tweet contains any of the keywords, False otherwise.
    """
    return any(keyword in tweet.lower() for keyword in keywords)


def get_text_embedding(text: str) -> np.ndarray:
    """
    Get the semantic embedding for a given text using OpenAI's model.

    Args:
    text (str): The text to get the embedding for.

    Returns:
    np.ndarray: The embedding vector for the given text.
    """
    response = openai.Embedding.create(
        model="text-similarity-babbage-001", input=[text]
    )
    return np.array(response["data"][0]["embedding"])


def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate the cosine similarity between two vectors.

    Args:
    vec1 (np.ndarray): The first vector.
    vec2 (np.ndarray): The second vector.

    Returns:
    float: The cosine similarity between vec1 and vec2.
    """
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def rank_importance(
    tweet_embedding: np.ndarray, issue_embeddings: List[np.ndarray]
) -> float:
    """
    Rank the importance of a tweet based on its similarity to contentious issues.

    Args:
    tweet_embedding (np.ndarray): The embedding of the tweet.
    issue_embeddings (List[np.ndarray]): A list of embeddings for various contentious issues.

    Returns:
    float: The highest similarity score between the tweet and contentious issues.
    """
    similarities = [
        cosine_similarity(tweet_embedding, issue_emb) for issue_emb in issue_embeddings
    ]
    return max(similarities)


def process_tweet(tweet: str, keywords: list, contentious_issues: list) -> float:
    """
    Process a tweet to determine its importance based on predefined keywords and contentious issues.

    Args:
    tweet (str): The tweet text to be processed.

    Returns:
    float: An importance score for the tweet. A score of 0 implies the tweet is not relevant to the specified keywords.
    """
    if contains_keywords(tweet, keywords):
        tweet_embedding = get_text_embedding(tweet)
        issue_embeddings = [get_text_embedding(issue) for issue in contentious_issues]
        importance_score = rank_importance(tweet_embedding, issue_embeddings)
        return importance_score
    else:
        return 0


def analyze_tweet_with_lm(tweet: str, model: str, system_message_content: str) -> str:
    """
    Analyze a tweet using a specified language model.

    Args:
    tweet (str): The text of the tweet.
    model (str): The model to be used for analysis ("gpt-3.5-turbo" or "gpt-4-turbo").
    system_message_content (str): The content of the system message to guide the language model.

    Returns:
    str: Analysis result or a note synthesized by the language model.
    """
    client = openai.OpenAI(api_key="YOUR_OPENAI_API_KEY")

    # System message guiding the language model
    system_message = {"role": "system", "content": system_message_content}

    # User prompt with the tweet
    user_message = {"role": "user", "content": tweet}

    # Create a chat completion with the OpenAI client
    response = client.chat.completions.create(
        model=model, messages=[system_message, user_message]
    )

    return response.choices[0].message["content"]


def analyze_tweet(tweet: str, likes: int, threshold: int, sensitive_words: List[str]):
    """
    Analyze a tweet using GPT-3.5 if it has more than a specified number of likes and contains sensitive words.

    Args:
    tweet (str): The text of the tweet.
    likes (int): The number of likes the tweet has.
    threshold (int): The like threshold for adding the tweet to the GPT-3.5 queue.
    sensitive_words (List[str]): List of sensitive words for fuzzy matching.

    Returns:
    str: The analysis result or a status message.
    """
    # Check for sensitive words using fuzzy matching
    matched_words = find_similar_words_preprocessed_modified(tweet, sensitive_words)

    # Check if the tweet meets the like threshold and contains sensitive words
    if likes >= threshold and matched_words:
        # Add to GPT-3.5 queue for analysis
        system_message_content = system_message_for_review
        return analyze_tweet_with_lm(tweet, "gpt-3.5-turbo", system_message_content)
    elif not matched_words:
        return "Tweet does not contain relevant sensitive words for analysis."
    else:
        return "Tweet does not meet the like threshold for analysis."


def assist_with_community_note(tweet: str, use_gpt4: bool) -> str:
    """
    Provide assistance for writing a community note, optionally using GPT-4.
    """
    model = "gpt-4-turbo" if use_gpt4 else "gpt-3.5-turbo"
    return analyze_tweet_with_lm(tweet, model, system_message_for_assistance)


# Assuming tweet content and like count
tweet = "Example tweet content."
likes = 150

# Analyze the tweet for misinformation
analysis_result = analyze_tweet(
    tweet, likes, threshold=100, sensitive_words=sensitive_words
)
print("Analysis Result:", analysis_result)

# If the tweet likely contains misinformation, assist in writing a community note
if "needs review" in analysis_result:
    use_gpt4_for_assistance = likes >= 10000
    note_assistance = assist_with_community_note(tweet, use_gpt4_for_assistance)
    print("Assisted Community Note:", note_assistance)
