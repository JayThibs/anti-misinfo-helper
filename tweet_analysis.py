import os
import openai
from typing import List
from .fuzzy_matching import find_similar_words_preprocessed_modified

openai.api_key = os.getenv("OPENAI_API_KEY")
system_message_for_review = (
    "You are a highly advanced assistant tasked with evaluating tweets..."
)
system_message_for_assistance = (
    "You are an intelligent assistant specialized in aiding community note writers..."
)


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
