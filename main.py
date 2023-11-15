from src.tweet_analysis import analyze_tweet, assist_with_community_note
from src.data_preparation import load_keywords

# Load keywords and contentious issues
sensitive_words = load_keywords("data/keywords.csv")
contentious_issues = load_keywords("data/contentious_issues.csv")

tweets = ["I hate the new update!", "I love the new update!", "QAnon is a true."]
likes = [100, 10000, 100000]
misinfo_estimate_threshold = 50
potential_misinfo_tweets = []

for i, tweet in enumerate(tweets):
    analysis_result, score = analyze_tweet(tweet, likes[i], 100, sensitive_words)
    if score > misinfo_estimate_threshold:
        potential_misinfo_tweets.append((tweet, analysis_result, score, likes[i]))
    print("\n", f"Tweet {i+1}:", tweet)
    print("Analysis Result:", analysis_result)
    print("\n", "-" * 50)

# Decide further actions based on the score
for i, (tweet, analysis_result, score, likes) in enumerate(potential_misinfo_tweets):
    # TODO: Sort queue by score and likes
    note_assistance = assist_with_community_note(
        tweet, "gpt-4"
    )  # use gpt-4-1106-preview when available
    print("\n", f"Tweet {i+1}:", tweet)
    print("Assisted Community Note:", note_assistance)
