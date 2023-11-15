from src.tweet_analysis import analyze_tweet, assist_with_community_note
from src.data_preparation import load_keywords

# Load keywords and contentious issues
sensitive_words = load_keywords("data/keywords.csv")
contentious_issues = load_keywords("data/contentious_issues.csv")

# Example usage
tweets = ["I hate the new update!", "I love the new update!", "QAnon is a true."]
likes = 150
misinfo_estimate_threshold = 50

for tweet in tweets:
    analysis_result, score = analyze_tweet(tweet, likes, 100, sensitive_words)
    print("Analysis Result:", analysis_result)

# Decide further actions based on the score
if score > misinfo_estimate_threshold:
    # If the score is above a certain threshold, proceed with further actions
    use_gpt4_for_assistance = likes >= 10000
    note_assistance = assist_with_community_note(tweet, use_gpt4_for_assistance)
    print("Assisted Community Note:", note_assistance)
