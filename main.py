from src.tweet_analysis import analyze_tweet, assist_with_community_note
from src.data_preparation import load_keywords

# Load keywords and contentious issues
sensitive_words = load_keywords("data/keywords.csv")
contentious_issues = load_keywords("data/contentious_issues.csv")

# Example usage
tweet = "Example tweet content."
likes = 150

analysis_result = analyze_tweet(tweet, likes, 100, sensitive_words)
print("Analysis Result:", analysis_result)

if "needs review" in analysis_result:
    use_gpt4_for_assistance = likes >= 10000
    note_assistance = assist_with_community_note(tweet, use_gpt4_for_assistance)
    print("Assisted Community Note:", note_assistance)
