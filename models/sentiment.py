import mwclient
import time
from transformers import pipeline
import pandas as pd
from statistics import mean

site = mwclient.Site('en.wikipedia.org')
page = site.pages['Bitcoin']
revs = list(page.revisions())

revs = sorted(revs, key=lambda rev: rev["timestamp"])

sentiment_pipeline = pipeline("sentiment-analysis")

def find_sentiment(text):
    if not text or not isinstance(text, str):
        return 0
    try:
        sent = sentiment_pipeline([text[:250]])[0]
        score = sent["score"]
        if sent["label"] == "NEGATIVE":
            score *= -1
    except Exception as e:
        return 0
    return score

edits = {}

for rev in revs:
    date = time.strftime("%Y-%m-%d", rev["timestamp"])
    if date not in edits:
        edits[date] = dict(sentiments=list(), edit_count=0)
    
    edits[date]["edit_count"] += 1
    
    comment = rev.get("comment", "")
    edits[date]["sentiments"].append(find_sentiment(comment))

for key in edits:
    if len(edits[key]["sentiments"]):
        edits[key]["sentiment"] = mean(edits[key]["sentiments"])
        edits[key]["neg_sentiment"] = len([s for s in edits[key]["sentiments"] if s < 0]) / len(edits[key]["sentiments"])
    else:
        edits[key]["sentiment"] = 0
        edits[key]["neg_sentiment"] = 0
    del edits[key]["sentiments"]

edits_df = pd.DataFrame.from_dict(edits, orient="index")

from datetime import datetime

edits_df.index = pd.to_datetime(edits_df.index)

all_dates = pd.date_range(start=edits_df.index.min(), end=datetime.today())

edits_df = edits_df.reindex(all_dates, fill_value=0)

rolling_edits = edits_df.rolling(30, min_periods=30).mean()

rolling_edits = rolling_edits.dropna()

if __name__ == "__main__":
    # Test edge cases for find_sentiment
    test_texts = ["", None, "Good update", "Bad update", 12345]
    for text in test_texts:
        print(f"Sentiment for '{text}': {find_sentiment(text)}")

    # Run the main script logic
    site = mwclient.Site('en.wikipedia.org')
    page = site.pages['Bitcoin']
    revs = list(page.revisions())

    revs = sorted(revs, key=lambda rev: rev["timestamp"])

    edits = {}

    for rev in revs:
        date = time.strftime("%Y-%m-%d", rev["timestamp"])
        if date not in edits:
            edits[date] = dict(sentiments=list(), edit_count=0)

        edits[date]["edit_count"] += 1

        comment = rev.get("comment", "")
        edits[date]["sentiments"].append(find_sentiment(comment))

    for key in edits:
        if len(edits[key]["sentiments"]):
            edits[key]["sentiment"] = mean(edits[key]["sentiments"])
            edits[key]["neg_sentiment"] = len([s for s in edits[key]["sentiments"] if s < 0]) / len(edits[key]["sentiments"])
        else:
            edits[key]["sentiment"] = 0
            edits[key]["neg_sentiment"] = 0
        del edits[key]["sentiments"]

    edits_df = pd.DataFrame.from_dict(edits, orient="index")

    from datetime import datetime

    edits_df.index = pd.to_datetime(edits_df.index)

    all_dates = pd.date_range(start=edits_df.index.min(), end=datetime.today())

    edits_df = edits_df.reindex(all_dates, fill_value=0)

    rolling_edits = edits_df.rolling(30, min_periods=30).mean()

    rolling_edits = rolling_edits.dropna()

    rolling_edits.to_csv(r"E:\Python programs\bitcoin_price_predictor\wikipedia_edits.csv")

    if not rolling_edits.empty:
        print("Sentiment data saved to wikipedia_edits.csv")
        print("Sample of the data:")
        print(rolling_edits.tail())
        
        last_sentiment = rolling_edits['sentiment'].iloc[-1]
        if last_sentiment > 0.05:
            print("Overall sentiment is Positive")
        elif last_sentiment < -0.05:
            print("Overall sentiment is Negative")
        else:
            print("Overall sentiment is Neutral")
    else:
        print("Could not determine sentiment. Not enough historical data found to generate a sentiment score.")

