import joblib
import requests
import sys

def get_news_sentiment():
    try:
        # Load the model and vectorizer
        model = joblib.load('sentiment_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
    except FileNotFoundError as e:
        print(f"Error loading model or vectorizer: {e}", file=sys.stderr)
        return {"error": "Model or vectorizer not found", "details": str(e)}
    except Exception as e:
        print(f"Unexpected error loading model or vectorizer: {e}", file=sys.stderr)
        return {"error": "Unexpected error loading model components", "details": str(e)}

    try:
        # Fetch the news
        api_key = '350b3451917542d6bcfeb8d4a780e197'
        url = f'https://newsapi.org/v2/everything?q=crypto&apiKey={api_key}'
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        response_json = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news from NewsAPI: {e}", file=sys.stderr)
        return {"error": "Failed to fetch news", "details": str(e)}
    except ValueError as e: # Catches JSON decoding errors
        print(f"Error decoding NewsAPI response: {e}", file=sys.stderr)
        return {"error": "Invalid NewsAPI response", "details": str(e)}
    except Exception as e:
        print(f"Unexpected error during NewsAPI call: {e}", file=sys.stderr)
        return {"error": "Unexpected error during news fetch", "details": str(e)}

    if 'articles' in response_json and response_json['articles']:
        articles = response_json['articles']
        sentiment_data = []
        # Analyze the sentiment
        for article in articles:
            headline = article['title']
            published_at = article.get('publishedAt', 'N/A') # Get publishedAt, default to N/A
            article_url = article.get('url', '#') # Get URL, default to #
            if headline:
                try:
                    headline_vec = vectorizer.transform([headline])
                    sentiment = model.predict(headline_vec)[0]
                    sentiment_data.append({
                        'headline': headline,
                        'sentiment': "Positive" if sentiment == 4 else "Negative",
                        'publishedAt': published_at,
                        'url': article_url
                    })
                except Exception as e:
                    print(f"Error analyzing sentiment for headline '{headline}': {e}", file=sys.stderr)
                    # Optionally, skip this article or add an error flag to it
        return sentiment_data
    else:
        print(f"No articles found or unexpected NewsAPI response structure: {response_json}", file=sys.stderr)
        return {"error": "No articles found or unexpected response", "details": response_json}

def get_overall_market_sentiment():
    try:
        model = joblib.load('sentiment_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
    except FileNotFoundError as e:
        print(f"Error loading model or vectorizer: {e}", file=sys.stderr)
        return {"error": "Model or vectorizer not found", "details": str(e)}

    try:
        api_key = '350b3451917542d6bcfeb8d4a780e197'
        # Fetch more articles for overall sentiment
        url = f'https://newsapi.org/v2/everything?q=crypto&pageSize=100&apiKey={api_key}'
        response = requests.get(url)
        response.raise_for_status()
        response_json = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news for overall sentiment: {e}", file=sys.stderr)
        return {"error": "Failed to fetch news for overall sentiment", "details": str(e)}

    if 'articles' in response_json and response_json['articles']:
        articles = response_json['articles']
        positive_count = 0
        negative_count = 0
        neutral_count = 0

        for article in articles:
            headline = article['title']
            if headline:
                headline_vec = vectorizer.transform([headline])
                sentiment = model.predict(headline_vec)[0]
                if sentiment == 4: # Positive
                    positive_count += 1
                elif sentiment == 0: # Negative
                    negative_count += 1
                else: # Assuming anything else is neutral for simplicity with this model
                    neutral_count += 1
        
        total_articles = len(articles)
        if total_articles == 0:
            return {"overall_sentiment": "Neutral", "details": "No articles to analyze"}

        # Determine overall sentiment based on counts
        if positive_count > negative_count and positive_count > neutral_count:
            overall_sentiment = "Positive"
        elif negative_count > positive_count and negative_count > neutral_count:
            overall_sentiment = "Negative"
        else:
            overall_sentiment = "Neutral"

        return {
            "overall_sentiment": overall_sentiment,
            "positive_articles": positive_count,
            "negative_articles": negative_count,
            "neutral_articles": neutral_count,
            "total_articles": total_articles
        }
    else:
        print(f"No articles found for overall sentiment or unexpected NewsAPI response structure: {response_json}", file=sys.stderr)
        return {"error": "No articles found for overall sentiment", "details": response_json}

def get_currency_sentiment(currency_name: str):
    try:
        model = joblib.load('sentiment_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
    except FileNotFoundError as e:
        print(f"Error loading model or vectorizer: {e}", file=sys.stderr)
        return {"error": "Model or vectorizer not found", "details": str(e)}

    try:
        api_key = '350b3451917542d6bcfeb8d4a780e197'
        # Fetch news for the specific currency
        url = f'https://newsapi.org/v2/everything?q={currency_name}&apiKey={api_key}'
        response = requests.get(url)
        response.raise_for_status()
        response_json = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news for {currency_name}: {e}", file=sys.stderr)
        return {"error": f"Failed to fetch news for {currency_name}", "details": str(e)}

    if 'articles' in response_json and response_json['articles']:
        articles = response_json['articles']
        sentiment_data = []
        for article in articles:
            headline = article['title']
            published_at = article.get('publishedAt', 'N/A')
            article_url = article.get('url', '#')
            if headline:
                try:
                    headline_vec = vectorizer.transform([headline])
                    sentiment = model.predict(headline_vec)[0]
                    sentiment_data.append({
                        'headline': headline,
                        'sentiment': "Positive" if sentiment == 4 else "Negative",
                        'publishedAt': published_at,
                        'url': article_url
                    })
                except Exception as e:
                    print(f"Error analyzing sentiment for headline '{headline}': {e}", file=sys.stderr)
        return sentiment_data
    else:
        print(f"No articles found for {currency_name} or unexpected NewsAPI response structure: {response_json}", file=sys.stderr)
        return {"error": f"No articles found for {currency_name}", "details": response_json}

def get_overall_currency_sentiment(currency_name: str):
    try:
        model = joblib.load('sentiment_model.pkl')
        vectorizer = joblib.load('vectorizer.pkl')
    except FileNotFoundError as e:
        print(f"Error loading model or vectorizer: {e}", file=sys.stderr)
        return {"error": "Model or vectorizer not found", "details": str(e)}

    try:
        api_key = '350b3451917542d6bcfeb8d4a780e197'
        url = f'https://newsapi.org/v2/everything?q={currency_name}&pageSize=100&apiKey={api_key}'
        response = requests.get(url)
        response.raise_for_status()
        response_json = response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching news for overall {currency_name} sentiment: {e}", file=sys.stderr)
        return {"error": f"Failed to fetch news for overall {currency_name} sentiment", "details": str(e)}

    if 'articles' in response_json and response_json['articles']:
        articles = response_json['articles']
        positive_count = 0
        negative_count = 0
        neutral_count = 0

        for article in articles:
            headline = article['title']
            if headline:
                headline_vec = vectorizer.transform([headline])
                sentiment = model.predict(headline_vec)[0]
                if sentiment == 4: # Positive
                    positive_count += 1
                elif sentiment == 0: # Negative
                    negative_count += 1
                else: # Assuming anything else is neutral for simplicity with this model
                    neutral_count += 1
        
        total_articles = len(articles)
        if total_articles == 0:
            return {"overall_sentiment": "Neutral", "details": "No articles to analyze"}

        if positive_count > negative_count and positive_count > neutral_count:
            overall_sentiment = "Positive"
        elif negative_count > positive_count and negative_count > neutral_count:
            overall_sentiment = "Negative"
        else:
            overall_sentiment = "Neutral"

        return {
            "overall_sentiment": overall_sentiment,
            "positive_articles": positive_count,
            "negative_articles": negative_count,
            "neutral_articles": neutral_count,
            "total_articles": total_articles
        }
    else:
        print(f"No articles found for overall {currency_name} sentiment or unexpected NewsAPI response structure: {response_json}", file=sys.stderr)
        return {"error": f"No articles found for overall {currency_name} sentiment", "details": response_json}