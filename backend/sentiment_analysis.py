import joblib
import requests
import sys
import os

# Define the absolute path to the models directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')

def get_news_sentiment():
    try:
        # Load the model and vectorizer
        model = joblib.load(os.path.join(MODELS_DIR, 'sentiment_model.pkl'))
        vectorizer = joblib.load(os.path.join(MODELS_DIR, 'vectorizer.pkl'))
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
                    sentiment_proba = model.predict_proba(headline_vec)[0]
                    if sentiment_proba[1] > sentiment_proba[0]:
                        sentiment_label = "Positive"
                    elif sentiment_proba[0] > sentiment_proba[1]:
                        sentiment_label = "Negative"
                    else:
                        sentiment_label = "Neutral"
                    sentiment_data.append({
                        'headline': headline,
                        'sentiment': sentiment_label,
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
        model = joblib.load(os.path.join(MODELS_DIR, 'sentiment_model.pkl'))
        vectorizer = joblib.load(os.path.join(MODELS_DIR, 'vectorizer.pkl'))
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
                sentiment_proba = model.predict_proba(headline_vec)[0]
                if sentiment_proba[1] > sentiment_proba[0]:
                    positive_count += 1
                elif sentiment_proba[0] > sentiment_proba[1]:
                    negative_count += 1
                else:
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

def get_sentiment_score():
    sentiment_data = get_overall_market_sentiment()
    if "error" in sentiment_data:
        return 0 # Return a neutral score in case of an error
    
    positive_count = sentiment_data.get("positive_articles", 0)
    negative_count = sentiment_data.get("negative_articles", 0)
    total_articles = sentiment_data.get("total_articles", 1) # Avoid division by zero
    
    if total_articles == 0:
        return 0 # Return neutral if no articles
        
    return (positive_count - negative_count) / total_articles

def get_currency_sentiment(currency_name: str):
    try:
        model = joblib.load(os.path.join(MODELS_DIR, 'sentiment_model.pkl'))
        vectorizer = joblib.load(os.path.join(MODELS_DIR, 'vectorizer.pkl'))
    except FileNotFoundError as e:
        print(f"Error loading model or vectorizer: {e}", file=sys.stderr)
        return {"error": "Model or vectorizer not found", "details": str(e)}

    api_key = '350b3451917542d6bcfeb8d4a780e197'

    # Try a sequence of queries to improve hit rate
    queries = [currency_name, f"{currency_name} crypto", f"{currency_name} coin", f"{currency_name} token"]
    response_json = {"articles": []}

    for q in queries:
        try:
            url = f'https://newsapi.org/v2/everything?q={q}&apiKey={api_key}'
            response = requests.get(url)
            response.raise_for_status()
            tmp = response.json()
            if 'articles' in tmp and tmp['articles']:
                response_json = tmp
                break
        except requests.exceptions.RequestException as e:
            print(f"Error fetching news for {q}: {e}", file=sys.stderr)
            continue

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
                    sentiment_proba = model.predict_proba(headline_vec)[0]
                    if sentiment_proba[1] > sentiment_proba[0]:
                        sentiment_label = "Positive"
                    elif sentiment_proba[0] > sentiment_proba[1]:
                        sentiment_label = "Negative"
                    else:
                        sentiment_label = "Neutral"
                    sentiment_data.append({
                        'headline': headline,
                        'sentiment': sentiment_label,
                        'publishedAt': published_at,
                        'url': article_url
                    })
                except Exception as e:
                    print(f"Error analyzing sentiment for headline '{headline}': {e}", file=sys.stderr)
        return sentiment_data
    else:
        print(f"No articles found for {currency_name} or unexpected NewsAPI response structure: {response_json}", file=sys.stderr)
        # Return empty list instead of error so UI can display 'no headlines' message
        return []

def get_overall_currency_sentiment(currency_name: str):
    try:
        model = joblib.load(os.path.join(MODELS_DIR, 'sentiment_model.pkl'))
        vectorizer = joblib.load(os.path.join(MODELS_DIR, 'vectorizer.pkl'))
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
                sentiment_proba = model.predict_proba(headline_vec)[0]
                if sentiment_proba[1] > sentiment_proba[0]:
                    positive_count += 1
                elif sentiment_proba[0] > sentiment_proba[1]:
                    negative_count += 1
                else:
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
        print(f"No articles found for overall {currency_name} sentiment or unexpected NewsAPI response structure: {response_json}", file=sys.stderr)
        return {"error": f"No articles found for overall {currency_name} sentiment", "details": response_json}

def get_fear_and_greed_index():
    """
    Fetches the Fear & Greed Index from the alternative.me API.
    """
    url = "https://api.alternative.me/fng/?limit=1"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
        data = response.json()
        if data and 'data' in data and len(data['data']) > 0:
            return data['data'][0]
        else:
            return {"error": "No data found"}
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {e}"}