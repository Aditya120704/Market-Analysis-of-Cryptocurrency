import joblib
import requests
import sys
import os
from pytrends.request import TrendReq
import pandas as pd
import json
from datetime import datetime, timedelta, timezone
import time
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Define the absolute path to the models directory
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models')
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache')

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
        if data and 'data' in data:
            # The API returns a list even for a single day, so we take the first element
            return data['data'][0] if data['data'] else {"error": "No data found"}
        else:
            return {"error": "No data found"}
    except requests.exceptions.RequestException as e:
        return {"error": f"API request failed: {e}"}

def get_google_trends(keywords: list, timeframe='today 5-y', cache_duration_minutes=10):
    """
    Fetches historical Google Trends data for a list of keywords, with caching.
    """
    # Sanitize keywords to create a valid filename
    sanitized_keywords = "_".join(keywords).replace(" ", "_")
    cache_filename = f"google_trends_{sanitized_keywords}.json"
    cache_filepath = os.path.join(CACHE_DIR, cache_filename)

    # Check if a valid cache file exists
    if os.path.exists(cache_filepath):
        # Check the modification time of the cache file
        mod_time = os.path.getmtime(cache_filepath)
        if (time.time() - mod_time) / 60 < cache_duration_minutes:
            try:
                # Load from cache
                return pd.read_json(cache_filepath, orient='index')
            except (ValueError, FileNotFoundError):
                # If cache is corrupted or not found, proceed to fetch new data
                pass

    # If no valid cache, fetch new data
    try:
        time.sleep(1) # To avoid rate limiting
        pytrends = TrendReq(hl='en-US', tz=360)
        pytrends.build_payload(keywords, cat=0, timeframe=timeframe, geo='', gprop='')
        trends_df = pytrends.interest_over_time()

        if not trends_df.empty:
            if 'isPartial' in trends_df.columns:
                trends_df = trends_df.drop(columns=['isPartial'])
                
                # Localize the index to UTC
                trends_df.index = trends_df.index.tz_localize('UTC')
            
            # Save to cache
            trends_df.to_json(cache_filepath, orient='index')
            
            return trends_df.fillna(False)
        else:
            return pd.DataFrame() # Return empty DataFrame on failure

    except Exception as e:
        print(f"Error fetching Google Trends data: {e}", file=sys.stderr)
        return pd.DataFrame() # Return empty DataFrame on failure

def get_historical_fear_and_greed_index(days: int = 0):
    """
    Fetches the historical Fear & Greed Index from the alternative.me API.
    :param days: Number of days to fetch. 0 for all available data.
    """
    url = f"https://api.alternative.me/fng/?limit={days}"
    try:
        response = requests.get(url)
        response.raise_for_status()
        data = response.json()
        if data and 'data' in data:
            fng_data = data['data']
            if not isinstance(fng_data, list):
                fng_data = [fng_data]
            
            fng_df = pd.DataFrame(fng_data)
            # Convert timestamp to datetime and set as index, localize to UTC
            fng_df['timestamp'] = pd.to_numeric(fng_df['timestamp']).apply(lambda x: datetime.fromtimestamp(x, tz=timezone.utc))
            fng_df.set_index('timestamp', inplace=True)
            fng_df['value'] = pd.to_numeric(fng_df['value']) # Ensure value is numeric
            return fng_df[['value']] # Return only the value column
        else:
            return pd.DataFrame() # Return empty DataFrame on no data
    except requests.exceptions.RequestException as e:
        print(f"Error fetching historical F&G data: {e}", file=sys.stderr)
        return pd.DataFrame() # Return empty DataFrame on error

NEWS_SENTIMENT_CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'news_sentiment_cache.json')

def get_historical_news_sentiment_scores(days: int = 30):
    """
    Fetches recent news, calculates sentiment, and aggregates daily scores.
    Includes caching to reduce API calls.
    """
    # Load model and vectorizer
    try:
        model = joblib.load(os.path.join(MODELS_DIR, 'sentiment_model.pkl'))
        vectorizer = joblib.load(os.path.join(MODELS_DIR, 'vectorizer.pkl'))
    except FileNotFoundError as e:
        print(f"Error loading sentiment model or vectorizer: {e}", file=sys.stderr)
        return pd.DataFrame() # Return empty DataFrame on error

    # Load cache
    cache = {}
    if os.path.exists(NEWS_SENTIMENT_CACHE_PATH):
        with open(NEWS_SENTIMENT_CACHE_PATH, 'r') as f:
            try:
                cache = json.load(f)
            except json.JSONDecodeError:
                cache = {} # Corrupted cache

    today = datetime.now().date()
    start_date = today - timedelta(days=days - 1)
    
    sentiment_scores = {}

    # Check cache for existing data
    for d_str, data in cache.items():
        d = datetime.strptime(d_str, '%Y-%m-%d').date()
        if start_date <= d <= today:
            sentiment_scores[d_str] = data['score']

    # Fetch missing data
    current_date = start_date
    api_key = '350b3451917542d6bcfeb8d4a780e197' # Your NewsAPI key

    while current_date <= today:
        date_str = current_date.strftime('%Y-%m-%d')
        if date_str not in sentiment_scores:
            try:
                # NewsAPI 'everything' endpoint has a 100 article limit per request for free tier
                # and a 1-month historical limit for free tier. For more, need paid plan.
                # We'll fetch for a single day to get a daily score.
                url = f'https://newsapi.org/v2/everything?q=crypto&from={date_str}&to={date_str}&pageSize=100&apiKey={api_key}'
                response = requests.get(url)
                response.raise_for_status()
                response_json = response.json()

                positive_count = 0
                negative_count = 0
                total_articles = 0

                if 'articles' in response_json and response_json['articles']:
                    for article in response_json['articles']:
                        headline = article['title']
                        if headline:
                            total_articles += 1
                            headline_vec = vectorizer.transform([headline])
                            sentiment_proba = model.predict_proba(headline_vec)[0]
                            if sentiment_proba[1] > sentiment_proba[0]:
                                positive_count += 1
                            elif sentiment_proba[0] > sentiment_proba[1]:
                                negative_count += 1

                daily_score = (positive_count - negative_count) / total_articles if total_articles > 0 else 0
                sentiment_scores[date_str] = daily_score
                # Add to cache
                cache[date_str] = {'score': daily_score, 'positive': positive_count, 'negative': negative_count, 'total': total_articles}

            except requests.exceptions.RequestException as e:
                print(f"Error fetching news for {date_str}: {e}", file=sys.stderr)
                sentiment_scores[date_str] = 0 # Default to neutral on error
            except Exception as e:
                print(f"Error processing news for {date_str}: {e}", file=sys.stderr)
                sentiment_scores[date_str] = 0 # Default to neutral on error
        
        current_date += timedelta(days=1)

    # Save updated cache
    with open(NEWS_SENTIMENT_CACHE_PATH, 'w') as f:
        json.dump(cache, f, indent=4)

    # Convert to DataFrame
    df = pd.DataFrame.from_dict(sentiment_scores, orient='index', columns=['news_sentiment'])
    df.index = pd.to_datetime(df.index).tz_localize('UTC')
    df.sort_index(inplace=True)
    return df