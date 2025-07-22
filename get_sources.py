import requests
import json

api_key = '350b3451917542d6bcfeb8d4a780e197'
url = f'https://newsapi.org/v2/sources?apiKey={api_key}'
response = requests.get(url)
with open('sources.json', 'w') as f:
    json.dump(response.json(), f)