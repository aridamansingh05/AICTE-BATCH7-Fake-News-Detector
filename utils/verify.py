import requests

API_KEY = "0b76edae40964b28bd766ffaf022b815"

def verify_with_newsapi(query):
    url = f"https://newsapi.org/v2/everything?q={query}&apiKey={API_KEY}"
    
    response = requests.get(url)
    data = response.json()

    if data.get("totalResults", 0) > 0:
        return True
    return False
