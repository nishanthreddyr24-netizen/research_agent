import requests

url = "http://localhost:8000/api/chat"
data = {
    "query": "what is the detection precision and recall?",
    "mode": "local"
}

response = requests.post(url, json=data)
print(response.json())