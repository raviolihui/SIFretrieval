import requests

response = requests.get("https://api.stacspec.org/v1.0.0/core/")

print(response.text)