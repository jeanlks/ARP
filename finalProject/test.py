import requests

response = requests.get('https://restcountries.eu/rest/v2/name/brasil')
data = response.json()
print(data)