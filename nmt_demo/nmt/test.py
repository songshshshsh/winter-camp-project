import requests

print(requests.post('http://localhost:5000/api/predict', json = {'text': 'I love coding', 'label0': 'INTJ', 'label1': 'ISFP'}).text)
