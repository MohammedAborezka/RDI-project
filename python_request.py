import requests

url = 'http://127.0.0.1:8000'
myobj = {'name':'إمارة أبوظبي هي إحدى إمارات دولة الإمارات العربية المتحدة السبع'}

x = requests.post(url, json = myobj)

for word, y in enumerate(x.json()["tokens"]):
     print(y+"---->"+x.json()["lables"][word])
