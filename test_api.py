from infer import Infer
from fastapi import FastAPI

model_path = "/content/repo/fine_tuned_model/tokenizer"
tokenizer_path = "/content/repo/fine_tuned_model"
labels = ["B-LOC","B-MISC", "B-ORG", "B-PERS", "I-LOC","I-MISC", "I-ORG","I-PERS","O"]

inference = Infer(model_path, tokenizer_path)
app = FastAPI()

text = 'ماذا يفعل طلال عبد الهادي في دبي بعد ما رجع من برلين؟ كان يعمل هناك في شركة فولكسفاجن، صحيح؟'

@app.get('/')
def root():
  new_labels , word_ids = inference.infer(text)
  pre_labels = [labels[int(x)] for x in new_labels.tolist()[0]]
  pre_labels.reverse()
  return f'Text: {text} \n Labels:{pre_labels}'

@app.get("/NER/{name}")
async def NER(name):
    predict = Infer()
    prediction,word_ids=predict.infer(name)
# Map the predictions into str labels
    pre_labels = [labels[int(x)] for x in prediction.tolist()[0]]
    pre_labels.reverse()
# import pdb; pdb.Pdb(stdout=sys.stdout).set_trace()
    print(predict.tokens.tokens())
    print(pre_labels)
    out = "Text:"+name+"\n labels : "
    return pre_labels,predict.tokens.tokens()


# Main code to Run 
import nest_asyncio
from pyngrok import ngrok
import uvicorn

ngrok_tunnel = ngrok.connect(8000)
print('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app, port=8000)