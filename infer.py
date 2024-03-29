import utils
from datasets import ClassLabel
import dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import classification_report
import sys
import torch
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi_standalone_docs import StandaloneDocs


args = sys.argv
labels = ["B-LOC","B-MISC", "B-ORG", "B-PERS", "I-LOC","I-MISC", "I-ORG","I-PERS","O"]

class Infer():
  def __init__(self):
    # Initialize the tokenizer and the model
    self.tokenizer = AutoTokenizer.from_pretrained("fine_tuned_model/tokenizer")
    self.model = AutoModelForTokenClassification.from_pretrained("fine_tuned_model")
  
  def infer(self,sentence):
    # Select the tokenizer for one str sentence or for a tokenized sentence (list)
    if type(sentence) == str:
      self.tokens = self.tokenizer(sentence,truncation=True ,max_length=512,return_tensors='pt')
    else:
      self.tokens = self.tokenizer(sentence,truncation=True ,is_split_into_words=True,max_length=512,return_tensors='pt')

    #print(tokens.tokens())
    # Send the tokenized input into the model
    res = self.model(**self.tokens)
    predicted_label = res.logits.argmax(-1)
    return predicted_label , self.tokens.word_ids()
   # pre_labels = [labels[int(x)] for x in predicted_label.tolist()[0]]
    #pre_labels.reverse()

def main():
     pass
  
  

if __name__ == "__main__":
  main()

class Item(BaseModel):
    name: str
    

app = FastAPI()

@app.post("/ner")
async def create_item(item: Item):
    predict = Infer()
    prediction,word_ids=predict.infer(item.name)
# Map the predictions into str labels
    pre_labels = [labels[int(x)] for x in prediction.tolist()[0]]
    pre_labels
# import pdb; pdb.Pdb(stdout=sys.stdout).set_trace()
    #print(predict.tokens.tokens())
    #print(pre_labels)
    return {"lables":pre_labels[1:-1],"tokens":predict.tokens.tokens()[1:-1]}






