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

args = sys.argv
labels = ["B-LOC","B-MISC", "B-ORG", "B-PERS", "I-LOC","I-MISC", "I-ORG","I-PERS","O"]

model_path = "/content/repo/fine_tuned_model/tokenizer"
tokenizer_path = "/content/repo/fine_tuned_model"

class Infer():
  def __init__(self,model_path = model_path, tokenizer_path = tokenizer_path):
    # Initialize the tokenizer and the model
    self.tokenizer = AutoTokenizer.from_pretrained(model_path)
    self.model = AutoModelForTokenClassification.from_pretrained(tokenizer_path)

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
  args = sys.argv
  
  predict = Infer()
  prediction,word_ids=predict.infer(args[1])
  # Map the predictions into str labels
  pre_labels = [labels[int(x)] for x in prediction.tolist()[0]]
  pre_labels.reverse()
  # import pdb; pdb.Pdb(stdout=sys.stdout).set_trace()
  print(predict.tokens.tokens())
  print(pre_labels)

if __name__ == "__main__":
  main()





