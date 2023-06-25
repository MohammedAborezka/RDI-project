import utils
from datasets import ClassLabel
import dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.metrics import classification_report
from infer import Infer
from tqdm import tqdm
import sys

test_args = sys.argv 

# read the test dataset
# Path of the dataset >> "/content/repo/toknized_test_df.csv"
sentence_test , labels_test = utils.prepare_data(test_args[1])

# Convert text labels into numbers
labels = ["B-LOC","B-MISC", "B-ORG", "B-PERS", "I-LOC","I-MISC", "I-ORG","I-PERS","O"]
c2l = ClassLabel(num_classes=9, names=labels)
embeded_labels_test = [c2l.str2int(label) for label in labels_test ]

# text_encodings_test = utils.toknize_text(sentence_test)
# text_encodings_with_labels_test = utils.align_labels(text_encodings_test,embeded_labels_test)

# test_dataset = dataset.Dataset(text_encodings_with_labels_test, text_encodings_with_labels_test["labels"])

# tokenizer = AutoTokenizer.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-mix-ner")
# model = AutoModelForTokenClassification.from_pretrained("/content/repo/fine_tuned_model")
# data_collator = DataCollatorForTokenClassification(tokenizer,padding=True)

# def flaten(data):
#   '''
#   convert the list to single dimension

#   Paramters
#   ----------
#   data: list of list
#     nested list
  
#   Return
#   ---------
#   list
#     one dimension list of entities
#   '''
#   new_list = []
#   for x in data : new_list.extend(x)
#   return new_list

def compute_metrics(p,t):
  '''
  take actual and predicated labels compute accuracy,recall,precision, f1_score
  and print classification report 

  Paramters
  ----------
  p: list
    list of predicated labels
  t: list 
    list of true labels
  
  Return
  ---------
  dict
    dict of {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1} for the test data
  '''
  predictions = p
  true = t
  predictions = utils.flaten(predictions)
  #predictions = np.argmax(predictions, axis=1)
  true = utils.flaten(true)
  #predictions = [p for i,p in enumerate(predictions) if true[i] != -100]
  #true = [t for t in true if t != -100]
  # import pdb, sys; pdb.Pdb(stdout=sys.stdout).set_trace()
  accuracy = accuracy_score(y_true=true, y_pred=predictions)
  recall = recall_score(y_true=true, y_pred=predictions,average="macro")
  precision = precision_score(y_true=true, y_pred=predictions,average="macro")
  f1 = f1_score(y_true=true, y_pred=predictions,average="macro")
  # print(classification_report(true, predictions))
  print(classification_report(true, predictions,target_names=labels,labels=list(range(9))))

  return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

# outputs = model(**text_encodings_with_labels_test)
# logits = outputs.logits
# predictions = logits.argmax(-1)
# trainer = Trainer(model=model,data_collator=data_collator)
# eval1 = trainer.predict(test_dataset)


inference = Infer(test_args[2],test_args[3])
predicted_labels = []
# iterate over all the sentences from the dataset and run infer for each one  
for sen in tqdm(sentence_test):
  new_labels , word_ids = inference.infer(sen)
  joined_outputs = []
  last_word_id = None
  # Remove duplicated labels for duplicated tokens
  for output, wid in zip(new_labels.tolist()[0], word_ids):
    if wid!=None and wid!=last_word_id: 
      joined_outputs.append(output)
    last_word_id = wid 
  predicted_labels.append(joined_outputs)

# Compute the metrics for the predicted labels with the true labels
print(compute_metrics(predicted_labels,embeded_labels_test))




