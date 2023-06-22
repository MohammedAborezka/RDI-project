from model import Model
import utils
import dataset
from datasets import ClassLabel
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score
import torch
from transformers import TrainingArguments, Trainer
from transformers import DataCollatorForTokenClassification
from transformers import AutoTokenizer, AutoModelForTokenClassification

sentence_train , labels_train = utils.prepare_data("./train_toknized_df.csv")
sentence_eval , labels_eval = utils.prepare_data("./eval_toknized_df.csv")

labels = ["B-LOC","B-MISC", "B-ORG", "B-PERS", "I-LOC","I-MISC", "I-ORG","I-PERS","O"]
c2l = ClassLabel(num_classes=9, names=labels)

embeded_labels_train = [c2l.str2int(label) for label in labels_train ]
embeded_labels_eval = [c2l.str2int(label) for label in labels_eval ]


text_encodings_train = utils.toknize_text(sentence_train)
text_encodings_with_labels_train = utils.align_labels(text_encodings_train,embeded_labels_train)

text_encodings_eval = utils.toknize_text(sentence_eval)
text_encodings_with_labels_eval = utils.align_labels(text_encodings_eval,embeded_labels_eval)

train_dataset = dataset.Dataset(text_encodings_with_labels_train, text_encodings_with_labels_train["labels"])
val_dataset = dataset.Dataset(text_encodings_with_labels_eval, text_encodings_with_labels_eval["labels"])


tokenizer = AutoTokenizer.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-mix-ner")
data_collator = DataCollatorForTokenClassification(tokenizer,padding=True)
#+++++++++++++++++++check length of labels and sentences are the same+++++++++++++++++
# for i,x in enumerate(text_encodings_with_labels_train["input_ids"]):
#   if len(x) != len(text_encodings_with_labels_train["labels"][i]):
#     print(True,len(x))

# def compute_metrics(p):
#     predictions, labels = p
#     print(predictions[0])
#     #select predicted index with maximum logit for each token
#     predictions = np.argmax(predictions, axis=2)
#     # Remove ignored index (special tokens)
#     true_predictions = [
#         [label_mapper[p] for (p, l) in zip(prediction, label) if l != -100]
#         for prediction, label in zip(predictions, labels)
#     ]
#     true_labels = [
#         [label_mapper[l] for (p, l) in zip(prediction, label) if l != -100]
#         for prediction, label in zip(predictions, labels)
#     ]
#     # print(true_predictions[0])
#     # print(true_labels[0])
#     results = metric.compute(predictions=true_predictions, references=true_labels)
#     return {
#         "precision": results["overall_precision"],
#         "recall": results["overall_recall"],
#         "f1": results["overall_f1"],
#         "accuracy": results["overall_accuracy"],
#     }
def flaten(data):
  new_list = []
  for x in data : new_list.extend(x)
  return new_list

def compute_metrics(p):
  predictions, true = p
  predictions = flaten(predictions)
  predictions = np.argmax(predictions, axis=1)
  true = flaten(true)
  predictions = [p for i,p in enumerate(predictions) if true[i] != -100]
  true = [t for t in true if t != -100]
  #import pdb, sys; pdb.Pdb(stdout=sys.stdout).set_trace()
  accuracy = accuracy_score(y_true=true, y_pred=predictions)
  recall = recall_score(y_true=true, y_pred=predictions,average="macro")
  precision = precision_score(y_true=true, y_pred=predictions,average="macro")
  f1 = f1_score(y_true=true, y_pred=predictions,average="macro")
  print('\n',classification_report(true, predictions,target_names=labels))

  return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}


args = TrainingArguments(
    output_dir="output",
    num_train_epochs=2,
    per_device_train_batch_size=8,
    evaluation_strategy = 'epoch'
    

)
# model = AutoModelForTokenClassification.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-mix-ner")
model = Model()
trainer = Trainer(
    model=model.model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    data_collator = data_collator

)

trainer.train()
#import pdb, sys; pdb.Pdb(stdout=sys.stdout).set_trace()
trainer.evaluate()
trainer.save_model("fine_tuned_model")
tokenizer.save_pretrained("fine_tuned_model/tokenizer")