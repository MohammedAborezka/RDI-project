import pandas as pd
from ast import literal_eval
from transformers import AutoTokenizer, AutoModelForTokenClassification

def prepare_data(path):
  my_data = pd.read_csv(path,index_col=0)
  my_data["label"]=my_data["label"].apply(lambda x :literal_eval(x))
  my_data["sentence"]=my_data["sentence"].apply(lambda x :literal_eval(x))
  return list(my_data["sentence"]),list(my_data["label"])

def toknize_text(data_to_be_tokenized):
  tokenizer = AutoTokenizer.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-mix-ner")
  return tokenizer(data_to_be_tokenized,truncation=True , is_split_into_words=True,max_length=512)
  
def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word = None
    for word_id in word_ids:
        if word_id != current_word:
            # Start of a new word!
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            # Special token
            new_labels.append(-100)
        else:
            # Same word as previous token
            label = labels[word_id]
            # If the label is B-XXX we change it to I-XXX
            if label <= 3:
              label+=4
            new_labels.append(label)

    return new_labels 

# def align_labels(encodings,all_labels):

#     my_list = []
#     for index,labels in enumerate(all_labels):
#       new_labels = []
#       for i, label in enumerate(labels):
#           word_ids = encodings[index].word_ids(i)
#           new_labels.append(align_labels_with_tokens(labels, word_ids))
#       my_list.append(new_labels)
    
#     encodings["labels"] = my_list
#     return encodings


def align_labels(encodings,labels):
    all_labels = labels
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = encodings.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    encodings["labels"] = new_labels
    return encodings

