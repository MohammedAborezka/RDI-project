import pandas as pd
from ast import literal_eval
from transformers import AutoTokenizer, AutoModelForTokenClassification

# initialize the tokenizer
tokenizer = AutoTokenizer.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-mix-ner")
def flaten(data):
  '''
  convert the list to single dimension

  Paramters
  ----------
  data: list of list
    nested list
  
  Return
  ---------
  list
    one dimension list of entities
  '''
  new_list = []
  for x in data : new_list.extend(x)
  return new_list
def prepare_data(path):
  '''
  Read the dataset from csv and convert it into two list (sentences and labels)

  Parameters
  ----------
  path: str
    path to the dataset.csv
  
  Return
  --------
  list,list
    list of sentence and list of labels
  '''

  my_data = pd.read_csv(path,index_col=0)
  # Convert the str into list
  my_data["label"]=my_data["label"].apply(lambda x :literal_eval(x))
  my_data["sentence"]=my_data["sentence"].apply(lambda x :literal_eval(x))

  return list(my_data["sentence"]),list(my_data["label"])

def toknize_text(data_to_be_tokenized):
  '''
  tokenize input data 

  Paramters
  ----------
  data_to_be_tokenized: list
    list of sentences to be tokenized
  
  Return
  ---------
  dict
    dict of {input_ids,attention_mast,token_type_ids} for the tokenized data
  '''
  return tokenizer(data_to_be_tokenized,truncation=True , is_split_into_words=True,max_length=512)
  
def align_labels_with_tokens(labels, word_ids):
  '''
  Match the labels with the new tokens generated from the tokenizer for one sentence

  Parameters
  ----------
  labels: list
    list of true labels
  word_ids: list
    list of ids for new tokens from tokens.word_ids() 
  
  Return
  -------
  list
    list of new labels aligned with tokens generated from the tokenizer
  '''
  new_labels = []
  current_word = None
  # iterate over word_ids from the tokenizer
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


def align_labels(encodings,labels):
  '''
  Uses align_labels_with_tokens funtion to apply the alignment for more than one sentence,
  and add it to the encodings dict.
  
  Parameters
  -----------
  encodings: dict
    dict of encodings for the dataset
  labels: list
    list of true labels

  Return
  -------
  dict
    dict of encodings {input_ids,attention_mask,token_type_ids,labels}
  '''
  all_labels = labels
  new_labels = []
  for i, labels in enumerate(all_labels):
      word_ids = encodings.word_ids(i)
      new_labels.append(align_labels_with_tokens(labels, word_ids))

  encodings["labels"] = new_labels
  return encodings

