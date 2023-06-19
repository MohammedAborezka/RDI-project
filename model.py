from transformers import AutoTokenizer, AutoModelForTokenClassification

class model():
   def _init_(self):
    self.model = AutoModelForTokenClassification.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-mix-ner")
def forward():
  pass