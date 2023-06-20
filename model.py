from torch.nn.modules import module
from transformers import AutoTokenizer, AutoModelForTokenClassification

class Model(nn.modlue):
   def _init_(self):
    self.model = AutoModelForTokenClassification.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-mix-ner")
    def get_model(self):
      return self.model
def forward():
  pass