from torch import nn
from transformers import AutoModelForTokenClassification

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = AutoModelForTokenClassification.from_pretrained("CAMeL-Lab/bert-base-arabic-camelbert-mix-ner")

  def forward(self,input_ids,attention_mask):
    # # import pdb, sys; pdb.Pdb(stdout=sys.stdout).set_trace()
    output = self.model(input_ids, attention_mask=attention_mask)
    return output