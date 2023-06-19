import utils
from datasets import ClassLabel
import dataset

sentence_train , labels_train = utils.prepare_data("/content/repo/train_toknized_df.csv")
sentence_eval , labels_eval = utils.prepare_data("/content/repo/eval_toknized_df.csv")

labels = ["B-LOC","B-MISC", "B-ORG", "B-PERS", "I-LOC","I-MISC", "I-ORG","I-PERS","O"]
c2l = ClassLabel(num_classes=9, names=labels)

embeded_labels_train = [c2l.str2int(label) for label in labels_train ]
embeded_labels_eval = [c2l.str2int(label) for label in labels_eval ]

text_encodings_train = utils.toknize_text(sentence_train)
text_encodings_aligned_train = utils.align_labels(text_encodings_train,embeded_labels_train)

text_encodings_eval = utils.toknize_text(sentence_eval)
text_encodings_aligned_eval = utils.align_labels(text_encodings_eval,embeded_labels_eval)

train_dataset = dataset.Dataset(text_encodings_aligned_train, embeded_labels_train)
val_dataset = dataset.Dataset(text_encodings_aligned_eval, embeded_labels_eval)


print(text_encodings_aligned_train)