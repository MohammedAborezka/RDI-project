# Arabic NER Using ANERcorp - CAMeL Lab Dataset

A named entity recognition model for Arabic text to recognize persons, locations, and organizations

## Table of Contents

- [Introduction](#introduction)
- [Environment](#environment)
- [Dataset](#dataset)
- [Usage](#usage)
- [Model](#model)
- [Results](#results)


## Introduction
Named Entity Recognition (NER) is a task of Natural Language Processing (NLP) that involves identifying and classifying named entities in a text into predefined categories such as person names, organizations, locations, and others. We used hugging face transformer: CAMeL-Lab/bert-base-arabic-camelbert-mix-ner as our base model, and we fine-tuned it with ANERcorp - CAMeL Lab Dataset. The Final model accuracy is 0.97 on the test dataset.

### Steps of the project:
    1. Explore the ANERCorp Dataset from CamelLabs
    2. Find and add more datasets
    3. Train one or more models
    4. Compare different models using F1 Score

### Current progress:
    - [x] 1. Explore the ANERCorp Dataset from CamelLabs [ANERCorp.ipynb].
    - [x] 2. Train on  [ANERCorp.ipynb].
    - [x] 3. Train one or more models.
    - [ ] 4. Compare different models using F1 Score. (On progress)
    - [ ] 5- Add more Datasets


## Environment
The code is compatible with `Python 3.10.12` and `torch 2.0.1` and `transformers 4.30.2`

## Dataset
The dataset used was the [ANERcorp - CAMeL Lab](https://camel.abudhabi.nyu.edu/anercorp). ANERCorp by Benajiba et al is an Arabic named entities manually annotated corpus that is freely available for research purposes. It contains 150,286 tokens and 32,114 named entities from 316 articles that were selected from different types and different newspapers in order to obtain a corpus as generalized as possible. The corpus was annotated following the annotation scheme defined in the MUC-6 with IOB tagging. Words tagged with nine classes were used for the annotation:
-	B-PER and I-PER for persons
-	B-LOC and I-LOC for locations
-	B-ORG and I-ORG for organizations
-	B-MISC and I-MISC for generic miscellaneous
-	O which means out of the problem context
  
## Usage
First, you have to get the dataset from the link provided above, then clone the repo `!git clone https://github.com/MohammedAborezka/RDI-project.git`
Second, pass the path of the dataset to sentence_analysis_camel.ipynb file to get the tokenized dataset.  
Then:  
1- Train  
   `!python train.py` 
   
2- Test  
   `!python test.py`  
   
3- Inference  
   `python infer.py <text_to_use>`  

## Model
CAMeLBERT-Mix NER Model is a Named Entity Recognition (NER) model that was built by fine-tuning the CAMeLBERT Mix model. For the fine-tuning, we used the ANERcorp dataset. We Fine-tuned the model using the same dataset and this is the final model [Final Model](https://drive.google.com/drive/u/0/folders/11VpoPORhv6IMNMc7wA0woSXeQys7Vc5m)

## Results
1- Train Results:
Accuracy: 99%

Classification report:

                  precision    recall  f1-score   support

           B-LOC       0.95      0.98      0.97       692
          B-MISC       0.91      0.91      0.91       171
           B-ORG       0.94      0.87      0.90       312
          B-PERS       0.97      0.98      0.97       523
           I-LOC       0.91      0.95      0.93       335
          I-MISC       0.89      0.85      0.87       175
           I-ORG       0.95      0.89      0.92       463
          I-PERS       0.98      0.99      0.98       942
               O       1.00      1.00      1.00     25719
           
        accuracy                           0.99     29332
       macro avg       0.94      0.93      0.94     29332
    weighted avg       0.99      0.99      0.99     29332


2- Test Results:
Accuracy: 97%

Classification report:

                     precision    recall  f1-score   support

             B-LOC       0.89      0.95      0.92       668
            B-MISC       0.80      0.63      0.70       235
             B-ORG       0.83      0.71      0.77       450
            B-PERS       0.87      0.86      0.86       858
             I-LOC       0.83      0.83      0.83        83
            I-MISC       0.82      0.37      0.51       165
             I-ORG       0.76      0.67      0.71       275
            I-PERS       0.90      0.90      0.90       641
                 O       0.98      0.99      0.99     20672
           
          accuracy                           0.97     24047
         macro avg       0.85      0.77      0.80     24047
      weighted avg       0.97      0.97      0.97     24047



