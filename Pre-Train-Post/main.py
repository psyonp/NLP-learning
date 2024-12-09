#This project will break down the pipeline function from Huggingface, to the three steps:
#   1. Preprocessing
#   2. Running Model
#   3. Postprocessing

from transformers import AutoTokenizer
from transformers import DistilBertForSequenceClassification
import torch

#Preprocessing
model_name = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(model_name) #This function fetches the necessary data associated with the tokenizer for the BERT model
model = DistilBertForSequenceClassification.from_pretrained(model_name) #download the model from Huggingface

#Create a list of raw inputs, then use the tokenizer to convert them to tensors
inputs = ["I love playing pool.", "I hate pineapples on pizza."]
preprocessed_inputs = tokenizer(inputs, padding = True, truncation = True, return_tensors = "pt")

#Run the model
outputs = model(**preprocessed_inputs)

#Postprocess the data
predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
print(predictions)
print(model.config.id2label)