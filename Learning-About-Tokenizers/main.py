#Learning about tokenizers in Huggingface

from transformers import AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("bert-base-cased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

#Letting the function do the hard work for us
tensors = tokenizer("All these words will be quantified, it doesn't matter if I write in reverse or not. Azzip tae lliw I.")
print("Tensors directly from function: \n")
print(tensors)

#Now we're going to split up the tokenization process
sequence = "I will eat pizza, and I will not tolerate pineapples on it."

#This will tokenize subwords
tokens = tokenizer.tokenize(sequence)
print("\nList of tokens: ")
print(tokens)

#This will convert the tokens to ids, which can then be fed to models
ids = tokenizer.convert_tokens_to_ids(tokens)
print("\nList of ids: ")
print(ids)

#This will decode the ids into words
decoded_sequence = tokenizer.decode(ids)
print("\nDecoded sequence: ")
print(decoded_sequence)