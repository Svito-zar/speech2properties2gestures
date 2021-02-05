from transformers import DistilBertTokenizer, DistilBertModel
import torch

tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-german-cased')
model = DistilBertModel.from_pretrained('distilbert-base-german-cased')

input_ids = torch.tensor(tokenizer.encode("Hallo")).unsqueeze(0)
outputs = model(input_ids)
last_hidden_states = outputs[0]

print(last_hidden_states[0,1:-1,:10])
print(last_hidden_states[0,1:-1,:10].shape)

input_ids = torch.tensor(tokenizer.encode("Hallo, wie geht es dir")).unsqueeze(0)
outputs = model(input_ids)
last_hidden_states = outputs[0]

print(last_hidden_states[0,1:-1,:10])
print(last_hidden_states[0,1:-1,:10].shape)

input_ids = torch.tensor(tokenizer.encode("Hallo, wie geht es")).unsqueeze(0)
outputs = model(input_ids)
last_hidden_states = outputs[0]

print(last_hidden_states[0,1:-1,:10])

print(last_hidden_states[0,1:-1,:10].shape)