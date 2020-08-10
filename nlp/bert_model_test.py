import torch
from transformers import BertModel, BertTokenizer

model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

torch.tensor(tokenizer.encode("deep learning generates a lot of interest"))

tokens = torch.tensor(tokenizer.encode("deep learning's fun") ).view(1, -1)

representations = model(tokens)[0]
print(representations.shape)

