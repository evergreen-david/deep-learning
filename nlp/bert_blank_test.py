import torch
from transformers import (
    RobertaForMaskedLM,
    RobertaModel,
    RobertaTokenizer
)
import numpy as np

model = RobertaForMaskedLM.from_pretrained('roberta-base')
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

txt = 'Hi, my name is <mask>'
tokens = tokenizer(txt, return_tensors='pt')

mask_index = np.where(tokens['input_ids'][0].numpy() == 50264)[0][0]

probs = model(tokens['input_ids'], attention_mask=tokens['attention_mask'])

for token in probs[0][:, mask_index, :].sort(descending=True)[1][0][:10]:
    print(tokenizer.decode(token.item()))
