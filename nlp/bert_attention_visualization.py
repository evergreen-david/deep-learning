import torch
from transformers import BertModel, BertTokenizer

import matplotlib.pyplot as plt

model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

txt = "I heard it today that she doesn't meet the requirements, and by the way the lawyer who wrote the piece is highly qualified, very talented. I assumed the Democrats would've checked that out before she gets chosen for vice president."

tokens = tokenizer(txt, return_tensors='pt')
attentions = model(tokens['input_ids'],
    attention_mask=tokens['attention_mask'],
    output_attentions=True)[-1]

# len(attentions) = 12 : multi-head attention의 head가 12개
# attentions[0].shape = (1, 12, 50, 50) : 문장 1개, 12개의 head, 50개의 단어
plt.figure(figsize=(12, 12))
plt.pcolor(attentions[9][0, 3, :, :].detach().numpy()) # 9번째 layer에 있는 multi-head block 중 head의 3번째 head
plt.show()
