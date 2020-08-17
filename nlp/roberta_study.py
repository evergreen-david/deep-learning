!pip install nlp
!pip install transformers

import nlp
from transformers import RobertaModel, RobertaTokenizer

import torch
import torch as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

mnli = nlp.load_dataset(path='glue', name='mnli')
mnli

roberta_tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
roberta_model = RobertaModel.from_pretrained('roberta-base')

mnli_train = mnli['train']
len(mnli['train'])

mnli_train[0]
mnli_train[0]['label']

class MNLI(Dataset):
    def __init__(self):
        self.mnli = nlp.load_dataset(path='glue', name='mnli')['train']
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    def __len__(self):
        return len(self.mnli)

    def __getitem__(self, index):
        data = self.mnli[index]
        tokens = self.tokenizer(data['hypothesis'], data['premise'], 
            max_length=100,
            padding='max_length',
            truncation=True,
            return_tensors='pt')
                       
        return tokens['input_ids'].squeeze(), tokens['attention_mask'].squeeze(), data['label']

batch = next(iter(DataLoader(MNLI(), batch_size=32)))
batch[0].shape

token_result = roberta_tokenizer("Hi, I like coffee", return_tensors='pt')
token_result

representations1 = roberta_model(input_ids = token_result['input_ids'], attention_mask = token_result['attention_mask'])
representations1

representations2 = roberta_model(input_ids = token_result['input_ids'], attention_mask = token_result['attention_mask'], output_hidden_states=True)
representations2

representations2[0].shape

len(representations1)
len(representations2)

final_rep, _, hidden_rep = roberta_model(input_ids = token_result['input_ids'], attention_mask = token_result['attention_mask'], output_hidden_states=True)
len(hidden_rep)
hidden_rep[0].shape
hidden_rep[-1].shape
