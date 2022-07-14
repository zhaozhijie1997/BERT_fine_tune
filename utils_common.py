import json, time
import numpy as np 
import pandas as pd
# import fasttext

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup

class Triage(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        
    def __getitem__(self, index):
        title = str(self.data.TITLE[index])
        title = " ".join(title.split())
        inputs = self.tokenizer.encode_plus(
            title,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True,
            truncation=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']

        return_data = {
            'ids': torch.tensor(ids, dtype=torch.long),
            'mask': torch.tensor(mask, dtype=torch.long),
                      }

        if 'ENCODE_CAT' in self.data.columns:
            return_data['targets'] =  torch.tensor(self.data.ENCODE_CAT[index], dtype=torch.long)
        return return_data
    
    def __len__(self):
        return self.len


class Textprocess(Dataset):
    def __init__(self, data, seq_len, embedding=None, vocab=None):
        self.len = len(data)
        self.data = data
        self.seq_len = seq_len
        if embedding:
            self.embedding = fasttext.load_model(embedding)
            self.vocab = dict(zip(self.embedding.words, [i for i in range(len(self.embedding.words))]))
        if vocab:
            self.vocab = vocab
        self.ids = None 
        self.ids_padded = None
        
        self.word_to_idx()
        self.padding_sentences()

    def word_to_idx(self):
        self.ids = list() 
        for sentence in self.data.TITLE:
            temp_sentence = list()
            for word in sentence.split():
                if word in self.vocab:
                    temp_sentence.append(self.vocab[word])
                else:
                    continue
            self.ids.append(temp_sentence)

    def padding_sentences(self):
        # Each sentence which does not fulfill the required le
        # it's padded with the index 0
        self.ids_padded = list()

        for sentence in self.ids:
            if len(sentence) > self.seq_len:
                self.ids_padded.append(sentence[:self.seq_len])
            else:
                while len(sentence) < self.seq_len:
                    sentence.insert(len(sentence), self.vocab['pad'])
                self.ids_padded.append(sentence)

        self.ids_padded = np.array(self.ids_padded)


    def __getitem__(self, index):        
        ids_padded = self.ids_padded[index]
        return_data = {
            'ids': torch.tensor(ids_padded, dtype=torch.long),
                      }

        if 'ENCODE_CAT' in self.data.columns:
            return_data['targets'] =  torch.tensor(self.data.ENCODE_CAT[index], dtype=torch.long)
        return return_data
    
    def __len__(self):
        return self.len

