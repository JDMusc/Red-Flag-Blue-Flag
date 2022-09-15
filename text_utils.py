from toolz import pipe as p
from typing import Union

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import tqdm

import dplyr_lite as d


tokenizer = lambda _: _.split()

class Vocab(object):
    def __init__(self, vocab:dict, numericizer:dict, unk_num: int):
        self.numericizer = numericizer
        self.vocab = vocab
        self.unk_num = unk_num
    
    @staticmethod
    def tokenize(txt:str):
        return txt.split()
    
    def numericize(self, tokens:Union[list,str]):
        if type(tokens) is str:
            return self.numericize(Vocab.tokenize(tokens))
        
        return [self.numericizer.get(t, self.unk_num) for t in tokens]

    
def create_vocab(abstracts:pd.DataFrame, 
                tokenizer = tokenizer,
                cutoff_freq:int = 4,
                ):
    vocab_no_cutoff = {}
    for _, row in tqdm.tqdm(abstracts.iterrows(), total = len(abstracts)):
        tokens = tokenizer(row.txt)
        for t in tokens:
            vocab_no_cutoff[t] = vocab_no_cutoff.get(t, 0) + 1
    
    vocab_dict = {k: v for k, v in vocab_no_cutoff.items() if v >= cutoff_freq}

    vocab_ranks = d.p(
        vocab_dict.keys(),
        d.curry(pd.DataFrame, columns=['term']),
        d.mutate(freq = vocab_dict.values()),
        d.arrange('freq', ascending = False),
        d.apply('reset_index'),
        d.mutate(term_rank = lambda _: list(range(len(_)))),
        d.select(['term', 'term_rank', 'freq']),
    )

    unk_num = max(vocab_ranks.term_rank) + 1

    numericizer_dict = {
        r['term']: r['term_rank'] for _, r in vocab_ranks.iterrows()
    }

    return Vocab(
        vocab=vocab_dict,
        numericizer=numericizer_dict,
        unk_num=unk_num
    )


def numericize_sample(sample:str, vocab:Vocab, sample_len:int):
    nums = d.p(sample, vocab.numericize, np.array)
    n_nums = len(nums)
    pad = sample_len - n_nums

    return (
        nums[:sample_len] if pad <= 0
        else np.pad(nums, pad_width = (0, pad), constant_values = 0)
    )


class TextDataset(Dataset):
    def __init__(self, 
        abstracts:pd.DataFrame,
        vocab:Vocab,
        sample_len:int
    ):
        self.data = abstracts
        self.vocab = vocab
        self.sample_len = sample_len
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data.txt.values[idx]
        y = self.data.label_hep.values[idx]
        return (
            numericize_sample(sample, self.vocab, self.sample_len),
            y
        )
    

def ixs_to_dataloader(ixs, abstracts, vocab, sample_len,
                    bs = 64, shuffle = True):
    return p(
        TextDataset(abstracts.iloc[ixs,: ], vocab, sample_len),
        d.curry(DataLoader, 
            batch_size = bs, shuffle = shuffle,
            collate_fn = collate_batch
        )
    )

def collate_batch(batch, device = 'cuda'):
    xs = torch.tensor([_[0] for _ in batch]).to(device)
    ys = torch.tensor([_[1] for _ in batch]).to(device)

    return xs, ys