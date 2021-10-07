import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import json
from collections import defaultdict, OrderedDict
import math
import numpy as np
from typing import List, Dict, Optional, Tuple
from TorchCRF import CRF

torch.manual_seed(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BiLSTM_CRF(nn.Module):

    def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tag_to_ix = tag_to_ix
        self.num_ne_tags = len(tag_to_ix)
        self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
        # bilstm
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim // 2,
            num_layers=1,
            bidirectional=True
        )
        # linear layer
        self.hidden_to_tag = nn.Linear(
            hidden_dim,
            self.tagset_size
        )
        # CRF - switch to AllenNLP CRF later
        self.crf = CRF(
            self.num_ne_tags,
            batch_first=True
        )
        # init hidden
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (torch.randn(2, 1, self.hidden_dim // 2).to(device),
                torch.randn(2, 1, self.hidden_dim // 2).to(device))

    def forward(self):
        pass



if __name__ == "__main__":

    conll_tag_to_ix = {
        "o": 0,
        "PER": 1,
        "ORG": 2,
        "LOC": 3,
        "MISC": 4,
    }

    model = BiLSTM_CRF(
        vocab_size=10,
        tag_to_ix=conll_tag_to_ix,
        embedding_dim=100,
        hidden_dim=100
    )

    print(model)