import collections
import os
import sys
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

# @todo: need to find a better way to do imports
from ..util.conll_util import UNK
from ..util.util import device


class Conll2003_Char(Dataset):
    def __init__(self, tokens: List[List[str]], labels: List[List[int]],
                 idx_to_tokens: Dict[int, str], tokens_to_idx: Dict[str, int],
                 idx_to_tags: Dict[int, str], tags_to_idx: Dict[str, int],
                 idx_to_chars: Dict[int, str], chars_to_idx: Dict[str, int]):
        self.tokens = tokens
        self.labels = labels
        self.tags_to_idx = tags_to_idx
        self.idx_to_tags = idx_to_tags
        self.tokens_to_idx = tokens_to_idx
        self.idx_to_tokens = idx_to_tokens
        self.chars_to_idx = chars_to_idx
        self.idx_to_chars = idx_to_chars

    def process_token_str(self, token_str: List[str]) -> Tuple[torch.LongTensor, List[torch.LongTensor]]:
        processed_token_str = []
        processed_char_tokens = []
        for idx, token in enumerate(token_str):
            if token in self.tokens_to_idx:
                processed_token_str.append(self.tokens_to_idx[token])
            else:
                processed_token_str.append(self.tokens_to_idx[UNK])
            char_lst = []
            for char in token:
                if char in self.chars_to_idx:
                    char_lst.append(self.chars_to_idx[char])
                else:
                    char_lst.append(self.chars_to_idx[UNK])
            processed_char_tokens.append(char_lst)
        processed_token_str = torch.LongTensor(processed_token_str).to(device)
        return processed_token_str, processed_char_tokens

    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, List[torch.LongTensor], torch.LongTensor]:
        token_tensor, char_tensor_lst = self.process_token_str(self.tokens[idx])
        label_tensor = torch.LongTensor(self.labels[idx])
        return token_tensor, char_tensor_lst, label_tensor

    def __len__(self) -> int:
        return len(self.tokens)
