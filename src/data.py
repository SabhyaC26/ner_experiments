from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from util.util import device, UNK


class Conll2003(Dataset):
    def __init__(self, tokens: List[List[str]], labels: List[List[int]],
                 idx_to_tokens: Dict[int, str], tokens_to_idx: Dict[str, int],
                 idx_to_tags: Dict[int, str], tags_to_idx: Dict[str, int]):
        self.tokens = tokens
        self.labels = labels
        self.tags_to_idx = tags_to_idx
        self.idx_to_tags = idx_to_tags
        self.tokens_to_idx = tokens_to_idx
        self.idx_to_tokens = idx_to_tokens

    def process_token_str(self, token_str: List[str]) -> List[torch.LongTensor]:
        processed_token_str = []
        for token in token_str:
            if token in self.tokens_to_idx:
                processed_token_str.append(self.tokens_to_idx[token])
            else:
                processed_token_str.append(self.tokens_to_idx[UNK])
        processed_token_str = torch.LongTensor(processed_token_str).to(device)
        return processed_token_str

    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, torch.LongTensor]:
        token_tensor = self.process_token_str(self.tokens[idx])
        label_tensor = torch.LongTensor(self.labels[idx])
        return token_tensor, label_tensor

    def __len__(self) -> int:
        return len(self.tokens)
