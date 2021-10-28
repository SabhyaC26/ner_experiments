import collections
import datasets
from typing import List, Tuple, Dict, Union

UNK = '<UNK>'
PAD = '<P>'


def load_data() -> Tuple[Union[dict, list], Union[dict, list], Union[dict, list]]:
    conll_dataset = datasets.load_dataset('conll2003')
    train = conll_dataset['train']
    val = conll_dataset['validation']
    test = conll_dataset['test']
    return train, val, test


def build_token_mappings(examples: List[List[str]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    vocab = set()
    for example in examples:
        for token in example:
            vocab.add(token)
    tokens_to_idx = collections.defaultdict(int)
    idx_to_tokens = collections.defaultdict(str)
    tokens_to_idx[PAD] = 0
    idx_to_tokens[0] = PAD
    tokens_to_idx[UNK] = 1
    idx_to_tokens[1] = UNK
    for i, token in enumerate(sorted(vocab)):
        tokens_to_idx[token] = i + 2
        idx_to_tokens[i + 2] = token
    return tokens_to_idx, idx_to_tokens


def build_tag_mappings(ner_tags: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
    tags_to_idx = collections.defaultdict(int)
    idx_to_tags = collections.defaultdict(str)
    for i, tag in enumerate(ner_tags):
        tags_to_idx[tag] = i
        idx_to_tags[i] = tag
    return tags_to_idx, idx_to_tags

def build_char_mappings(examples: List[List[str]]) -> Tuple[Dict[str, int], Dict[int, str]]:
    char_vocab = set()
    for example in examples:
        for token in example:
            for char in token:
                char_vocab.add(char)
    chars_to_idx = collections.defaultdict(int)
    idx_to_chars = collections.defaultdict(str)
    chars_to_idx[PAD] = 0
    idx_to_chars[0] = PAD
    chars_to_idx[UNK] = 1
    idx_to_chars[1] = UNK
    for i, token in enumerate(sorted(char_vocab)):
        chars_to_idx[token] = i + 2
        idx_to_chars[i + 2] = token
    return chars_to_idx, idx_to_chars
