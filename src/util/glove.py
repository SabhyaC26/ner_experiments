import os
import numpy as np
import torch

from typing import Dict

def load_glove_embeddings(glove_folder: str, embedding_dim: int, init: str,
                          tokens_to_idx: Dict[str, int]) -> torch.Tensor:
    assert (embedding_dim in {50, 100, 200, 300})
    glove_path = os.path.join(glove_folder, f'glove.6B.{embedding_dim}d.txt')
    max_idx = len(tokens_to_idx.keys())
    if init == 'zeros':
        embedding_matrix = np.zeros((max_idx+1, embedding_dim))
    elif init == 'random':
        embedding_matrix = np.random.rand(max_idx+1, embedding_dim)
    else:
        raise ValueError('Illegal embedding initialization type!')
    with open(glove_path) as glove_file:
        for i, line in enumerate(glove_file):
            elements = line.split(' ')
            token = elements[0]
            if token not in tokens_to_idx.keys():
                continue
            token_idx = tokens_to_idx[token]
            if token_idx <= max_idx:
                embedding_matrix[token_idx] = np.asarray(elements[1:], dtype='float32')
    return torch.from_numpy(embedding_matrix).float()



if __name__ == '__main__':
    embedding_matrix = load_glove_embeddings(glove_folder='embeddings/glove/', embedding_dim=50, init='zeros', )
    print(embedding_matrix[0])
    print(embedding_matrix[1])
    print(embedding_matrix[100])
