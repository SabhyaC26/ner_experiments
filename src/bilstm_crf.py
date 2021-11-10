from typing import Dict, List, Optional, Tuple

import allennlp.modules.conditional_random_field as crf
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

from src.util.util import device


class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size: int, num_tags: int, embedding_dim: int,
                 embeddings: Optional[torch.tensor], lstm_hidden_dim: int,
                 lstm_num_layers: int, dropout: float, constraints: Optional[List[Tuple[int, int]]],
                 pad_idx: int):
        super(BiLSTM_CRF, self).__init__()
        self.vocab_size = vocab_size
        self.num_tags = num_tags
        self.constraints = constraints
        self.pad_idx = pad_idx
        self.embeddings = embeddings
        # Embeddings
        if self.embeddings is not None:
            self.embedding_layer = nn.Embedding.from_pretrained(
                embeddings=embeddings,
                freeze=False,
                padding_idx=0
            )
        else:
            self.embedding_layer = nn.Embedding(
                num_embeddings=self.vocab_size,
                embedding_dim=embedding_dim,
                padding_idx=0
            )
        # LSTM + Linear Layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=lstm_hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
            num_layers=lstm_num_layers
        )
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(
            in_features=lstm_hidden_dim,
            out_features=self.num_tags
        )
        # Conditional Random Field for Loss & Decoding
        self.crf = crf.ConditionalRandomField(
            num_tags=self.num_tags,
            constraints=self.constraints
        )

    """
    @todo Check for 0s in non pad idxs
    @body When fixing the masking, I saw 0s in non pad positions --> could be a bug
    """
    def create_mask(self, src: torch.LongTensor) -> torch.LongTensor:
        mask = (src != self.pad_idx).permute(0, 1)
        return mask

    def forward(self, src: torch.LongTensor, input_lens: torch.LongTensor,
                labels: torch.LongTensor, decode: bool) -> Dict[str, any]:
        # @todo check if dropout needs to be applied on embeddings
        embedded = self.dropout(self.embedding_layer(src))
        embedded.to(device)
        packed_embedded = rnn.pack_padded_sequence(embedded, input_lens, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.lstm(packed_embedded)
        out, _ = rnn.pad_packed_sequence(packed_output, batch_first=True)
        out = self.dropout(out)
        out = self.linear(out)
        out.to(device)
        # pass through crf
        mask = self.create_mask(src)
        result = {}
        if decode:
            result['tags'] = self.crf.viterbi_tags(logits=out, mask=mask)
        else:
            result['loss'] = -self.crf(inputs=out, tags=labels, mask=mask)
        return result
