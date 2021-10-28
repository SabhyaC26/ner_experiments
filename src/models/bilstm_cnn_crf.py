from typing import Dict, List, Optional, Tuple

import allennlp.modules.conditional_random_field as crf
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn

from ..util.util import device


class BiLSTM_CNN_CRF(nn.Module):
    def __init__(self):
        super(BiLSTM_CNN_CRF, self).__init__()

    # @todo figure out the mechanics of char tensors word tensors
    def forward(self, src: torch.LongTensor, char: torch.LongTensor):
        pass
