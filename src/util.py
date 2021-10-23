import torch
import torch.nn.utils.rnn as rnn
from typing import List, Tuple

def pad_batch(batch: Tuple[torch.LongTensor, torch.LongTensor]) \
    -> Tuple[torch.LongTensor, torch.LongTensor, torch.LongTensor]:
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  (xs, ys) = zip(*batch)
  x_lens = torch.LongTensor([len(x) for x in xs])
  x_pad = rnn.pad_sequence(xs, padding_value=0, batch_first=True)
  x_pad = x_pad.to(device)
  y_pad = rnn.pad_sequence(ys, padding_value=0, batch_first=True)
  y_pad = y_pad.to(device)
  return x_pad, x_lens, y_pad

def pad_test_batch(batch: torch.LongTensor) -> Tuple[torch.LongTensor, torch.LongTensor]:
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  x_lens = torch.LongTensor([len(x) for x in batch])
  x_pad= rnn.pad_sequence([x for x in batch], padding_value=0, batch_first=True)
  x_pad = x_pad.to(device)
  return x_pad, x_lens

def format_output_labels(tags):
  pass

def entity_level_mean_f1(preds, gold):
  pass