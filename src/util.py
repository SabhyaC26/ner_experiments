import torch
import torch.nn.utils.rnn as rnn

def pad_batch(batch):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  (xs, ys) = zip(*batch)
  x_lens = [len(x) for x in xs]
  y_lens = x_lens
  x_pad = rnn.pad_sequence(xs, padding_value=0, batch_first=True)
  x_pad = x_pad.to(device)
  y_pad = rnn.pad_sequence(ys, padding_value=0, batch_first=True)
  y_pad = y_pad.to(device)
  return x_pad, y_pad, x_lens, y_lens
