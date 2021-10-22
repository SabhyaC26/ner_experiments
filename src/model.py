import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
import allennlp.modules.conditional_random_field as crf

class BiLSTM_CRF(nn.Module):
  def __init__(self, device, vocab_size, num_tags, embedding_dim=300,
               lstm_hidden_dim=300, lstm_num_layers=1, dropout=0.2,
               constraints=None):
    super(BiLSTM_CRF, self).__init__()
    self.device = device
    self.vocab_size = vocab_size
    self.num_tags = num_tags
    self.constraints=constraints
    # TODO: change to pretrained embeddings
    self.embeddings = nn.Embedding(
      num_embeddings=self.vocab_size,
      embedding_dim=embedding_dim
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
    # Conditional Random Field for Decoding
    self.crf = crf.ConditionalRandomField(
      num_tags=self.num_tags,
      constraints=self.constraints
    )

  def forward(self, input, input_lens):
    # max_length = input.shape[1]
    embedded = self.dropout(self.embeddings(input))
    packed_embedded = rnn.pack_padded_sequence(embedded, input_lens, batch_first=True, enforce_sorted=False)
    packed_output, hidden = self.lstm(packed_embedded)
    output, _ = rnn.pad_packed_sequence(packed_output, batch_first=True)
    self.output = self.dropout(output)
    output = self.linear(output)
    output.to(self.device)
    return output