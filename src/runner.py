import argparse
import datasets
import torch
###
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import Conll2003, UNK, PAD
from util import pad_batch
from model import BiLSTM_CRF

def load_data():
  conll_dataset = datasets.load_dataset('conll2003')
  train_dataset = conll_dataset['train']
  valid_dataset = conll_dataset['validation']
  test_dataset = conll_dataset['test']
  return train_dataset, valid_dataset, test_dataset

def get_device():
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  return device

def train_model(model, dataloader):
  for x_padded, y_padded, x_lens, y_lens in dataloader:
    output = model(x_padded, x_lens)
    print(output.shape)

def main(args):
  # build dataset & dataloader
  train, val, test = load_data()
  ner_tags = train.features['ner_tags'].feature.names
  device = get_device()
  train_data = Conll2003(examples=train['tokens'][:10], labels=train['ner_tags'][:10], ner_tags=ner_tags, device=device)
  train_dataloader = DataLoader(dataset=train_data, batch_size=2, shuffle=True, collate_fn=pad_batch)

  # define model
  bilstm_crf = BiLSTM_CRF(
    device=device,
    vocab_size=len(train_data.idx_to_tokens.keys()),
    num_tags=len(train_data.idx_to_tags.keys()),
    embedding_dim=300,
    lstm_hidden_dim=512,
    lstm_num_layers=1,
    dropout=0.2)
  bilstm_crf.to(device)

  # run model
  train_model(model=bilstm_crf, dataloader=train_dataloader)

if __name__ == '__main__':
  # args
  # embedidng dim, hidden dim, num layers, dropout, epochs, batch size
  main(None)