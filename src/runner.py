import argparse
import datasets
import torch
import allennlp.modules.conditional_random_field as crf
###
from torch.utils.data import DataLoader
from tqdm import tqdm
from data import Conll2003, UNK, PAD
from util import pad_batch, pad_test_batch
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

def train_model(model, dataloader, optimizer, clip):
  model.train()
  epoch_loss = 0
  with tqdm(dataloader, unit='batch') as tqdm_loader:
    for x_padded, x_lens, y_padded in tqdm_loader:
      optimizer.zero_grad()
      result = model(x_padded, x_lens, y_padded, decode=False)
      neg_log_likelihood = result['loss']
      neg_log_likelihood.backward()
      torch.nn.utils.clip_grad_norm(model.parameters(), clip)
      optimizer.step()
      epoch_loss += neg_log_likelihood.item()
  return epoch_loss/len(dataloader.dataset)

def evaluate_model(model, dataloader):
  model.eval()
  epoch_loss = 0
  with torch.no_grad():
    with tqdm(dataloader, unit='batch') as tqdm_loader:
      for x_padded, x_lens, y_padded in tqdm_loader:
        result = model(x_padded, x_lens, y_padded, decode=False)
        neg_log_likelihood = result['loss']
        epoch_loss += neg_log_likelihood.item()
  return epoch_loss/len(dataloader.dataset)

# batch decoding - hasn't been tested
def decode_batch(model, batch, idx_to_tags):
  model.eval()
  with torch.no_grad():
    padded_batch = pad_test_batch(batch)
    x_padded, x_lens = padded_batch
    result = model(x_padded, x_lens, None, decode=True)
    predicted_tags = result['tags']
    actual_pred_tags = []
    for pred in predicted_tags:
      actual_pred_tags.append([idx_to_tags[i] for i in pred])
  return actual_pred_tags

def main(args):
  # build dataset & dataloader
  train, val, test = load_data()
  ner_tags = train.features['ner_tags'].feature.names
  device = get_device()
  train_data = Conll2003(examples=train['tokens'][:10], labels=train['ner_tags'][:10], ner_tags=ner_tags, device=device)
  val_data = Conll2003(examples=val['tokens'], labels=val['ner_tags'], ner_tags=ner_tags, device=device)
  train_dataloader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, collate_fn=pad_batch)
  val_dataloader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=True, collate_fn=pad_batch)

  # define model
  crf_constraints = crf.allowed_transitions(
    constraint_type='BIO',
    labels=train_data.idx_to_tags
  )
  bilstm_crf = BiLSTM_CRF(
    device=device,
    vocab_size=len(train_data.idx_to_tokens.keys()),
    num_tags=len(train_data.idx_to_tags.keys()),
    embedding_dim=args.embedding_dim,
    lstm_hidden_dim=args.hidden_dim,
    lstm_num_layers=args.num_layers,
    dropout=args.dropout,
    constraints=crf_constraints,
    pad_idx=train_data.tokens_to_idx[PAD]
  )
  bilstm_crf.to(device)

  # run model
  optimizer = torch.optim.Adam(bilstm_crf.parameters())
  train_model(model=bilstm_crf, dataloader=train_dataloader, optimizer=optimizer, clip=args.clip)

if __name__ == '__main__':
  # TODO: implement parsing command line args
  # embedidng dim, hidden dim, num layers, dropout, epochs, batch size, clip
  parser = argparse.ArgumentParser(description='Args BiLSTM_CRF')
  parser.add_argument('--out', help='output directory for logs', required=True)
  parser.add_argument('--emebedding-dim', help='dimension for embeddings', default=300)
  parser.add_argument('--hidden-dim', help='dimension for hidden layer', default=512)
  parser.add_argument('--num-layers', help='number of lstm layers', default=1)
  parser.add_argument('--dropout', help='regularization parameter', default=0.2)
  parser.add_argument('--batch-size', help='train/val batch size', default=64)
  parser.add_argument('--clip', help='gradient clipping parameter', default=1)
  args = parser.parse_args()
  main(args)