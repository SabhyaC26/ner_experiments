import argparse
import time

import torch.optim
import wandb

import allennlp.modules.conditional_random_field as crf
from torch.utils.data import DataLoader
from tqdm import tqdm

from conll import Conll2003
from bilstm_crf import BiLSTM_CRF
from util.conll_util import *
from util.glove import load_glove_embeddings
from util.util import *


def train_model(model, dataloader, optimizer, clip: int) -> float:
    model.train()
    epoch_loss = 0
    with tqdm(dataloader, unit='batch') as tqdm_loader:
        for x_padded, x_lens, y_padded in tqdm_loader:
            optimizer.zero_grad()
            result = model(x_padded, x_lens, y_padded, decode=False)
            neg_log_likelihood = result['loss']
            wandb.log({'train_loss': neg_log_likelihood})
            neg_log_likelihood.backward()
            if clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_loss += neg_log_likelihood.item()
    return epoch_loss / len(dataloader.dataset)


def evaluate_model(model, dataloader) -> float:
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        with tqdm(dataloader, unit='batch') as tqdm_loader:
            for x_padded, x_lens, y_padded in tqdm_loader:
                result = model(x_padded, x_lens, y_padded, decode=False)
                neg_log_likelihood = result['loss']
                wandb.log({'val_loss': neg_log_likelihood})
                epoch_loss += neg_log_likelihood.item()
    return epoch_loss / len(dataloader.dataset)


def test_model(test_data, model, batch_size: int, tokens_to_idx: Dict[str, int],
               idx_to_tags: Dict[int, str]):
    with torch.no_grad():
        predictions = []
        for batch_idx in range((len(test_data) // batch_size) + 1):
            if (batch_size * (batch_idx + 1)) > len(test_data):
                batch = test_data.select(range(
                    batch_size * (batch_idx),
                    len(test_data)
                ))
            else:
                batch = test_data.select(range(
                    batch_size * batch_idx,
                    batch_size * (batch_idx + 1)
                ))
            encoded_tokens = []
            for token_seq in batch['tokens']:
                encoded_seq = []
                for token in token_seq:
                    if token in tokens_to_idx:
                        encoded_seq.append(tokens_to_idx[token])
                    else:
                        encoded_seq.append(tokens_to_idx[UNK])
                encoded_tokens.append(torch.LongTensor(encoded_seq))
            batch_predictions = decode_batch(model, encoded_tokens, idx_to_tags=idx_to_tags)
            for pred in batch_predictions:
                predictions.append(pred)
    return predictions


def decode_batch(model, batch: List[torch.LongTensor], idx_to_tags: Dict[int, str]):
    model.eval()
    with torch.no_grad():
        padded_batch = pad_test_batch(batch)
        x_padded, x_lens = padded_batch
        result = model(x_padded, x_lens, None, decode=True)
        actual_pred_tags = []
        for pred, _ in result['tags']:
            actual_pred_tags.append([idx_to_tags[i] for i in pred])
    return actual_pred_tags


def main(args, config, run_id):
    # init logger
    logger = init_logger(args.out, run_id, 'BiLSTM-CRF')
    logger.info('BiLSTM-CRF Model on Conll-2003 NER')

    # mappings + datasets + dataloaders
    train, val, test = load_data()
    # test = test.select(range(100))
    ner_tags = train.features['ner_tags'].feature.names
    tokens_to_idx, idx_to_tokens = build_token_mappings(train['tokens'])
    tags_to_idx, idx_to_tags = build_tag_mappings(ner_tags)
    train_data = Conll2003(
        tokens=train['tokens'], labels=train['ner_tags'],
        idx_to_tokens=idx_to_tokens, tokens_to_idx=tokens_to_idx,
        idx_to_tags=idx_to_tags, tags_to_idx=tags_to_idx
    )
    val_data = Conll2003(
        tokens=val['tokens'], labels=val['ner_tags'],
        idx_to_tokens=idx_to_tokens, tokens_to_idx=tokens_to_idx,
        idx_to_tags=idx_to_tags, tags_to_idx=tags_to_idx
    )
    train_dataloader = DataLoader(dataset=train_data, batch_size=config['batch_size'], shuffle=True, collate_fn=pad_batch)
    val_dataloader = DataLoader(dataset=val_data, batch_size=config['batch_size'], shuffle=True, collate_fn=pad_batch)

    # load glove embeddings
    logger.info('Loading glove embeddings.')
    glove_folder = args.glove
    glove_embeddings = load_glove_embeddings(glove_folder=glove_folder, embedding_dim=config['embedding_dim'],
                                             init='zeros', tokens_to_idx=tokens_to_idx)
    logger.info('Glove embeddings have been loaded')

    # define model
    crf_constraints = crf.allowed_transitions(
        constraint_type='BIO',
        labels=train_data.idx_to_tags
    )
    bilstm_crf = BiLSTM_CRF(
        vocab_size=len(train_data.idx_to_tokens.keys()),
        num_tags=len(train_data.idx_to_tags.keys()),
        embedding_dim=config['embedding_dim'],
        embeddings=glove_embeddings,
        lstm_hidden_dim=config['hidden_dim'],
        lstm_num_layers=config['num_layers'],
        dropout=config['dropout'],
        constraints=crf_constraints,
        pad_idx=train_data.tokens_to_idx[PAD]
    )
    bilstm_crf.to(device)

    # log number of model params
    num_params = count_parameters(bilstm_crf)
    logger.info(f'The model has {num_params:,} trainable parameters')

    # run model
    if config['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(bilstm_crf.parameters(), lr=config['lr'], momentum=config['momentum'])
    elif config['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(bilstm_crf.parameters(), lr=config['lr'])
    
    best_val_loss = float('inf')
    for epoch in range(config['epochs']):
        start_time = time.time()
        train_loss = train_model(model=bilstm_crf, dataloader=train_dataloader, optimizer=optimizer, clip=config['clip'])
        val_loss = evaluate_model(model=bilstm_crf, dataloader=val_dataloader)
        end_time = time.time()

        predicted_labels = test_model(test_data=test, model=bilstm_crf, batch_size=config['batch_size'],
                                      tokens_to_idx=tokens_to_idx, idx_to_tags=idx_to_tags)

        gold_labels = []
        for label_lst in test['ner_tags']:
            gold_labels.append([train_data.idx_to_tags[i] for i in label_lst])

        p, r, f1 = compute_entity_level_f1(predicted_labels=predicted_labels, gold_labels=gold_labels)
        wandb.log({'test_precision': p,
                   'test_recall': r,
                   'test_f1': f1,
                   'epoch': epoch})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            out_path = os.path.join(args.out, 'weights', f'bilstm_crf_{run_id}.pt')
            torch.save(bilstm_crf.state_dict(), out_path)

        epoch_mins, epoch_secs = calculate_epoch_time(start_time, end_time)
        logger.info(f"#######################EPOCH_{epoch + 1}#######################")
        logger.info(f'Epoch: {epoch + 1:02} | Time: {epoch_mins}m {epoch_secs}s')
        logger.info(f'\t Train Loss: {train_loss:.3f}')
        logger.info(f'\t Val. Loss: {val_loss:.3f}')
        logger.info(f'\t Test F1: {f1:.3f}')
        logger.info(f'\t Test Precision: {p:.3f}')
        logger.info(f'\t Test Recall: {r:.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Args for BiLSTM_CRF')
    parser.add_argument('--out', help='output directory for logs', required=True, type=str)
    parser.add_argument('--glove', help='path too the folder with glove files', type=str)
    parser.add_argument('--embedding_dim', help='embedding dimension', default=100, type=int)
    parser.add_argument('--hidden_dim', help='hidden dimension of lstm', default=256, type=int)
    parser.add_argument('--num_layers', help='number of lstm layers', default=1, type=int)
    parser.add_argument('--batch_size', help='batch size for train/val/test', default=64, type=int)
    parser.add_argument('--clip', help='gradient clipping threshold, 0 for no clipping', default=0, type=int)
    parser.add_argument('--dropout', help='dropout threshold', default=0.5, type=float)
    parser.add_argument('--epochs', help='number of train epochs', default=20, type=int)
    parser.add_argument('--lr', help='learning rate for optim', default=0.001, type=float)
    parser.add_argument('--optimizer', help='specify optimizer', default='sgd', type=str)
    parser.add_argument('--momentum', help='momentum if using sgd', default=0.9, type=float)
    args = parser.parse_args()

    wandb.init(project="ner_experiments", entity="sabhyac26", reinit=True)
    wandb.config = {
        "embedding_dim": args.embedding_dim,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "dropout": args.dropout,
        "batch_size": args.batch_size,
        "clip": args.clip,
        "epochs": args.epochs,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "momentum": args.momentum
    }
    config = wandb.config
    main(args=args, config=config, run_id=wandb.run.name)
    wandb.finish()
