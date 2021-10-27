import argparse

import torch
from torch.utils.data import DataLoader

from data.conll_char import Conll2003_Char
from util.conll_util import *
from util.util import *


def main(args):
    # init logging
    logger = init_logger(args.out, args.run_id, 'BiLSTM-CNN-CRF')

    # mappings + datasets + dataloaders
    train, val, test = load_data()
    # test = test.select(range(100))
    ner_tags = train.features['ner_tags'].feature.names
    tokens_to_idx, idx_to_tokens = build_token_mappings(train['tokens'])
    tags_to_idx, idx_to_tags = build_tag_mappings(ner_tags)
    chars_to_idx, idx_to_chars = build_char_mappings(train['tokens'])

    train_data = Conll2003_Char(
        tokens=train['tokens'], labels=train['ner_tags'],
        idx_to_tokens=idx_to_tokens, tokens_to_idx=tokens_to_idx,
        idx_to_tags=idx_to_tags, tags_to_idx=tags_to_idx,
        idx_to_chars=idx_to_chars, chars_to_idx=chars_to_idx
    )
    val_data = Conll2003_Char(
        tokens=val['tokens'], labels=val['ner_tags'],
        idx_to_tokens=idx_to_tokens, tokens_to_idx=tokens_to_idx,
        idx_to_tags=idx_to_tags, tags_to_idx=tags_to_idx,
        idx_to_chars=idx_to_chars, chars_to_idx=chars_to_idx
    )

    # @todo: will need a new collate function for this
    # train_dataloader = DataLoader(dataset=train_data, batch_size=args.batch_size, shuffle=True, collate_fn=pad_batch)
    # val_dataloader = DataLoader(dataset=val_data, batch_size=args.batch_size, shuffle=True, collate_fn=pad_batch)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args for BiLSTM_CRF')
    parser.add_argument('--out', help='output directory for logs', required=True, type=str)

    parser.add_argument('--run-id', help='id for the current run', type=int, required=True)
    args = parser.parse_args()
    main(args)