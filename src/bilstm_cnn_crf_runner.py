import torch
import argparse

from util.util import *
from util.conll_util import *

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
    print(chars_to_idx)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Args for BiLSTM_CRF')
    parser.add_argument('--out', help='output directory for logs', required=True, type=str)

    parser.add_argument('--run-id', help='id for the current run', type=int, required=True)
    args = parser.parse_args()
    main(args)