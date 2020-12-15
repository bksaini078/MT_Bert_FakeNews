from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from Mean_Teacher.clf.bert import BERT

MODELS = {
    'bert': BERT,
    'pi_bert': NotImplemented,
    'mean_bert': NotImplemented
}


def run(args):
    data_dir = Path(args.data_folder) / args.data
    model_output_dir = Path(args.model_output_folder)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    model = MODELS[args.model](args)

    # TODO add path check
    if args.do_train:
        train_fname = data_dir / 'train_0.tsv'
        train_data = pd.read_csv(train_fname, sep='\t')[:10]
        model.train(train_data=train_data)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, choices=['bert'])
    parser.add_argument('--pretrained_model', type=str, choices=['bert-base-uncased'])
    parser.add_argument('--data', type=str, choices=['fakehealth'])
    parser.add_argument('--data_folder', type=str)
    parser.add_argument('--model_output_folder', type=str)
    parser.add_argument('--max_len', type=int)
    parser.add_argument('--do_train', action='store_true')

    args = parser.parse_args()
    run(args)
