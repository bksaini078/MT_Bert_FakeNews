from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import tensorflow as tf
from numpy.random import seed
from sklearn.metrics import classification_report

from Mean_Teacher.clf.bert import BERT
from logger import logger

MODELS = {
    'bert': BERT,
    'pi_bert': NotImplemented,
    'mean_bert': NotImplemented
}


def run(args):
    data_dir = Path(args.data_folder) / args.data
    model_output_dir = Path(args.model_output_folder)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    ensure_reproducibility(args.seed)
    model = MODELS[args.model](args)

    if args.do_train:
        train_fname = data_dir / 'train.tsv'
        train_data = pd.read_csv(train_fname, sep='\t')
        model.train(train_data=train_data)
        model.save_weights(f'{model_output_dir}/{args.model}.h5')

    model.load_weights(f'{model_output_dir}/{args.model}.h5')
    test_fname = data_dir / 'test.tsv'
    test_data = pd.read_csv(test_fname, sep='\t')
    results = model.predict(test_data)
    logger.info("=======Test Results======")
    logger.info(classification_report(y_true=test_data['label'], y_pred=results['labels'], digits=4))
    pd.DataFrame(results).to_csv(f'{model_output_dir}/{args.model}.results.tsv')


def ensure_reproducibility(random_seed):
    seed(random_seed)
    tf.random.set_seed(random_seed)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, choices=['bert'])
    parser.add_argument('--pretrained_model', type=str, choices=['bert-base-uncased'])
    parser.add_argument('--data', type=str, choices=['fakehealth'])
    parser.add_argument('--data_folder', type=str)
    parser.add_argument('--model_output_folder', type=str)
    parser.add_argument('--max_len', type=int)
    parser.add_argument('--dropout', type=float)
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--seed', type=int)

    args = parser.parse_args()
    run(args)
