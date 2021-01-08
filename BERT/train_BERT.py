from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import tensorflow as tf
from numpy.random import seed
from sklearn.metrics import classification_report

from BERT.bert import BERT
from logger import logger

MODELS = {
    'bert': BERT,
    'pi_bert': NotImplemented,
    'mean_bert': NotImplemented
}


def train_bert(args):
    data_folder = Path(args.data_folder)
    data_dir = data_folder / args.data
    model_output_dir = Path(args.model_output_folder)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    ensure_reproducibility(args.seed)
    model = MODELS[args.model](args)

    test_fname = data_dir / 'test.tsv'
    test_data = pd.read_csv(test_fname, sep='\t')
    # test_data=test_data[:30] 

    model_name = f'{model_output_dir}/{args.pretrained_model}_seed_{args.seed}_portion_{data_folder.name}_{data_dir.name}.model'
    results_name = f'{model_output_dir}/{args.pretrained_model}_seed_{args.seed}_portion_{data_folder.name}_{data_dir.name}_results.tsv'
    if args.do_train:
        train_fname = data_dir / 'train.tsv'
        train_data = pd.read_csv(train_fname, sep='\t')
        # train_data= train_data[:200]
        model.train(train_data=train_data, test_data=test_data)
        model.save_weights(model_name)
        logger.info(f"Model saved to {model_name}")
    model.load_weights(model_name)
    logger.info(f"Model loaded from {model_name}")
    results = model.predict(test_data)
    logger.info("=======Test Results======")
    logger.info(classification_report(y_true=test_data['label'], y_pred=results['labels'], digits=4))

    pd.DataFrame(results['labels']).to_csv(results_name)


def ensure_reproducibility(random_seed):
    seed(random_seed)
    tf.random.set_seed(random_seed)


# if __name__ == '__main__':
#     parser = ArgumentParser()
#     parser.add_argument('--model', type=str, choices=['bert'])
#     parser.add_argument('--pretrained_model', type=str, choices=['bert-base-uncased', 'bert-base-cased'])
#     parser.add_argument('--data', type=str, choices=['fakehealth','Kaggle', 'gossipcop'])
#     parser.add_argument('--data_folder', type=str)
#     parser.add_argument('--model_output_folder', type=str)
#     parser.add_argument('--max_len', type=int)
#     parser.add_argument('--dropout', type=float)
#     parser.add_argument('--do_train', action='store_true')
#     parser.add_argument('--epochs', type=int)
#     parser.add_argument('--batch_size', type=int)
#     parser.add_argument('--lr', type=float)
#     parser.add_argument('--seed', type=int)
#
#     args = parser.parse_args()
#     run(args)
