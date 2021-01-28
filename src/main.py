from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from numpy.random import seed
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

from logger import logger
from reader import mask_labels
from src.clf.bert import BERT
from src.clf.mt_bert import MTBERT

MODELS = {
    'bert': BERT,
    'pi_bert': NotImplemented,
    'mean_bert': MTBERT
}


def ensure_reproducibility(random_seed):
    seed(random_seed)
    tf.random.set_seed(random_seed)


def train_unlabel_split(train_data, ratio):
    '''This part we are doing because of having balanced train data during training '''
    data_n_unlabel = round(ratio * len(train_data))
    data_n_train = len(train_data) - data_n_unlabel
    train_data_sort = train_data.sort_values(by=['label']).reset_index(drop=True)
    train = train_data_sort.head(round(data_n_train / 2))
    train = train.append(train_data_sort.tail(round(data_n_train / 2)))
    unlabel_data = train_data_sort[round(data_n_train / 2) + 1:data_n_unlabel + round(data_n_train / 2) + 1]
    train = train.sample(frac=1, random_state=42).reset_index(drop=True)
    unlabel_data = unlabel_data.sample(frac=1, random_state=42).reset_index(drop=True)
    unlabel_data['label'] = np.full((len(unlabel_data), 1), -1)
    return train, unlabel_data


def run(args):
    data_folder = Path(args.data_folder)
    data_dir = data_folder / args.data
    model_output_dir = Path(args.model_output_folder)
    model_output_dir.mkdir(parents=True, exist_ok=True)
    ensure_reproducibility(args.seed)
    model = MODELS[args.model](args)

    test_fname = data_dir / 'test.tsv'
    test_data = pd.read_csv(test_fname, sep='\t')

    model_name = f"{model_output_dir}/{args.pretrained_model}_{args.model}_seed_{args.seed}_portion_{data_folder.name}_{data_dir.name}"
    results_name = f"{model_output_dir}/{args.pretrained_model}_{args.model}_seed_{args.seed}_portion_{data_folder.name}_{data_dir.name}_results.tsv"
    if args.do_train or not Path(model_name).exists():
        train_fname = data_dir / 'train.tsv'
        train_data = pd.read_csv(train_fname, sep='\t')

        if args.ratio_label:
            train_data = mask_labels(train_data, args.ratio_label)
        if args.use_noise_data:
            noise_fname = data_dir / 'unlabeled.tsv'
            noise_data = pd.read_csv(noise_fname, sep='\t')
            train_data = train_data.sample(frac=1, random_state=args.seed).reset_index(drop=True)
            # spliting according to the label and unlabel ratio
            # train_data, unlabel_data= train_unlabel_split(train_data, args.split_ratio)
            # print(train_data,unlabel_data)
            # noise data
            noise_data = noise_data.sample(frac=1, random_state=args.seed).reset_index(drop=True)
            model.train(train_data=train_data, noise_data=noise_data)
        else:
            train_data = train_data.sample(frac=1, random_state=args.seed).reset_index(drop=True)
            model.train(train_data=train_data)

        model.save_weights(model_name)
        logger.info(f"Model saved to {model_name}")

    model.load_weights(model_name)
    logger.info(f"Model loaded from {model_name}")
    results = model.predict(test_data)
    logger.info("Test Results")
    logger.info(classification_report(y_true=test_data['label'], y_pred=results['labels'], digits=4))
    logger.info("F1-Macro")
    logger.info(f1_score(y_true=test_data['label'], y_pred=results['labels'], average='macro'))
    logger.info("F1-Micro")
    logger.info(f1_score(y_true=test_data['label'], y_pred=results['labels'], average='micro'))
    logger.info("F1-Weighted")
    logger.info(f1_score(y_true=test_data['label'], y_pred=results['labels'], average='weighted'))
    pd.DataFrame(results['labels']).to_csv(results_name)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, choices=['bert', 'mean_bert'], default='bert')
    parser.add_argument('--pretrained_model', type=str, choices=['distilbert-base-uncased', 'bert-base-uncased'],
                        default='distilbert-base-uncased')
    parser.add_argument('--data', type=str, choices=['fakehealth'], default='fakehealth')
    parser.add_argument('--data_folder', type=str, default='Data/ExperimentFolds/3')
    parser.add_argument('--model_output_folder', type=str, default='trained_models')
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--do_train', action='store_false')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--loss_weight', type=float, default=0.5)
    parser.add_argument('--use_noise_data', action='store_true')
    parser.add_argument('--noise_batch', type=int, default=2)
    parser.add_argument('--decay', type=float, default=0.99)
    parser.add_argument('--model_option', type=str, choices=['student', 'teacher'], default='teacher')
    parser.add_argument('--loss_type', type=str, choices=['kl_divergence', 'mse'], default='mse')
    # to split the train dataset in label and unlabel data
    parser.add_argument('--split_ratio', type=int, default=0.9)
    # parser.add_argument('--use_unlabel',action='store_true')
    # unlabel ratio in which during training unlabel and label data will make batch
    parser.add_argument('--ratio_label', type=float)

    args = parser.parse_args()
    run(args)
