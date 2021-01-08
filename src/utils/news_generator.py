import pandas as pd
import numpy as np
from transformers import pipeline, set_seed
from logger import logger
import argparse
import json
from datetime import datetime
from tqdm import tqdm


def generate_grover_jsonl(args):
    path = f'{args.data_folder}/{args.data}/train.tsv'
    df = pd.read_csv(path, sep='\t', header='infer')
    data = df[df['label'] == args.data_type]

    news_for_generation = []
    for idx, news in tqdm(data.iterrows(), total=len(data)):
        date = datetime.strptime(news['publish_date'][:10], "%Y-%m-%d")
        news_for_generation.append({'title': news['title'],
                                    'text': news['content'],
                                    "summary": "null",
                                    "authors": [],
                                    "publish_date": f"{date.month}-{date.day}-{date.year}",
                                    "url": news['url'],
                                    "domain": ""})

    with open(args.output_folder, 'w') as f:
        f.write('\n'.join(map(json.dumps, news_for_generation)))

    logger.info(f'Data for Grover news generation is saved to {path}')


def transform_grover_outputs_to_df(args):
    path = f'{args.output_folder}/{args.data}/unlabeled.tsv'

    with open(args.data_folder, 'r') as json_file:
        news_list = list(json_file)

    news_to_save = []
    for news in news_list:
        result = json.loads(news)
        date = datetime.strptime(result['publish_date'][:10], "%m-%d-%Y")
        result["publish_date"] = f"{date}"
        result["content"] = result["text"]
        result["news_id"] = result["url"]
        result["label"] = None
        news_to_save.append(result)
    processed_news = pd.DataFrame(news_to_save)[["content", "title", "publish_date", "url", "news_id", "label"]]
    processed_news.to_csv(path, index=False, sep='\t')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--data_folder', type=str)
    parser.add_argument('--output_folder', type=str)
    parser.add_argument('--data_type', choices=['fake', 'true'])
    parser.add_argument('--generate', action='store_true')
    parser.add_argument('--transform', action='store_true')
    args = parser.parse_args()

    if args.generate:
        generate_grover_jsonl(args)

    if args.transform:
        transform_grover_outputs_to_df(args)
