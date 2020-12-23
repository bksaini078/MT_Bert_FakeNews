import pandas as pd
import numpy as np
from transformers import pipeline
import argparse

def generate_fake_news(args):
    generator = pipeline('text-generation', model='gpt2')
    path = f'{args.data_folder}/{args.data}/'
    path_save = f'{args.processed_output_folder}/{args.data}/'
    df = pd.read_csv( path + 'train.tsv', sep='\t', header='infer', usecols=['content', 'title', 'label'])
    df = df.drop_duplicates()
    df = df.dropna()
    df = df.reset_index(drop=True)
    df_fake = df[df['label'] == 'fake']
    df_fake = df_fake.reset_index( drop=True)
    df_unlabel = pd.DataFrame()
    df_unlabel_l = pd.DataFrame()
    for i in range(len(df_fake)) :
        news_g = generator( df_fake.loc[i, 'title'], max_lenght=2000, num_return_sequences=5)
        for j in range(len(news_g)) :
            df_unlabel_l = df_unlabel_l.append({'title' : df_fake.loc[i, 'title']}, ignore_index=True)
        df_unlabel = df_unlabel.append(news_g, ignore_index=True)
    df_unlabel['title'] = df_unlabel_l['title']
    df_unlabel = df_unlabel.rename(columns={'generated_text' : 'content'})
    df_unlabel['label'] = np.full((len(df_unlabel), 1), 'fake')
    df_unlabel = df_unlabel.sample(frac=1).reset_index(drop=True)
    df_unlabel.to_csv(path_save + 'unlabel.tsv', sep='\t')
    return

if __name__=='__main__':
    parser= argparse.ArgumentParser()
    parser.add_argument('--data', default='fakehealth', type=str)
    parser.add_argument('--data_folder', type=str)
    parser.add_argument('--processed_output_folder', type=str)
    args = parser.parse_args()
    generate_fake_news(args)


