from os import *
from os.path import isfile, join
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import argparse
from pathlib import Path
from Mean_Teacher.clf.bert import clean_helper

def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)
    sentence= sentence.lower()

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence
TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)

def data_preprocessing(data):
    #removing missing rows
    data=data.dropna()
    data= data.drop_duplicates()
    data= data.reset_index(drop=True)
    data.loc[:,'content']= data['content'].astype(str)
    data_1=data.copy()
   #cleaning data
    article= list(data['content'].values )
    for i in range(len(data)):
        # clean_text = clean_helper( article[i] )
        data_1.loc[i,'content']= clean_helper(article[i])

    #converting into np arrray and hot one encoding
    # np.array(y_train)
    return data_1
def processing(args):
    path = f'{args.data_folder}/{args.data}/'
    # path_save = Path(f'{args.processed_output_folder}/{args.data}/')
    path_save = Path(join(args.processed_output_folder,args.data))
    print(path_save)
    path_save.mkdir(parents=True, exist_ok=True)
    path_save= f'{args.processed_output_folder}/{args.data}/'
    onlyfiles = [f for f in listdir ( path ) if isfile ( join ( path, f ) )]

    for f in onlyfiles :
        df = pd.DataFrame ()
        if 'DS' in f :
            continue
        if 'unlabel' not in f :
            print ( f )
            df = pd.read_csv ( path + f, sep='\t', header='infer', usecols=["title", "content", "label"] )
            df = df.reset_index ( drop=True )
            df['content'] = df['content'].str.cat ( df['title'], sep="SEP" )
            df = df.drop ( ['title'], axis=1 )
            df_1 = data_preprocessing ( df )
            f = f.split ( '.' )[0]
            df_1.replace ( to_replace=['fake', 'true'], value=[0, 1], inplace=True )
            np.save ( path_save + f + '_x.npy', df_1['content'] )
            np.save ( path_save + f + '_y.npy', tf.one_hot ( np.array ( df_1['label'].values ), 1 ) )
        else :
            print ( f )
            df = pd.read_csv ( path + f, sep='\t', header='infer', usecols=["title", "content", "label"] )
            df = df.reset_index ( drop=True )
            df['content'] = df['title'].str.cat ( df['content'], sep="SEP" )
            df = df.drop ( ['title'], axis=1 )
            df_1 = data_preprocessing ( df )
            f = f.split ( '.' )[0]
            np.save (f'{path_save}{f}_x.npy', df_1['content'] )
    return
if __name__ == '__main__':
    parser= argparse.ArgumentParser()
    parser.add_argument ('--data', default='fakehealth', type=str )
    parser.add_argument('--data_folder', type=str)
    parser.add_argument('--processed_output_folder', type=str)
    args = parser.parse_args()
    processing(args)






