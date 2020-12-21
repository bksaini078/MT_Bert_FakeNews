from os import *
from os.path import isfile, join
import pandas as pd
import numpy as np
import tensorflow as tf
import re
import argparse

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
    data.loc[:,'Article']= data['Article'].astype(str)
    data_1=data.copy()
   #cleaning data
    article= list(data['Article'].values )
    for i in range(len(data)):
        data_1.loc[i,'Article']= preprocess_text(article[i])

    #converting into np arrray and hot one encoding
    # np.array(y_train)
    return data_1
def processing(args):
    path = "Data/ExperimentsFolds/"+args.data+'/'
    path_save = 'Data/Processed/'+args.data+'/'
    onlyfiles = [f for f in listdir ( path ) if isfile ( join ( path, f ) )]

    for f in onlyfiles :
        df = pd.DataFrame ()
        if 'DS' in f :
            continue
        if 'unlabel' not in f :
            print ( f )
            df = pd.read_csv ( path + f, sep='\t', header='infer', usecols=["title", "content", "label"] )
            df = df.reset_index ( drop=True )
            df['Article'] = df['content'].str.cat ( df['title'], sep=" " )
            df = df.drop ( ['title', 'content'], axis=1 )
            df_1 = data_preprocessing ( df )
            f = f.split ( '.' )[0]
            df_1.replace ( to_replace=['fake', 'true'], value=[0, 1], inplace=True )
            np.save ( path_save + f + '_x.npy', df_1['Article'] )
            np.save ( path_save + f + '_y.npy', tf.one_hot ( np.array ( df_1['label'].values ), 1 ) )
        else :
            print ( f )
            df = pd.read_csv ( path + f, sep='\t', header='infer', usecols=["title", "content", "label"] )
            df = df.reset_index ( drop=True )
            df['content'] = df['title'].str.cat ( df['content'], sep=" " )
            df = df.drop ( ['title'], axis=1 )
            df_1 = data_preprocessing ( df )
            f = f.split ( '.' )[0]
            np.save ( path_save + f + '_x.npy', df_1['content'] )
    return
if __name__ == '__main__':
    parser= argparse.ArgumentParser()
    parser.add_argument ( '--data', default='fakehealth', type=str )
    args = parser.parse_args()
    processing(args)






