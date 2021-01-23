from sklearn.model_selection import train_test_split
from BERT.bert import *
from transformers import AutoTokenizer
import pandas as pd
import numpy as np
from reader import mask_labels


def data_load(args, fold, path) :
    # will change after some time
    # path='Data/ExperimentsFolds/fakehealth/'
    train_data = pd.read_csv ( path + 'train.tsv', sep='\t' )  # ,nrows=40 )
    train_data = mask_labels ( train_data, args.ratio_label )
    train_data = train_data.sample ( frac=1, random_state=args.seed ).reset_index ( drop=True )
    # tried experimenting equalt labels , TODO: Will be removed in future
    # but true news parameter are decreasing
    # true=train_data[train_data.label=='true']
    # fake= train_data[train_data.label=='fake']
    # train_data= true[:417].append(fake)
    # train_data = train_data.sample(frac=1).reset_index(drop=True)

    # print(train_data.groupby('label').count())

    test_data = pd.read_csv ( path + 'test.tsv', sep='\t' )  # ,nrows=30)
    noise = pd.read_csv ( path + 'noise.tsv', sep='\t' )  # ,nrows=40 )

    tokenizer = AutoTokenizer.from_pretrained ( args.pretrained_model )

    # creating news example, taking vocab size also, incase
    train_data, vocab_size = create_news_examples ( train_data, args.max_len, tokenizer )
    x_train, y_train = create_inputs_targets ( train_data )

    test_data, _ = create_news_examples ( test_data, args.max_len, tokenizer )
    x_test, y_test = create_inputs_targets ( test_data )

    x_noise, _ = create_news_examples ( noise, args.max_len, tokenizer )
    x_noise, _ = create_inputs_targets ( x_noise )

    print ( 'train size:', np.shape ( x_train[0] ) )

    print ( 'test size:', np.shape ( x_test[0] ) )

    return x_train, y_train, x_test, y_test, x_noise


def data_slices(args, x_train, y_train, x_noise) :
    # this PI model will be removed in the future
    if args.model == 'PI' :
        train_dataset = tf.data.Dataset.from_tensor_slices ( (x_train[0], y_train) )
        train_dataset = train_dataset.shuffle ( buffer_size=1024 ).batch ( args.batch_size )

    else :
        # TODO:here need to include the ratio of amount of noise per batch
        train_dataset = tf.data.Dataset.from_tensor_slices ( (x_train[0], x_train[1], y_train) ).batch (
            args.batch_size )

        # QUESTION: Here I couldn't understand the logic noise data will always same size with labeled batch
        # Also you are not handling noiseed dataset where we will mask the train set, I have done all those stuff see on my code for the reference
        noise_batch = int ( args.batch_size * args.noise_ratio )
        print ( noise_batch )
        noise_dataset = tf.data.Dataset.from_tensor_slices ( (x_noise[0], x_noise[1]) ).batch ( noise_batch )

        # in case we have less noise data
        noise_dataset = noise_dataset.repeat ( args.batch_size )

    return train_dataset, noise_dataset






