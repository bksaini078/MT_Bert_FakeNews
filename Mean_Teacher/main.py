import argparse
from tokenization import tokenization,complete_article,tokenization_emb
from Bert_Tokenisation import bert_tokenization
from MeanTeacher import MeanTeacher
from Supervised import train_supervised
from keras.utils import to_categorical
import numpy as np
import tensorflow as tf
from train_pi_model import train_Pimodel


if __name__ == '__main__':

    # parameters from arugument parser 
    parser = argparse.ArgumentParser()

    # k fold function calling 
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--maxlen', default=600, type=int)
    # Model 0 Supervised 1 Mean teacher 2 Pi Model
    parser.add_argument('--model', default=0, type=int)
    # attention mechanism , BERT
    parser.add_argument('--method', default='Attn', type=str)
    # parser.add_argument('--unlabel', default='All', type=str)
    parser.add_argument('--data', default='fakehealth', type=str)
    #for mean teacher 
    parser.add_argument('--ratio', default=0.5, type=float)
    parser.add_argument('--alpha',  default=0.99,type=float)
    parser.add_argument('--noise_ratio', type=float, default=0.2)
    args = parser.parse_args()
    path= 'Data/Processed/'+args.data+'/'
    print(args)

    for i in range(5):
        x_train = np.load(path + 'train_' + str(i) + '_x.npy', allow_pickle=True)
        y_train = np.load(path + 'train_' + str(i) + '_y.npy', allow_pickle=True)

        x_val = np.load(path + 'dev_' + str(i) + '_x.npy', allow_pickle=True)
        y_val = np.load(path + 'dev_' + str(i) + '_y.npy', allow_pickle=True)

        x_test = np.load(path + 'test_x.npy', allow_pickle=True)
        y_test = np.load(path + 'test_y.npy', allow_pickle=True)
        x_unlabel = np.load ( path + 'unlabeled_x.npy', allow_pickle=True )

        # if args.unlabel=='Mix' :
        #     x_unlabel = np.load(path + 'unlabeled_' + 'mix' + '_x.npy', allow_pickle=True)
        # elif args.unlabel=='All':
        #     x_unlabel = np.load(path + 'unlabeled_' + 'all' + '_x.npy', allow_pickle=True)
        print('train data size:', np.shape(x_train))
        print('train data True/Fake count:', np.count_nonzero(y_train), len(y_train) - np.count_nonzero(y_train))
        print('val data size:', np.shape(x_val))
        print('val data True/Fake count:', np.count_nonzero(y_val), len(y_val) - np.count_nonzero(y_val))
        print('test data size:', np.shape(x_test))
        print('test data True/Fake count:', np.count_nonzero(y_test), len(y_test) - np.count_nonzero(y_test))
        y_train = to_categorical ( y_train )
        y_val = to_categorical ( y_val )
        y_test = to_categorical ( y_test )

        if args.method =='BERT':
            x_train, vocab_size, tokenizer = bert_tokenization(x_train, args.maxlen)
            x_val, _, _ = bert_tokenization(x_val, args.maxlen)
            x_test, _, _ = bert_tokenization(x_test, args.maxlen)
            x_unlabel, _, _ = bert_tokenization( x_unlabel, args.maxlen)
        elif args.method=='Attn' and args.model != 2:
            comp_article= complete_article(path)
            x_train, x_val, x_test, x_unlabel, vocab_size, tokenizer = tokenization\
                (comp_article,x_train, x_val, x_test, x_unlabel,args.maxlen)
        # elif args.model==2:
        #     comp_article= complete_article(path)
        #     x_train, x_val, x_test, x_unlabel, vocab_size, tokenizer = tokenization_emb\
        #         (comp_article,x_train, x_val, x_test, x_unlabel,args.maxlen)
        else:
            print('No correct model or method selected')

        # train_supervised(epochs, batch_size, lr,x_train, y_train, x_test, y_test,maxlen,vocab_size)
        # calling model according to inputs
        if (args.model == 0):
            train_supervised(args,args.epochs, args.batch_size, args.lr, x_train, y_train, x_val, y_val, x_test, y_test, args.maxlen, vocab_size)
        elif (args.model == 1):
            MeanTeacher(args, args.epochs, args.batch_size, args.alpha, args.lr, args.ratio, args.noise_ratio,
                        x_train, y_train,x_val, y_val, x_test, y_test,x_unlabel, vocab_size, args.maxlen)
        elif (args.model == 2) :
            train_Pimodel(args.epochs, x_train,y_train, x_val, y_val, x_test, y_test, x_unlabel,args.lr,args.batch_size)

        else :
            print("No Mean teacher for given argument")
        # get_ipython().magic('reset -sf')
        # resetting the environment
        tf.keras.backend.clear_session()
    print('finished')
