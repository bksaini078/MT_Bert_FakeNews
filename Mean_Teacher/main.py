import argparse
from tokenization import tokenization,complete_article
from Bert_Tokenisation import bert_tokenization
from MeanTeacher import MeanTeacher
from Supervised import train_supervised
import numpy as np
import tensorflow as tf


if __name__ == '__main__':

    # parameters from arugument parser 
    parser = argparse.ArgumentParser()

    # k fold function calling 
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--maxlen', default=200, type=int)
    parser.add_argument('--meanteacher', default=0, type=bool)
    parser.add_argument('--method', default='Attn', type=str)
    parser.add_argument('--unlabel', default='Mix', type=str)
    parser.add_argument('--data', default='fakehealth', type=str)
    #for mean teacher 
    parser.add_argument('--ratio', default=0.5, type=float)
    parser.add_argument('--alpha',  default=0.99,type=float)
    parser.add_argument('--noise_ratio', type=float, default=0.2)
    args = parser.parse_args()
    path= 'Data/Processed/'+args.data+'/'

    for i in range(5):
        x_train = np.load(path + 'train_' + str(i) + '_x.npy', allow_pickle=True)
        y_train = np.load(path + 'train_' + str(i) + '_y.npy', allow_pickle=True)

        x_val = np.load(path + 'dev_' + str(i) + '_x.npy', allow_pickle=True)
        y_val = np.load(path + 'dev_' + str(i) + '_y.npy', allow_pickle=True)

        x_test = np.load(path + 'test_x.npy', allow_pickle=True)
        y_test = np.load(path + 'test_y.npy', allow_pickle=True)

        if args.unlabel=='Mix' :
            x_unlabel = np.load(path + 'unlabeled_' + 'mix' + '_x.npy', allow_pickle=True)
        elif args.unlabel=='All':
            x_unlabel = np.load(path + 'unlabeled_' + 'all' + '_x.npy', allow_pickle=True)
        print('train data size:', np.shape(x_train))
        print('train data True/Fake count:', np.count_nonzero(y_train), len(y_train) - np.count_nonzero(y_train))
        print('val data size:', np.shape(x_val))

        print('test data size:', np.shape(x_test))
        print('test data True/Fake count:', np.count_nonzero(y_test), len(y_test) - np.count_nonzero(y_test))


        if args.method =='BERT':
            x_train, vocab_size, tokenizer = bert_tokenization(x_train, args.maxlen)
            x_val, _, _ = bert_tokenization(x_val, args.maxlen)
            x_test, _, _ = bert_tokenization(x_test, args.maxlen)
            x_unlabel, _, _ = bert_tokenization( x_unlabel, args.maxlen)
        elif args.method=='Attn':
            comp_article= complete_article(path)
            x_train, x_val, x_test, x_unlabel, vocab_size, tokenizer = tokenization(comp_article,x_train, x_val, x_test, x_unlabel,args.maxlen)
        # train_supervised(epochs, batch_size, lr,x_train, y_train, x_test, y_test,maxlen,vocab_size)
        # calling model according to inputs
        if (args.meanteacher == 0):
            train_supervised(args,args.epochs, args.batch_size, args.lr, x_train, y_train, x_val, y_val, x_test, y_test, args.maxlen, vocab_size)
        elif (args.meanteacher == 1):
            MeanTeacher(args, args.epochs, args.batch_size, args.alpha, args.lr, args.ratio, args.noise_ratio, x_train, y_train,
                        x_val, y_val, x_test, y_test,x_unlabel, vocab_size, args.maxlen)

        else :
            print("No Mean teacher for given argument")
        # get_ipython().magic('reset -sf')
        # resetting the environment
        tf.keras.backend.clear_session()
    print('finished')
