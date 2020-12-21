import argparse
from MeanTeacher import MeanTeacher
from Supervised import train_supervised
from train_pi_model import train_Pimodel
from data_loader import *
from pathlib import Path


if __name__ == '__main__':

    # parameters from arugument parser 
    parser = argparse.ArgumentParser()

    # k fold function calling 
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--max_len', default=600, type=int)
    # Model 0 Supervised 1 Mean teacher 2 Pi Model
    parser.add_argument('--model', default='MT', type=str, choices=['MT','PI'])
    # attention mechanism , BERT
    parser.add_argument('--method', default='Attn', type=str,choices=['Attn', 'Bert'])
    # parser.add_argument('--unlabel', default='All', type=str)
    #for mean teacher
    parser.add_argument('--ratio', default=0.5, type=float)
    parser.add_argument('--alpha',  default=0.99,type=float)
    parser.add_argument('--noise_ratio', type=float, default=0.2)
    parser.add_argument('--pretrained_model',default= 'bert-base-uncased', type=str, choices=['bert-base-uncased', 'bert-base-cased'])
    parser.add_argument('--data', type=str, choices=['fakehealth', 'gossipcop','politifact'])
    parser.add_argument ('--dropout',default=0.2, type=float )
    parser.add_argument('--data_folder', type=str)
    parser.add_argument('--model_output_folder', type=str)

    args = parser.parse_args()
    data_folder = Path(args.data_folder)
    path = data_folder / args.data

    print(args)

    for i in range(1):
        if args.method =='Bert':
            x_train, y_train, x_val, y_val, x_test,y_test, x_unlabel = data_load_bert(args, path)
            vocab_size=0

        elif args.method=='Attn' :
            x_train, y_train,x_val,y_val, x_test,y_test, x_unlabel, vocab_size = data_load(args,path)

        else:
            print('No correct model or method selected')

        if (args.model=='MT'):
            MeanTeacher(args, x_train, y_train,x_val, y_val, x_test, y_test,x_unlabel, vocab_size, args.max_len)
        elif (args.model=='PI') :
            train_Pimodel(args, x_train, y_train,x_val, y_val, x_test, y_test,x_unlabel,vocab_size, args.max_len)
        else :
            print("No Mean teacher for given argument")

        # resetting the environment
        tf.keras.backend.clear_session()
    print('finished')
