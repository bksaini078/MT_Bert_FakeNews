import argparse
from Mean_Teacher.train_MeanTeacher import MeanTeacher
from PI_model.train_pi_model import Pimodel
from Mean_Teacher.data_loader import *
from pathlib import Path
from BERT.train_BERT import train_bert


if __name__ == '__main__':
    # parameters from arugument parser 
    parser = argparse.ArgumentParser()
    # k fold function calling 
    parser.add_argument('--lr', default=2e-5, type=float)
    parser.add_argument('--epochs', default=1, type=int)
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--max_len', default=512, type=int)
    parser.add_argument('--model', default='MT', type=str, choices=['MT','PI','bert'])
    parser.add_argument('--ratio', default=0.5, type=float)
    parser.add_argument('--alpha',  default=0.99,type=float)
    parser.add_argument('--noise_ratio', type=float, default=0.9)
    parser.add_argument('--pretrained_model',default= 'bert-base-uncased', type=str, choices=['bert-base-uncased', 'bert-base-cased'])
    parser.add_argument('--data', default= 'fakehealth',type=str)
    parser.add_argument ('--dropout',default=0.1, type=float )
    parser.add_argument('--data_folder', type=str)
    parser.add_argument('--model_output_folder', type=str)
    parser.add_argument ( '--do_train', action='store_true' )
    parser.add_argument ( '--seed', type=int )

    args = parser.parse_args()
    data_folder = Path(args.data_folder)
    path = f'{data_folder}/{args.data}/'
    test_split_number = args.data_folder.split ( '/' )[-1]
    path_report = Path('Reports/'+test_split_number+'/'+args.data+'/')
    path_report.mkdir(parents=True, exist_ok=True)

    print(args)
    print ( path )
    num_gpu = len ( tf.config.experimental.list_physical_devices('GPU'))
    if num_gpu > 0 :
        logger.info("GPU is found")
    else :
        logger.info("Training with CPU")
    for fold in range(1):
        x_train, y_train, x_val, y_val, x_test, y_test, x_unlabel,vocab_size = data_load(args, fold, path)
        if (args.model=='MT'):
            MeanTeacher(args, fold,x_train, y_train,x_val, y_val, x_test, y_test,x_unlabel, vocab_size)
        elif (args.model=='PI') :
            Pimodel(args, fold, x_train, y_train,x_val, y_val, x_test, y_test,x_unlabel,vocab_size)
        elif args.model=='bert':
            train_bert(args)

        else :
            print("No Mean teacher for given argument")
        # resetting the environment
        tf.keras.backend.clear_session()
    print('finished')
