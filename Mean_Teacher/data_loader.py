from sklearn.model_selection import train_test_split
from BERT.bert import *
from Mean_Teacher.tokenization import tokenization
from transformers import AutoTokenizer
import pandas as pd

def data_load(args,fold,path):
    x_train = np.load ( path + 'train_x.npy', allow_pickle=True )
    y_train = np.load ( path + 'train_y.npy', allow_pickle=True )
    # x_train= x_train[:200]
    # y_train=y_train[:200]
    # this we need when we are not having seperate val data TODO
    x_train, x_val, y_train, y_val = train_test_split( x_train, y_train, test_size=0.2, random_state=42 )
    x_test = np.load ( path + 'test_x.npy', allow_pickle=True )
    y_test = np.load ( path + 'test_y.npy', allow_pickle=True )

    # x_test=x_test[:40]
    # y_test=y_test[:40]
    x_unlabel = np.load ( path + 'unlabel_x.npy', allow_pickle=True )
    print ( 'train data size:', np.shape ( x_train ) )
    print ( 'train data True/Fake count:', np.count_nonzero ( y_train ),
            len ( y_train ) - np.count_nonzero ( y_train ) )
    print ( 'val data size:', np.shape ( x_val ) )
    print ( 'val data True/Fake count:', np.count_nonzero ( y_val ), len ( y_val ) - np.count_nonzero ( y_val ) )
    print ( 'test data size:', np.shape ( x_test ) )
    print ( 'test data True/Fake count:', np.count_nonzero ( y_test ), len ( y_test ) - np.count_nonzero ( y_test ) )
    print ( 'unlabel data size:', np.shape ( x_test ) )
    comp_article = np.hstack ( (x_train, x_val, x_test, x_unlabel) )
    x_train, x_val, x_test, x_unlabel, vocab_size, tokenizer = tokenization \
        (comp_article, x_train, x_val, x_test, x_unlabel, args.max_len )
    y_train = to_categorical ( y_train )
    y_val = to_categorical ( y_val )
    y_test = to_categorical ( y_test )
    return x_train, y_train,x_val,y_val, x_test,y_test, x_unlabel, vocab_size


def data_load_bert(args,fold, path):
    #will change after some time 
    # path='Data/ExperimentsFolds/fakehealth/'
    train_data = pd.read_csv ( path + 'train.tsv', sep='\t' )
    test_data= pd.read_csv(path+'test.tsv',sep='\t')
    unlabel = pd.read_csv ( path + 'unlabel.tsv', sep='\t' )

    # splitting val data TODO: Need to change in in future
    # train_data= train_data.sample(n=200, random_state=1)
    # test_data= test_data.sample(n=30, random_state=1)
    train_data, val_data = train_test_split ( train_data, test_size=0.2, random_state=42 )

    tokenizer = AutoTokenizer.from_pretrained ( args.pretrained_model )

    train_data = create_news_examples ( train_data, args.max_len, tokenizer )
    x_train, y_train = create_inputs_targets ( train_data )
    val_data = create_news_examples ( val_data, args.max_len, tokenizer )
    x_val, y_val = create_inputs_targets ( val_data )

    test_data = create_news_examples ( test_data, args.max_len, tokenizer )
    x_test, y_test = create_inputs_targets ( test_data )

    unlabel['label'] = np.full ( (len ( unlabel ), 1), 'fake' )# in case label column doesnot have fake label
    x_unlabel = create_news_examples ( unlabel, args.max_len, tokenizer )
    x_unlabel, _ = create_inputs_targets ( x_unlabel )

    return x_train, y_train, x_val, y_val, x_test,y_test, x_unlabel

def data_slices(args, x_train,y_train):

    if args.method=='Attn':
        train_dataset = tf.data.Dataset.from_tensor_slices ( (x_train, y_train) )
        train_dataset = train_dataset.shuffle ( buffer_size=1024 ).batch ( args.batch_size )
        # preparing the target dataset
        # tar_dataset = tf.data.Dataset.from_tensor_slices (x_unlabel_tar )
        # tar_dataset = tar_dataset.shuffle ( buffer_size=1024 ).batch ( args.batch_size )
        return train_dataset
    elif args.method=='Bert':
        train_dataset = tf.data.Dataset.from_tensor_slices ( (x_train[0], x_train[1], x_train[2], y_train) )
        train_dataset = train_dataset.shuffle ( buffer_size=1024 ).batch ( args.batch_size )
        # tar_dataset = tf.data.Dataset.from_tensor_slices ( (x_unlabel_tar[0], x_unlabel_tar[1], x_unlabel_tar[2]) )
        # tar_dataset = tar_dataset.shuffle ( buffer_size=1024 ).batch ( args.batch_size )
        return train_dataset


    else:
        print('Either model selection or method is incorrectly entered')
        return




