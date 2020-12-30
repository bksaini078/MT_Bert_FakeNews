from sklearn.model_selection import train_test_split
from BERT.bert import *
from Mean_Teacher.tokenization import tokenization
from transformers import AutoTokenizer
import pandas as pd

def data_load(args,fold, path):
    #will change after some time
    # path='Data/ExperimentsFolds/fakehealth/'
    train_data = pd.read_csv ( path + 'train.tsv', sep='\t' )
    test_data= pd.read_csv(path+'test.tsv',sep='\t')
    unlabel = pd.read_csv ( path + 'unlabel.tsv', sep='\t' )
    unlabel['label'] = np.full ( (len ( unlabel ), 1), 'fake' )  # in case label column doesnot have fake label

    # splitting val data TODO: Need to change in in future
    # train_data= train_data.sample(n=200, random_state=1)
    # test_data= test_data.sample(n=30, random_state=1)
    train_data, val_data = train_test_split ( train_data, test_size=0.2, random_state=42 )


    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    if args.method=='Attn':
        train_data,vocab_size = create_news_examples ( train_data, args.max_len, tokenizer )
        [x_train,_,_], y_train = create_inputs_targets ( train_data )
        val_data,_ = create_news_examples ( val_data, args.max_len, tokenizer )
        [x_val,_,_], y_val = create_inputs_targets ( val_data )
        test_data = create_news_examples ( test_data, args.max_len, tokenizer )
        x_test, y_test = create_inputs_targets ( test_data )
        [x_unlabel,_,_],_ = create_news_examples ( unlabel, args.max_len, tokenizer )
        x_unlabel, _ = create_inputs_targets ( x_unlabel )

    elif args.method=='Bert':
        # need seperation from bert because we need only input ids
        train_data,vocab_size = create_news_examples ( train_data, args.max_len, tokenizer)
        x_train, y_train = create_inputs_targets ( train_data )
        val_data,_ = create_news_examples ( val_data, args.max_len, tokenizer)
        x_val, y_val = create_inputs_targets ( val_data )
        test_data,_ = create_news_examples ( test_data, args.max_len, tokenizer)
        x_test, y_test = create_inputs_targets ( test_data )
        x_unlabel,_ = create_news_examples ( unlabel, args.max_len, tokenizer )
        x_unlabel, _ = create_inputs_targets ( x_unlabel )
    return x_train, y_train, x_val, y_val, x_test, y_test, x_unlabel, vocab_size

def data_slices(args, x_train,y_train):

    if args.method=='Attn':
        train_dataset = tf.data.Dataset.from_tensor_slices ( (x_train, y_train) )
        train_dataset = train_dataset.shuffle ( buffer_size=1024 ).batch ( args.batch_size )

        return train_dataset
    elif args.method=='Bert':
        train_dataset = tf.data.Dataset.from_tensor_slices ( (x_train[0], x_train[1], x_train[2], y_train) )
        train_dataset = train_dataset.shuffle ( buffer_size=1024 ).batch ( args.batch_size )

        return train_dataset


    else:
        print('Either model selection or method is incorrectly entered')
        return




