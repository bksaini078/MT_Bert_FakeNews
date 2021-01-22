from sklearn.model_selection import train_test_split
from BERT.bert import *
from transformers import AutoTokenizer
import pandas as pd


def data_load(args,fold, path):
    #will change after some time
    # path='Data/ExperimentsFolds/fakehealth/'
    train_data = pd.read_csv ( path + 'train.tsv', sep='\t')#,nrows=100 )
    # takinq equal labels
    val_data= train_data[:30]
    # true=train_data[train_data.label=='true']
    # fake= train_data[train_data.label=='fake']
    # train_data= true[:417].append(fake)
    # train_data = train_data.sample(frac=1).reset_index(drop=True)
    print(train_data.groupby('label').count()) 

    test_data= pd.read_csv(path+'test.tsv',sep='\t')#,nrows=40)
    unlabel = pd.read_csv(path + 'noise.tsv', sep='\t')#,nrows=100 )
    # in case label column doesnot have fake label, because preprocessing need label column.
    # but this column plays no role during training
    # unlabel['label'] = np.full ( (len ( unlabel ), 2), -1 )
    # splitting val data TODO: Need to change in in future
    # train_data, val_data = train_test_split ( train_data, test_size=0.0, random_state=42 )
    
    tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
    # need seperation from bert because we need only input ids
    train_data,vocab_size = create_news_examples ( train_data, args.max_len, tokenizer)
    x_train, y_train = create_inputs_targets ( train_data )
    val_data,_ = create_news_examples(val_data, args.max_len, tokenizer)
    x_val, y_val = create_inputs_targets(val_data )
    test_data,_ = create_news_examples(test_data, args.max_len, tokenizer)
    x_test, y_test = create_inputs_targets(test_data )
    x_unlabel,_ = create_news_examples (unlabel, args.max_len, tokenizer )
    x_unlabel, _ = create_inputs_targets ( x_unlabel )
    print('train size:',np.shape(x_train[0]))
    print('val size:',np.shape(x_val[0]))
    print('test size:', np.shape(x_test[0]))
   
    return x_train, y_train, x_val, y_val, x_test, y_test, x_unlabel, vocab_size

def data_slices(args, x_train,y_train,x_unlabel):
    if args.model=='PI':
        train_dataset = tf.data.Dataset.from_tensor_slices ( (x_train[0], y_train) )
        train_dataset = train_dataset.shuffle( buffer_size=1024 ).batch( args.batch_size)

    else:
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train[0], x_train[1], y_train) ).batch(args.batch_size)
        unlabel_dataset= tf.data.Dataset.from_tensor_slices((x_unlabel[0],x_unlabel[1])).batch(args.batch_size)
        unlabel_dataset=unlabel_dataset.repeat(3)
        # train_dataset = train_dataset.shuffle ( buffer_size=1024 ).batch ( args.batch_size )

    return train_dataset, unlabel_dataset






