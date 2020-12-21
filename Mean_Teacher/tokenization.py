from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing import sequence
import numpy as np
import tensorflow as tf

# def complete_article(path):
#     full_article_t=[]
#     for i in range(5):
#         # print('-------------FOLD Number:---------------',i)
#         x_train= np.load(path+'train_'+str(i)+'_x.npy', allow_pickle=True)
#         x_val= np.load(path+'dev_'+str(i)+'_x.npy', allow_pickle=True)
#         x_test= np.load(path+'test_x.npy', allow_pickle=True)
#         full_article_t= np.hstack((x_train, x_val, full_article_t, x_test))

#     return full_article_t

def tokenization(full_article_temp,x_train, x_val,  x_test, x_unlabel, max_len ):
    tokenizer = Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ',
                          char_level=False, oov_token=None, document_count=0)
    # full_article = np.hstack((x_train, x_test, x_unlabel))
    full_article= np.hstack(( x_unlabel, full_article_temp))
    tokenizer.fit_on_texts(full_article)
    x_train_token = tokenizer.texts_to_sequences(x_train)
    x_test_token = tokenizer.texts_to_sequences(x_test)
    x_val_token = tokenizer.texts_to_sequences(x_val)
    x_unlabel_token = tokenizer.texts_to_sequences(x_unlabel)

    x_train_seq = sequence.pad_sequences(x_train_token, maxlen=max_len,padding='post')
    x_test_seq = sequence.pad_sequences(x_test_token, maxlen=max_len,padding='post')
    x_val_seq = sequence.pad_sequences(x_val_token, maxlen=max_len,padding='post')
    x_unlabel_tar= sequence.pad_sequences(x_unlabel_token, maxlen=max_len,padding='post')
    # defining vocalbury size
    vocab_size = len(tokenizer.word_index) + 1


    x_train = x_train_seq
    x_test = x_test_seq
    return x_train,x_val_seq, x_test, x_unlabel_tar, vocab_size, tokenizer

# this function for pi model
def create_embedding(x,vocab_size):
    x= tf.keras.layers.Embedding( vocab_size, 128, input_length=None )(x)
    x = np.array(x )
    x = x.reshape(list(x.shape)+ [1] )
    tf.convert_to_tensor(x )
    return x


# this function is made for previous model we were refering
# def tokenization_emb(full_article_temp,x_train, x_val,  x_test, x_unlabel, max_len ):
#     tokenizer = Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n', lower=True, split=' ',
#                           char_level=False, oov_token=None, document_count=0)
#     # full_article = np.hstack((x_train, x_test, x_unlabel))
#     full_article= np.hstack(( x_unlabel, full_article_temp))
#     tokenizer.fit_on_texts(full_article)
#     x_train_token = tokenizer.texts_to_sequences(x_train)
#     x_test_token = tokenizer.texts_to_sequences(x_test)
#     x_val_token = tokenizer.texts_to_sequences(x_val)
#     x_unlabel_token = tokenizer.texts_to_sequences(x_unlabel)
#
#     x_train_seq = sequence.pad_sequences(x_train_token, max_len=max_len,padding='post')
#     x_test_seq = sequence.pad_sequences(x_test_token, max_len=max_len,padding='post')
#     x_val_seq = sequence.pad_sequences(x_val_token, max_len=max_len,padding='post')
#     x_unlabel_tar= sequence.pad_sequences(x_unlabel_token, max_len=max_len,padding='post')
#     # defining vocalbury size
#     vocab_size = len(tokenizer.word_index) + 1
#
#     # x_train_seq= tf.keras.utils.normalize(x_train_seq, axis=-1, order=2)
#     # x_test_seq = tf.keras.utils.normalize(x_test_seq, axis=-1, order=2 )
#     # x_val_seq = tf.keras.utils.normalize(x_val_seq, axis=-1, order=2 )
#     # x_unlabel_tar = tf.keras.utils.normalize(x_unlabel_tar, axis=-1, order=2 )
#
#     x_train = create_embedding(x_train_seq, vocab_size )
#     x_val_seq = create_embedding(x_val_seq, vocab_size )
#     x_test = create_embedding(x_test_seq, vocab_size )
#     x_unlabel_tar = create_embedding(x_unlabel_tar, vocab_size )
#
#     return x_train,x_val_seq, x_test, x_unlabel_tar, vocab_size, tokenizer
