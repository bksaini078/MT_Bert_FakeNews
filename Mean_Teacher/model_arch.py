import tensorflow as tf
from  tensorflow.keras.layers  import *
from tensorflow import keras
from tensorflow.keras.models import Model
from keras_self_attention import SeqSelfAttention

def BiLstmModel(maxlen, vocab_size):
  tf.keras.backend.clear_session()
  inputs = keras.Input(shape=(maxlen,))
  x =Embedding(vocab_size, 128, input_length=None)(inputs)
  x =Bidirectional(LSTM(128))(x)
  x= GaussianNoise(1)(x)
  x = Dropout(0.2)(x)
#   x =Dense(64,activation='relu')(x)
  x =Dense(64)(x)
  x = Dense(32)(x)
  x =Dense(2, activation='sigmoid')(x)
  return Model(inputs,x)

def BiLstmModel_attention(maxlen, vocab_size):
  tf.keras.backend.clear_session()
  inputs = keras.Input(shape=(maxlen,))
  x = Embedding(vocab_size, 128, input_length=None)(inputs)
  x = Bidirectional(LSTM(128,return_sequences= True))(x)
  x = tf.keras.layers.Attention(use_scale=True)([x,x])
#   x=SeqSelfAttention(attention_activation='sigmoid')(x)
  x= Flatten()(x)
  x = Dense(64)(x)
  x = Dense(32)(x)
  x = Dense(2, activation='sigmoid')(x)
  return Model(inputs,x)
