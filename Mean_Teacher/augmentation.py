import numpy as np
import tensorflow as tf

def unison_shuffled(x1,x2,y, args):
    '''Shuffling data , but i guess it is not required and will be remove in future '''
    assert len(x1)==len(y)== len(x2)
    p = np.random.permutation(len(x1))
    # print(p)
    
    return [x1[p],x2[p]], y[p]

def augment_data(x_train, y_train, x_unlabel, args):
    '''Adding unlabel data to train data'''
    # need to include noise parameter to control to maintain unlabel ratio
    indices = tf.range(start=0, limit=len(x_unlabel[0]), dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)

    shuffled_inp = tf.gather(x_unlabel[0], shuffled_indices)
    shuffled_attn = tf.gather(x_unlabel[1], shuffled_indices)


    y_train_n = np.full((len(x_unlabel[0]), 2), -1)
    input_id = np.append(x_train[0], shuffled_inp, axis=0)
    atten_id = np.append(x_train[1], shuffled_attn, axis=0)
    y = np.append(y_train, y_train_n, axis=0)



    # not performing random permutation, effecting the result
    # p = np.random.permutation(len(x0))[:len(x0)-2]

    # now unison permutation
    return [input_id, atten_id], y  # unison_shuffled(x0,x1,y,args)

