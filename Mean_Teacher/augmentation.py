import numpy as np
import tensorflow as tf


def unison_shuffled(x1, x2, y, args) :
    '''Shuffling data , but i guess it is not required and will be remove in future '''
    assert len ( x1 ) == len ( y ) == len ( x2 )
    p = np.random.permutation ( len ( x1 ) )
    # print(p)

    return [x1[p], x2[p]], y[p]


def augment_data(x_train, y_train, x_unlabel, args) :
    '''Adding unlabel data to train data'''
    #@TODO i couldn't get this function
    # need to include noise parameter to control to maintain unlabel ratio
    # indices = tf.range(start=0, limit=len(x_unlabel[0]), dtype=tf.int32)
    # shuffled_indices = tf.random.shuffle(indices)

    # shuffled_inp = tf.gather(x_unlabel[0], shuffled_indices)
    # shuffled_attn = tf.gather(x_unlabel[1], shuffled_indices)
    noised_attention= attention_noise(x_unlabel[1],args)

    y_train_n = np.full ( (len ( x_unlabel[0] ), 2), -1 )
    input_id = np.append ( x_train[0], x_unlabel[0], axis=0 )
    atten_id = np.append ( x_train[1], noised_attention, axis=0 )
    y = np.append ( y_train, y_train_n, axis=0 )
    # print(atten_id)

    # introduction attention noise :NOT having good performance
    # atten_noise = attention_noise ( atten_id, args )

    # now unison permutation
    return [input_id, atten_id], y  # unison_shuffled(x0,x1,y,args)


def attention_noise(attention, args) :
    noised_attention= np.zeros(attention.shape)
    if args.attn_noise==0:
        return attention
    for i in range(len(attention)):
        if np.random.binomial(1,0.3):# tossing to select particular article will be noised or not
            condition = np.random.choice(a=[False, True], size=attention[i].shape, p=[1 - args.attn_noise, args.attn_noise] )
            case_true = 1 - attention[i]  # if 1 then 0 if 0 then 1
            case_false = attention[i]
            noised_attention[i] = tf.where(condition, case_true, case_false)
        else:
            noised_attention[i]=attention[i]
    noised_attention=tf.convert_to_tensor(noised_attention)
    # print(noised_attention)
    return noised_attention
