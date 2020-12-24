import numpy as np

def unison_shuffled(x1,x2,x3,y ):
    assert len(x1)==len(y)== len(x2)
    p = np.random.permutation(len(x1))
    return [x1[p],x2[p],x3[p]], y[p]

def instant_noise_bert(x_train,y_train,x_unlabel, noise_ratio):
    noise= int(noise_ratio*len(x_train[0]))
    y_train_n = np.full((noise,2), -1)
    x0= np.append(x_train[0],x_unlabel[0][:noise], axis=0)
    x1= np.append(x_train[1],x_unlabel[1][:noise],axis=0)
    x2= np.append(x_train[2],x_unlabel[2][:noise],axis=0)
    y = np.append(y_train, y_train_n,axis=0)
    # now unison permutation
    return unison_shuffled(x0,x1,x2,y)

def instant_noise(x_train, y_train, x_unlabel, n_ratio) :
    '''this function introduce noise in the training data for mean teacher model ,
    this function is used in calculating classification cost, user have to provide
    amount of noise, want to add(ratio) in train data and test train split ratio too'''
    # amount of noise need to add in x_train data
    noise = int ( np.shape ( x_train )[0] * n_ratio )

    # taking column of x_train, need it later
    x_column = np.shape ( x_train )[1]

    if noise <= int ( np.shape ( x_unlabel )[0] ) :

        # taking number of noise from unlabel data
        ratio_noise = x_unlabel[:noise]

        # creating -1 label for noise data
        # y_unlabel = np.full ( (np.shape ( ratio_noise )[0], 1), -1 )
        y_unlabel = np.full((np.shape(ratio_noise)[0], 2), -1)
        

        # adding noise in train data
        x = np.append(x_train, ratio_noise, axis=0 )
        # print(np.shape(x))
        y = np.append(y_train, y_unlabel, axis=0 )
        x = np.append(x, y, axis=1 )
        x = x[noise :]
        row = np.shape(x)[0]

        # shufflin data
        x = np.random.permutation( x )
        # print(np.shape(x))

        # seperating label from x
        # y_train_n = np.reshape ( x[:, x_column], (row, 1) )
        y_train_n = np.reshape(x[:,[x_column,x_column+1]], (row, 2) )
        x_train_n = x[0 :len ( x ), 0 :x_column]
        # y_train_n= np.reshape(y[:len(x),0],(train_split,1))
        # print(np.shape(y_train_n))


    else :
        print ( 'error: Insufficient unlabel data available !' )

    return x_train_n, y_train_n
