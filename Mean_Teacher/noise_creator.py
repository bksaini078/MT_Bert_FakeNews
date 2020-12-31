import numpy as np

def unison_shuffled(x1,x2,x3,y, args):
    assert len(x1)==len(y)== len(x2)== len(x3)
    p = np.random.permutation(args.batch_size)
    return [x1[p],x2[p],x3[p]], y[p]

def instant_noise_bert(x_train,y_train,x_unlabel, args):
    noise= int(args.noise_ratio*len(x_train[0]))
    p = np.random.permutation(noise)
    y_train_n = np.full((noise,2), -1)
    x0= np.append(x_train[0],x_unlabel[0][p], axis=0)
    x1= np.append(x_train[1],x_unlabel[1][p],axis=0)
    x2= np.append(x_train[2],x_unlabel[2][p],axis=0)
    y = np.append(y_train, y_train_n,axis=0)
    # now unison permutation
    return unison_shuffled(x0,x1,x2,y,args)

