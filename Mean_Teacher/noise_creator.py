import numpy as np

def unison_shuffled(x1,x2,y, args):
    assert len(x1)==len(y)== len(x2)
    p = np.random.permutation(len(x1))
    # print(p)
    
    return [x1[p],x2[p]], y[p]

def instant_noise_bert(x_train,y_train,x_unlabel, args):
    # noise= int(args.noise_ratio*len(x_train[0]))
    
   
    # print(p)
    # y_train_n = np.full((noise,2), -1)
    
    y_train_n = np.full((len(x_unlabel[0]),2), -1)
   
    x0= np.append(x_train[0],x_unlabel[0],axis=0)
    x1= np.append(x_train[1],x_unlabel[1],axis=0)
   
    y = np.append(y_train, y_train_n,axis=0)
    # p = np.random.permutation(len(x0))[:len(x0)-2]
    
    # now unison permutation
    return [x0,x1],y #unison_shuffled(x0,x1,y,args)

