import tensorflow as tf 
import tensorflow.keras 
import pandas as pd 
import numpy as np
# from Mean_Teacher.report_writing import report_writing
# from Mean_Teacher.data_loader import data_slices

# from Mean_Teacher.evaluation import prec_rec_f1score
from PI_model.pi_costfunction import pi_model_loss,ramp_down_function,ramp_up_function
from PI_model.pi_model import PiModel


if __name__=='__main__':
    x_train= np.load('/content/drive/MyDrive/MT_Bert_FakeNews_V6/Data/pi_data/x_train.npy', allow_pickle=True)
    y_train= np.load('/content/drive/MyDrive/MT_Bert_FakeNews_V6/Data/pi_data/y_train.npy', allow_pickle=True)

    x_test= np.load('/content/drive/MyDrive/MT_Bert_FakeNews_V6/Data/pi_data/x_test.npy', allow_pickle=True)
    y_test= np.load('/content/drive/MyDrive/MT_Bert_FakeNews_V6/Data/pi_data/y_test.npy', allow_pickle=True)

    x_unlabel= np.load('/content/drive/MyDrive/MT_Bert_FakeNews_V6/Data/pi_data/x_test.npy', allow_pickle=True)
    print(x_train.shape)
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024 ).batch(1)

    pi_model= PiModel()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    pi_model.compile( optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'] )
    for (x,y) in enumerate(train_dataset):
        print(tf.shape(x))
        print(pi_model(x))
        
    # x1 = x_train[0].reshape(1,100, 100, 1)
    # x2 = x_train[1].reshape(1,150, 20, 10)

    # x3=np.shape(np.append(x1,x2,axis=0))
  
    


    

