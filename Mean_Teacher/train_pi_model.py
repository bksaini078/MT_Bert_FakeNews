import math
import tensorflow as tf
# import tensorflow.contrib.eager as tfe
import os
import numpy as np
from keras.utils import to_categorical
from tokenization import tokenization_emb,complete_article
from evaluation import prec_rec_f1score,acc_train
from report_writing import report_writing
import csv

# os.environ["CUDA_VISIBLE_DEVICES"]="3"

from multitask_semisupervised_model import PiModel, pi_model_loss, pi_model_gradients, ramp_up_function, ramp_down_function


def training_pi(args, epochs, x_train,y_train, x_val, y_val, x_test, y_test, x_unlabel,lr,batch_size):
    
    # Constants variables
    NUM_TRAIN_SAMPLES = 500
    NUM_TEST_SAMPLES = 100

    # Editable variables
    num_labeled_samples = len(x_train)
    num_validation_samples = len(x_val)
    # batch_size = 25
    # epochs = 200
    # lr = 0.003
    initial_beta1 = 0.9
    final_beta1 = 0.5

    # Assign it as tfe.variable since we will change it across epochs

    learning_rate = tf.Variable(lr,dtype= tf.float32)
    beta_1 = tf.Variable(initial_beta1)
     
    batches_per_epoch = int(num_labeled_samples/batch_size)
    batches_per_epoch_val = int(num_validation_samples / batch_size)
# #    sys.exit()
    train_labeled = tf.data.Dataset.from_tensor_slices ( (x_train, y_train) )
    train_labeled = train_labeled.shuffle ( buffer_size=1024 ).batch (batch_size)
    # preparing the target dataset
    train_unlabeled = tf.data.Dataset.from_tensor_slices ( x_unlabel )
    train_unlabeled = train_unlabeled.shuffle ( buffer_size=1024 ).batch ( batch_size )
    model = PiModel()
    # print(model.summary())
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=0.999)
    max_unsupervised_weight = 100 * num_labeled_samples*(NUM_TRAIN_SAMPLES - num_validation_samples)
   
    #sys.exit()
    for epoch in range(epochs):
        print('********************epoch:{}************************'.format(epoch))

        rampdown_value = ramp_down_function(epoch, epochs)
        rampup_value = ramp_up_function(epoch)

        if epoch == 0:
            unsupervised_weight = 0
        else:
            unsupervised_weight = max_unsupervised_weight * rampup_value

        learning_rate.assign(rampup_value * rampdown_value * lr)
        beta_1.assign(rampdown_value * initial_beta1 +(1.0 - rampdown_value) * final_beta1)
        iterator_label = iter(train_labeled)
        iterator_unlabel = iter(train_unlabeled )
        i=0

        for batch_nr in range(batches_per_epoch):
            print(i*'##')
            X_labeled_train, y_labeled_train = iterator_label.get_next()
            X_unlabeled_train = iterator_unlabel.get_next()
  
            loss_val, grads = pi_model_gradients(X_labeled_train, y_labeled_train, X_unlabeled_train,
                                                 model, unsupervised_weight)
            optimizer.apply_gradients(zip(grads, model.variables))
            train_acc= acc_train(args,y_labeled_train,X_labeled_train,model)
            print('Train accuracy:', train_acc)
            i=i+1

        print("**********Validation Accuracy Details***********")
        prec_rec_f1score(args,y_val,x_val,model)
        print("********************************************")
            
            
    print('############# Result With Test Data ###########' )
    test_accuracy, precision_true, precision_fake, recall_true, recall_fake, f1score_true,\
    f1score_fake, AUC=prec_rec_f1score(args,y_test,x_test,model)
    print('Writing Report')
    report_writing( args, 'Pi Temporal', args.lr, args.batch_size, epoch, 'NaN', 'NaN', train_acc, test_accuracy,
                     precision_true, precision_fake, recall_true, recall_fake, f1score_true, f1score_fake, AUC,
                    args.unlabel )



