import tensorflow as tf
import tensorflow.keras as tfk
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Lambda
from tensorflow.keras.models import Model
import numpy as np
<<<<<<< HEAD
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
#this is to enable eager execution
tf.compat.v1.enable_eager_execution()
=======
from keras.utils import to_categorical
from tokenization import tokenization_emb,complete_article
from evaluation import prec_rec_f1score,acc_train
from report_writing import report_writing
import csv

# os.environ["CUDA_VISIBLE_DEVICES"]="3"
>>>>>>> a111a2f77d44c52c0dadf2e832e4c9dfc2263b7c

#calling functions

from costfunction import Classification_costs,Overall_Cost,Consistency_Cost,EMA
from report_writing import report_writing
from model_arch import BiLstmModel_attention, BiLstmModel
from noise_creator import instant_noise
from evaluation import prec_rec_f1score
from pi_costfunction import pi_model_loss,ramp_down_function,ramp_up_function

<<<<<<< HEAD
def train_Pimodel(args, epochs, batch_size, alpha, lr, ratio, x_train, y_train, x_val, y_val, x_test, y_test,
                      x_unlabel_tar, vocab_size, maxlen) :
    NUM_TRAIN_SAMPLES = np.shape ( x_train )[0]
    NUM_TEST_SAMPLES = np.shape ( x_test )[0]
=======
def training_pi(args, epochs, x_train,y_train, x_val, y_val, x_test, y_test, x_unlabel,lr,batch_size):
    
    # Constants variables
    NUM_TRAIN_SAMPLES = 500
    NUM_TEST_SAMPLES = 100
>>>>>>> a111a2f77d44c52c0dadf2e832e4c9dfc2263b7c

    # Editable variables
    num_labeled_samples = int ( NUM_TRAIN_SAMPLES * 0.8 )
    num_validation_samples = np.shape ( x_val )[0]
    batch_size = 25
    max_learning_rate = 0.003
    initial_beta1 = 0.9
    final_beta1 = 0.5
    x_unlabel_tar = x_unlabel_tar[:NUM_TRAIN_SAMPLES]

    # preparing the training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices ( (x_train, y_train) )
    train_dataset = train_dataset.shuffle ( buffer_size=1024 ).batch ( batch_size )

    # preparing the target dataset
<<<<<<< HEAD
    tar_dataset = tf.data.Dataset.from_tensor_slices ( x_unlabel_tar )
    tar_dataset = tar_dataset.shuffle ( buffer_size=1024 ).batch ( batch_size )

    # declaring optimiser
    # trying changing learning rate , sometimes it gives good result
    train_metrics = tf.keras.metrics.Accuracy ()

    learning_rate = tf.Variable ( max_learning_rate )  # max learning rate
    beta_1 = tf.Variable ( initial_beta1 )
    optimizer = tf.keras.optimizers.Adam ( learning_rate=learning_rate, beta_1=beta_1, beta_2=0.999 )
    # optimizer=tf.keras.optimizers.Adam(learning_rate=lr)

    # Creating model
    student = BiLstmModel ( maxlen, vocab_size )
    # student.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='binary_crossentropy',
    #                     metrics=['accuracy'])

    # false positive rate and true positive rate
    print ( num_labeled_samples, NUM_TRAIN_SAMPLES )

    max_unsupervised_weight = 100 * num_labeled_samples * (NUM_TRAIN_SAMPLES)

    # x_unlabel_tar= tf.convert_to_tensor(x_unlabel_tar)
    for epoch in range ( 1, epochs + 1 ) :
        print ( *"*****************" )
        print ( 'Start of epoch %d' % (epoch,) )
        print ( *"*****************" )
        rampdown_value = ramp_down_function ( epoch, epochs )
        rampup_value = ramp_up_function ( epoch )
        if epoch == 0 :
=======
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
>>>>>>> a111a2f77d44c52c0dadf2e832e4c9dfc2263b7c
            unsupervised_weight = 0
        else :
            unsupervised_weight = max_unsupervised_weight * rampup_value

<<<<<<< HEAD
        learning_rate.assign ( rampup_value * rampdown_value * max_learning_rate )
        # print('learning rate: {}'.format(learning_rate))
        beta_1.assign ( rampdown_value * initial_beta1 + (1.0 - rampdown_value) * final_beta1 )
        # iteration over batches
        iterator_unlabel = iter ( tar_dataset )
        for step, (x_batch_train, y_batch_train) in enumerate ( train_dataset ) :
            with tf.GradientTape () as tape :
                x_batch_unlabel = iterator_unlabel.get_next ()
                loss_value = pi_model_loss( x_batch_train, y_batch_train, x_batch_unlabel, student,
                                             unsupervised_weight )
            grads = tape.gradient ( loss_value, student.variables )

            optimizer.apply_gradients ( zip ( grads, student.variables ) )

            # Run the forward pass of the layer
            logits = student ( x_batch_train, training=True )

            # calculating accuracy
        train_acc = train_metrics ( tf.argmax ( y_batch_train, 1 ), tf.argmax ( logits, 1 ) )
        loss = tf.compat.v1.losses.softmax_cross_entropy ( y_batch_train, logits )

        print ( 'epoch: {}, Train Accuracy :{}, Loss: {}'.format ( epoch, train_acc.numpy (), loss.numpy () ) )

        # Run a validation loop at the end of each epoch.
        print ( '*******Pi_Model*************' )
        prec_rec_f1score ( y_val, x_val, student )

        if epoch % 10 == 0 :
            print ( '---------------------------Pi Model TEST--------------------------' )
            test_accuracy, precision_true, precision_fake, recall_true, recall_fake, f1score_true, f1score_fake, AUC = prec_rec_f1score (
                y_test, x_test, student )
            #  report_writing(args,'Teacher', lr, batch_size, epoch, alpha, ratio, train_acc.numpy(),test_accuracy,
            #  precision_true, precision_fake, recall_true, recall_fake,f1score_true, f1score_fake, AUC,
            #  'BiLSTM-PI-')
            print ( '-----------------------------------------------------------------' )
    tf.keras.backend.clear_session ()
    return student
=======
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


>>>>>>> a111a2f77d44c52c0dadf2e832e4c9dfc2263b7c

