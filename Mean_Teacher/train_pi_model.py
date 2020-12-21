import tensorflow as tf
import numpy as np
#this is to enable eager execution
tf.compat.v1.enable_eager_execution()
from report_writing import report_writing
from model_arch import BiLstmModel_attention, BiLstmModel
from data_loader import data_slices

from evaluation import prec_rec_f1score
from pi_costfunction import pi_model_loss,ramp_down_function,ramp_up_function
from clf.bert import  BERT

def train_Pimodel(args, epochs, batch_size,  lr,  x_train, y_train, x_val, y_val, x_test, y_test,
                      x_unlabel_tar,  max_len) :
    NUM_TRAIN_SAMPLES = np.shape ( x_train )[0]
    NUM_TEST_SAMPLES = np.shape ( x_test )[0]

    # Editable variables
    num_labeled_samples = int ( NUM_TRAIN_SAMPLES * 0.8 )
    num_validation_samples = np.shape ( x_val )[0]
    max_learning_rate = 0.003
    initial_beta1 = 0.9
    final_beta1 = 0.5
    x_unlabel_tar = x_unlabel_tar[:NUM_TRAIN_SAMPLES]
    
    train_dataset,tar_dataset= data_slices(args, x_train,y_train,x_unlabel_tar)
    # preparing the training dataset
    if args.method=='Attn':
        student = BiLstmModel_attention ( max_len, vocab_size )
    elif args.method=='Bert':
        model = BERT(args)
        student = model.create_model()
        student.summary ()
    else:
        print('Either correct model name ')

    train_metrics = tf.keras.metrics.Accuracy ()

    learning_rate = tf.Variable ( max_learning_rate )  # max learning rate
    beta_1 = tf.Variable ( initial_beta1 )
    optimizer = tf.keras.optimizers.Adam ( learning_rate=learning_rate, beta_1=beta_1, beta_2=0.999 )
    # optimizer=tf.keras.optimizers.Adam(learning_rate=lr)

    print ( num_labeled_samples, NUM_TRAIN_SAMPLES )
    max_unsupervised_weight = 100 * num_labeled_samples * (NUM_TRAIN_SAMPLES)

    for epoch in range( 1, args.epochs + 1 ) :
        print ( *"*****************" )
        print ( 'Start of epoch %d' % (epoch,) )
        print ( *"*****************" )
        rampdown_value = ramp_down_function ( epoch, args.epochs )
        rampup_value = ramp_up_function ( epoch )
        if epoch == 0 :
            unsupervised_weight = 0
        else :
            unsupervised_weight = max_unsupervised_weight * rampup_value

        learning_rate.assign ( rampup_value * rampdown_value * max_learning_rate )

        beta_1.assign ( rampdown_value * initial_beta1 + (1.0 - rampdown_value) * final_beta1 )
        # iteration over batches
        iterator_unlabel = iter(tar_dataset)
        if args.method=='Attn':
            for step, (x_batch_train, y_batch_train) in enumerate ( train_dataset ):
                with tf.GradientTape () as tape :
                    x_batch_unlabel = iterator_unlabel.get_next ()
                    loss_value = pi_model_loss( x_batch_train, y_batch_train, x_batch_unlabel, student,unsupervised_weight )
                grads = tape.gradient( loss_value, student.variables )
                optimizer.apply_gradients ( zip ( grads, student.variables ))
                # Run the forward pass of the layer
                logits = student (x_batch_train, training=True )
                train_acc = train_metrics ( tf.argmax ( y_batch_train, 1 ), tf.argmax ( logits, 1 ) )
                loss = tf.compat.v1.losses.softmax_cross_entropy ( y_batch_train, logits )
                print ( 'epoch: {}, Train Accuracy :{}, Loss: {}'.format ( epoch, train_acc.numpy (), loss.numpy () ) )

        elif args.method=='Bert':
            for step, (inputs, attention, token_id, y_batch_train) in enumerate ( train_dataset ) :
                with tf.GradientTape () as tape :
                    inp, att, to_id = iterator_unlabel.get_next()
                    loss_value = pi_model_loss ( [inputs, attention, token_id], y_batch_train, [inp, att, to_id],
                                                 student, unsupervised_weight )
                grads = tape.gradient( loss_value, student.variables )
                optimizer.apply_gradients((grad, var) for (grad, var) in zip ( grads, student.variables ) if grad is not None )
                logits = student ( [inputs, attention, token_id], training=True )
                train_acc = train_metrics ( tf.argmax ( y_batch_train, 1 ), tf.argmax ( logits, 1 ) )
                loss = tf.compat.v1.losses.softmax_cross_entropy ( y_batch_train, logits )
                print ( 'epoch: {}, Train Accuracy :{}, Loss: {}'.format ( epoch, train_acc.numpy (), loss.numpy () ) )


        # calculating accuracy
        train_acc = train_metrics ( tf.argmax ( y_batch_train, 1 ), tf.argmax ( logits, 1 ) )
        
        # Run a validation loop at the end of each epoch.
        print ( '*******Pi_Model*************' )
        prec_rec_f1score ( args,y_val, x_val, student )

        if epoch % 10 == 0 :
            print ( '---------------------------Pi Model TEST--------------------------' )
            test_accuracy, precision_true, precision_fake, recall_true, recall_fake, f1score_true, f1score_fake, AUC = prec_rec_f1score (args,
                y_test, x_test, student )
            #  report_writing(args,'Teacher', lr, batch_size, epoch, alpha, ratio, train_acc.numpy(),test_accuracy,
            #  precision_true, precision_fake, recall_true, recall_fake,f1score_true, f1score_fake, AUC,
            #  'BiLSTM-PI-')
            print ( '-----------------------------------------------------------------' )
    tf.keras.backend.clear_session ()
    return student

