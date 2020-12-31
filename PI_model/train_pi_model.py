import tensorflow as tf
import numpy as np

# this is to enable eager execution
tf.compat.v1.enable_eager_execution ()
from Mean_Teacher.report_writing import report_writing
from Mean_Teacher.data_loader import data_slices
from Mean_Teacher.evaluation import prec_rec_f1score
from PI_model.pi_costfunction import pi_model_loss, ramp_down_function, ramp_up_function
from BERT.bert import BERT


def Pimodel(args, fold, x_train, y_train, x_val, y_val, x_test, y_test, x_unlabel_tar, vocab_size) :
    x_unlabel_tar = x_unlabel_tar[:len(x_train[0])]
    NUM_TRAIN_SAMPLES = len(x_train[0]) + len (x_unlabel_tar[0])
    NUM_TEST_SAMPLES =len(x_test[0])

    # Editable variables
    num_labeled_samples = int (len(x_train[0]))
    num_validation_samples =  len(x_val[0])
    max_learning_rate = args.lr
    initial_beta1 = 0.9
    final_beta1 = 0.5

    learning_rate = tf.Variable(max_learning_rate)  # max learning rate
    beta_1 = tf.Variable ( initial_beta1 )
    # optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=0.999 )
    optimizer = tf.keras.optimizers.Adam( learning_rate=args.lr)
    train_dataset = data_slices ( args, x_train, y_train )
    pi_model = BERT(args).create_model()
    pi_model.summary ()
    train_metrics = tf.keras.metrics.Accuracy ()

    # max_unsupervised_weight = 100 * num_labeled_samples * (NUM_TRAIN_SAMPLES)
    unsupervised_weight = 0.2
    progbar = tf.keras.utils.Progbar ( len ( train_dataset ), stateful_metrics=['Accuracy', 'Overall_Loss'] )
    for epoch in range ( 1, args.epochs + 1 ) :

        tf.print ( 'epoch: %d' % (epoch,) )

        # rampdown_value = ramp_down_function(epoch, args.epochs)
        # rampup_value = ramp_up_function(epoch)
        # if epoch == 0 :
        #     unsupervised_weight = 0
        # else :
        #     unsupervised_weight = max_unsupervised_weight * rampup_value
        # learning_rate.assign(rampup_value * rampdown_value * max_learning_rate )
        # print(f'Learning rate: {learning_rate}')
        # beta_1.assign(rampdown_value * initial_beta1 + (1.0 - rampdown_value) * final_beta1 )
        for step, (inputs, attention, token_id, y_batch_train) in enumerate ( train_dataset ) :
            with tf.GradientTape () as tape :
                loss_value = pi_model_loss ( args, [inputs, attention, token_id], y_batch_train, x_unlabel_tar,
                                             pi_model, unsupervised_weight )
            grads = tape.gradient ( loss_value, pi_model.variables )
            optimizer.apply_gradients (
                (grad, var) for (grad, var) in zip ( grads, pi_model.variables ) if grad is not None )
            logits = pi_model ( [inputs, attention, token_id], training=True )
            train_acc = train_metrics ( tf.argmax ( y_batch_train, 1 ), tf.argmax ( logits, 1 ) )
            progbar.add ( args.batch_size, values=[('Accuracy', train_acc), ('Overall_Loss', loss_value)] )

        # Run a validation loop at the end of each epoch.
        y_vp = pi_model ( x_val )
        val_acc = tf.keras.metrics.BinaryAccuracy ( tf.argmax ( y_val, 1 ), tf.argmax ( y_vp, 1 ) )
        tf.print ( 'val_acc:', val_acc )

    print ( '---------------------------Pi Model TEST--------------------------' )
    test_accuracy, precision_true, precision_fake, recall_true, recall_fake, f1score_true, f1score_fake, AUC = prec_rec_f1score (
        args, y_test, x_test, pi_model )
    report_writing ( args, fold, args.model + '_' + 'BERT', args.lr, args.batch_size, args.epochs, args.alpha,
                     args.ratio, train_acc.numpy (), test_accuracy,
                     precision_true, precision_fake, recall_true, recall_fake, f1score_true, f1score_fake, AUC )
    # pi_model.save (f'{args.model_output_folder}/{args.data}/{args.model}_BERT_{args.alpha}_{args.pretrained_model}_fold-{fold}')
    print ( '-----------------------------------------------------------------' )
    tf.keras.backend.clear_session ()
    return pi_model

