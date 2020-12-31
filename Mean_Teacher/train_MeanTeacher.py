import tensorflow as tf

tf.compat.v1.enable_eager_execution ()
from Mean_Teacher.costfunction import Overall_Cost, EMA
from Mean_Teacher.report_writing import report_writing
from Mean_Teacher.evaluation import prec_rec_f1score
from Mean_Teacher.data_loader import data_slices
from BERT.bert import BERT


def MeanTeacher(args, fold, x_train, y_train, x_val, y_val, x_test, y_test, x_unlabel_tar, vocab_size) :
    # preparing the training dataset
    train_dataset = data_slices ( args, x_train, y_train )
    # declaring optimiser
    optimizer = tf.keras.optimizers.Adam (
        learning_rate=args.lr )  # trying changing learning rate , sometimes it gives good result
    student = BERT ( args ).create_model ()
    teacher = BERT ( args ).create_model ()
    train_metrics = tf.keras.metrics.BinaryAccuracy ( name='Binary_Accuracy' )
    progbar = tf.keras.utils.Progbar ( len ( train_dataset ), stateful_metrics=['Accuracy', 'Loss'] )
    for epoch in range ( 1, args.epochs + 1 ) :
        tf.print ( 'epoch %d' % (epoch,) )
        for step, (inputs, attention, token_id, y_batch_train) in enumerate ( train_dataset ) :
            with tf.GradientTape () as tape :
                overall_cost = Overall_Cost ( args, [inputs, attention, token_id], y_batch_train, x_unlabel_tar,
                                              student, teacher )
            grads = tape.gradient ( overall_cost, student.trainable_weights )
            optimizer.apply_gradients (
                (grad, var) for (grad, var) in zip ( grads, student.trainable_weights ) if grad is not None )
            teacher = EMA ( student, teacher, alpha=args.alpha )
            logits_t = teacher ( [inputs, attention, token_id] )
            train_acc = train_metrics ( tf.argmax ( y_batch_train, 1 ), tf.argmax ( logits_t, 1 ) )
            progbar.add ( args.batch_size, values=[('Accuracy', train_acc), ('Loss', overall_cost)] )

        # Run a validation loop at the end of each epoch.
        print ( '*******TEACHER*************' )
        y_vp = teacher ( x_val )
        val_acc = tf.keras.metrics.BinaryAccuracy ( tf.argmax ( y_val, 1 ), tf.argmax ( y_vp, 1 ) )
        tf.print ( 'val_acc:', val_acc )

    print ( '------------------------WITH TEST DATA-----------------------------------------' )
    test_accuracy, precision_true, precision_fake, recall_true, recall_fake, f1score_true, f1score_fake, AUC = prec_rec_f1score (
        args, y_test, x_test, teacher )
    teacher.save (
        f'{args.model_output_folder}/{args.data}/{args.model}_BERT_{args.alpha}_{args.pretrained_model}_fold-{fold}' )
    report_writing ( args, fold, args.model + '_' + 'BERT' + '_Teacher', args.lr, args.batch_size, args.epochs,
                     args.alpha, args.ratio, train_acc.numpy (), test_accuracy,
                     precision_true, precision_fake, recall_true, recall_fake, f1score_true, f1score_fake, AUC )
    tf.keras.backend.clear_session ()
    return


