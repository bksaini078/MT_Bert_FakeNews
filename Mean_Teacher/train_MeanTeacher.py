import tensorflow as tf

tf.compat.v1.enable_eager_execution()
from Mean_Teacher.costfunction import Overall_Cost, EMA
from Mean_Teacher.report_writing import report_writing
from Mean_Teacher.evaluation import prec_rec_f1score
from Mean_Teacher.data_loader import data_slices
from BERT.bert import BERT
import numpy as np
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_fscore_support


def MeanTeacher(args, fold, x_train, y_train,  x_test, y_test, x_unlabel_tar) :
    # preparing the training dataset
    train_dataset, unlabel_dataset = data_slices(args, x_train, y_train,x_unlabel_tar)
    
    # declaring optimiser
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.lr) 
    
    #creating model
    student = BERT(args).create_model()
    teacher = BERT(args).create_model()
    
    # declaring metrics
    train_metrics = tf.keras.metrics.BinaryAccuracy(name='Binary_Accuracy')
    
    progbar = tf.keras.utils.Progbar(len(train_dataset), stateful_metrics=['Accuracy', 'Overall_Loss'])
    
    #training
    for epoch in range(1, args.epochs + 1) :
        tf.print('\nepoch %d' %(epoch,))
        for step,(inputs, attention,  y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape :
                iterator_unlabel = iter(unlabel_dataset)
                x_batch_unlabel = iterator_unlabel.get_next()
                overall_cost = Overall_Cost(args, [inputs, attention], y_batch_train, x_batch_unlabel,
                                              student, teacher)
              
            grads = tape.gradient(overall_cost, student.trainable_weights)
            optimizer.apply_gradients((grad, var) for(grad, var) in zip(grads, student.trainable_weights)if grad is not None)

            # applying student weights to teacher
            teacher = EMA(student, teacher, alpha=args.alpha)

            #calculating training accuracy
            logits_t = teacher([inputs, attention])
            train_acc = train_metrics(tf.argmax(y_batch_train, 1), tf.argmax(logits_t, 1))
            progbar.update(step, values=[('Accuracy', train_acc),('Overall_Loss', overall_cost)])


    #Evaluation of the model
    tf.print("\nStudent Testing data evaluation:")
    test_accuracy, precision_true, precision_fake, recall_true, recall_fake, f1score_true, f1score_fake, f1_macro,f1_micro,f1_weighted,AUC = prec_rec_f1score(
        args, y_test, x_test, student)
    
    tf.print("\nTeacher Testing data evaluation:")
    test_accuracy, precision_true, precision_fake, recall_true, recall_fake, f1score_true, f1score_fake, f1_macro,f1_micro,f1_weighted,AUC = prec_rec_f1score(
        args, y_test, x_test, teacher)
    # teacher.save(f'{args.model_output_folder}/{args.data}/{args.model}_BERT_{args.alpha}_{args.pretrained_model}_fold-{fold}')
    
    # reporting in report file 
    report_writing(args,  train_acc.numpy(), test_accuracy,precision_true, precision_fake, recall_true, recall_fake, f1score_true, f1score_fake,f1_macro,f1_micro,f1_weighted, AUC)
    tf.keras.backend.clear_session()
    return


