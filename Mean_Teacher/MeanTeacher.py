import tensorflow as tf 
import tensorflow.keras as tfk
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Lambda
from tensorflow.keras.models import Model
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import *
import tensorflow.keras.backend as K
from tensorflow.keras import regularizers
#this is to enable eager execution
tf.compat.v1.enable_eager_execution()

#calling functions

from costfunction import Classification_costs,Overall_Cost,Consistency_Cost,EMA
from report_writing import report_writing
from model_arch import BiLstmModel_attention, BiLstmModel
from noise_creator import instant_noise
from evaluation import prec_rec_f1score

def MeanTeacher(args, epochs, batch_size, alpha, lr, ratio, noise_ratio, x_train, y_train, x_val, y_val, x_test, y_test,
                      x_unlabel_tar, vocab_size, maxlen):
    # preparing the training dataset
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # preparing the target dataset
    tar_dataset = tf.data.Dataset.from_tensor_slices(x_unlabel_tar)
    tar_dataset = tar_dataset.shuffle(buffer_size=1024).batch(batch_size)

    # declaring optimiser
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=lr)  # trying changing learning rate , sometimes it gives good result
    train_metrics = tf.keras.metrics.BinaryAccuracy(name='Binary_Accuracy')
    val_acc_metric = tf.keras.metrics.BinaryAccuracy(name="Binary_Acc")
    teacher_acc_metric = tf.keras.metrics.BinaryAccuracy(name="Binary_Acc_teacher")

    # Creating model
    if args.method== 'BERT':
        student = BiLstmModel(maxlen, vocab_size)
        student.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='binary_crossentropy',
                        metrics=['accuracy'])
        teacher = BiLstmModel(maxlen, vocab_size)
        teacher.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='binary_crossentropy',
                        metrics=['accuracy'])
    elif args.method=='Attn':
        student = BiLstmModel_attention(maxlen, vocab_size)
        student.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='binary_crossentropy',
                        metrics=['accuracy'])
        teacher = BiLstmModel_attention(maxlen, vocab_size)
        teacher.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='binary_crossentropy',
                        metrics=['accuracy'])


    # collecting costs

    train_accuracy = []
    steps = []

   # this I am doing to get all steps details in epoch
    i = 0
    print('Train Mean teacher Model...')
    # teacher.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),loss=tf.keras.losses.BinaryCrossentropy(),metrics=['accuracy'])
    # teacher.fit(x_train, y_train, batch_size=batch_size, epochs=1)

    acc_t = 0
    # false positive rate and true positive rate
    fpr = []
    tpr = []
    # x_unlabel_tar= tf.convert_to_tensor(x_unlabel_tar)
    for epoch in range(1, epochs + 1):
        print(*"*****************")
        print('Start of epoch %d' % (epoch,))
        print(*"*****************")
        # iteration over batches
        for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
            with tf.GradientTape() as tape:
                # adding instant noise
                iterator_unlabel = iter(tar_dataset)
                x_batch_unlabel = iterator_unlabel.get_next()

                '''this is one method of adding -1 label using unlable data'''
                x_train_n, y_train_n = instant_noise(x_batch_train, y_batch_train, x_batch_unlabel, noise_ratio)

                # Run the forward pass of the layer
                logits = student(x_train_n, training=True)
                # logits_acc =  student(x_batch_sn, training= False)

                # doing this because then accuracy cannot be calculated
                logits_acc = student(x_batch_train, training=True)

                # calculating accuracy
                train_metrics(y_batch_train, logits_acc)

                # Calculating classification cost
                classification_cost = Classification_costs(logits, y_train_n)
                # classification.append(classification_cost)

                x_train_n1, _ = instant_noise(x_batch_train, y_batch_train, x_batch_unlabel, noise_ratio)

                tar_teacher = teacher(x_train_n1)  # x_batch_train
                #  tar_student= student(x_train_n)
                consistency_cost = Consistency_Cost(tar_teacher, logits)
                # consistency.append(consistency_cost)

                overall_cost = Overall_Cost(classification_cost, consistency_cost, ratio)

                # adding loss to student model
            grads = tape.gradient(overall_cost, student.trainable_weights)
            i = i + 1
            steps.append(i)

            # the value of the variables to minimize the loss.
            optimizer.apply_gradients(zip(grads, student.trainable_weights))
            teacher = EMA(student, teacher, alpha=args.alpha)

        train_acc = train_metrics.result()
        print(alpha)

        # appending training accuracy
        train_accuracy.append(train_acc)

        # Reset training metrics at the end of each epoch
        train_metrics.reset_states()

        # Run a validation loop at the end of each epoch.
        # print('*******STUDENT*************')
        # prec_rec_f1score(y_val, x_val, student)
        print('*******TEACHER*************')
        prec_rec_f1score(y_val, x_val, teacher)

        if  epoch %1 == 0:
            # print('---------------------------STUDENT--------------------------')
            # test_accuracy, precision_true, precision_fake, recall_true, recall_fake, f1score_true, f1score_fake, AUC = prec_rec_f1score(
            #     y_test, x_test, student)
            # report_writing(args,'Student', lr, batch_size, epoch, alpha, ratio, train_acc.numpy(),
            #                test_accuracy, precision_true, precision_fake, recall_true, recall_fake,
            #                f1score_true, f1score_fake, AUC, 'BiLSTM-'+args.method+'-MT-'+args.unlabel)
            print('------------------------WITH TEST DATA-----------------------------------------')

            print('---------------------------TEACHER---------------------------------')

            test_accuracy, precision_true, precision_fake, recall_true, recall_fake, f1score_true, f1score_fake, AUC = prec_rec_f1score(
                y_test, x_test, teacher)


            print('*'*80)
        # if epoch >= 10 and epoch% 5==0 :
        #     teacher.save(path+'/model/MT_unlabel_'+str(epoch))
    report_writing(args,'Teacher', lr, batch_size, epoch, alpha, ratio, train_acc.numpy(),test_accuracy, 
    precision_true, precision_fake, recall_true, recall_fake,f1score_true, f1score_fake, AUC, 'BiLSTM-'+args.method+'-MT-'+args.unlabel)
    tf.keras.backend.clear_session()
    return 


    