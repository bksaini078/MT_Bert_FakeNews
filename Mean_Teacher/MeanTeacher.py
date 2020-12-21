import tensorflow as tf 
#this is to enable eager execution
tf.compat.v1.enable_eager_execution()

#calling functions

from costfunction import Classification_costs,Overall_Cost,Consistency_Cost,EMA
from report_writing import report_writing
from model_arch import BiLstmModel_attention, BiLstmModel
from noise_creator import instant_noise
from evaluation import prec_rec_f1score
from data_loader import data_slices
from clf.bert import BERT

# def Training_MT_bert(args, train_dataset,tar_dataset, student, teacher):




def MeanTeacher(args, epochs, batch_size, alpha, lr, ratio, noise_ratio, x_train, y_train, x_val, y_val, x_test, y_test,
                      x_unlabel_tar, vocab_size, max_len):
    initial_beta1 = 0.9
    final_beta1 = 0.5
    max_learning_rate=0.003
    # preparing the training dataset
    train_dataset, tar_dataset = data_slices( args, x_train, y_train, x_unlabel_tar )
    learning_rate = tf.Variable(max_learning_rate) #max learning rate
    beta_1 = tf.Variable(initial_beta1)
    # optimizer= tf.keras.optimizers.Adam(learning_rate= learning_rate, beta_1=beta_1,beta_2=0.999) 
    # declaring optimiser
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)  # trying changing learning rate , sometimes it gives good result
    train_metrics = tf.keras.metrics.BinaryAccuracy(name='Binary_Accuracy')

    # Creating model
    if args.method=='Attn':
        student = BiLstmModel_attention(max_len, vocab_size)
        student.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        teacher = BiLstmModel_attention ( max_len, vocab_size )
        teacher.compile ( optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'] )
    elif args.method=='Bert':
        models = BERT(args)
        student = models.create_model()
        modelt = BERT(args)
        teacher = modelt.create_model()
    else:
        print('Either correct model name ')

    for epoch in range(1, args.epochs + 1):
        print(*"*****************")
        print('Start of epoch %d' % (epoch,))
        print(*"*****************")
        # iteration over batches
        iterator_unlabel = iter ( tar_dataset )
        if args.method=='Attn':
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    # adding instant noise
                    x_batch_unlabel = iterator_unlabel.get_next()
                    '''this is one method of adding -1 label using unlable data'''
                    # x_train_n, y_train_n = instant_noise(x_batch_train, y_batch_train, x_batch_unlabel, noise_ratio)
                    overall_cost, train_acc = Overall_Cost(args,x_batch_train,y_batch_train,x_batch_unlabel, student, teacher, args.ratio)
                grads = tape.gradient(overall_cost, student.trainable_weights)
                # the value of the variables to minimize the loss.
                optimizer.apply_gradients(zip(grads, student.trainable_weights))
                teacher = EMA(student, teacher, alpha=args.alpha)
        elif args.method=='Bert':
            for step, (inputs, attention, token_id, y_batch_train) in enumerate ( train_dataset ) :
                with tf.GradientTape () as tape :
                    inp, att, to_id = iterator_unlabel.get_next ()
                    overall_cost, train_acc = overall_cost(args,[inputs, attention, token_id], y_batch_train, [inp, att, to_id],
                                                 student, teacher, args.ratio)
                grads = tape.gradient(overall_cost, student.trainable_weights )
                    # the value of the variables to minimize the loss.
                optimizer.apply_gradients(zip( grads, student.trainable_weights))
                teacher = EMA(student, teacher, alpha=args.alpha)
            print(train_acc)

        # Run a validation loop at the end of each epoch.
        print('*******STUDENT*************')
        prec_rec_f1score(args,y_val, x_val, student)
        print('*******TEACHER*************')
        prec_rec_f1score(args,y_val, x_val, teacher)

        if  epoch %5 == 0:
            # print('---------------------------STUDENT--------------------------')
            # test_accuracy, precision_true, precision_fake, recall_true, recall_fake, f1score_true, f1score_fake, AUC = prec_rec_f1score(
            #     args,y_test, x_test, student)
            print('------------------------WITH TEST DATA-----------------------------------------')

            print('---------------------------TEACHER---------------------------------')

            test_accuracy, precision_true, precision_fake, recall_true, recall_fake, f1score_true, f1score_fake, AUC = prec_rec_f1score(
                args,y_test, x_test, teacher)


            print('*'*80)
        # if epoch >= 10 and epoch% 5==0 :
        #     teacher.save(path+'/model/MT_unlabel_'+str(epoch))
    # report_writing(args,'Teacher', lr, batch_size, epoch, alpha, ratio, train_acc.numpy(),test_accuracy, 
    # precision_true, precision_fake, recall_true, recall_fake,f1score_true, f1score_fake, AUC, 'BiLSTM-'+args.method+'-MT-'+args.unlabel)
    tf.keras.backend.clear_session()
    return 


    