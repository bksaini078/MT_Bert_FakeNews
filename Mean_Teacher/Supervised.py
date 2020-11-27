import tensorflow as tf

from report_writing import report_writing
from model_arch import BiLstmModel_attention, BiLstmModel
from noise_creator import instant_noise
from evaluation import prec_rec_f1score

def train_supervised(args,epochs, batch_size, lr, x_train, y_train, x_val, y_val ,x_test, y_test ,maxlen ,vocab_size ):
    if args.method=='BERT':
        model_supervised = BiLstmModel(maxlen, vocab_size)
    elif args.method=='Attn':
        model_supervised= BiLstmModel_attention(maxlen, vocab_size)
        # model_supervised.summary()

    model_supervised.compile(optimizer= tf.keras.optimizers.Adam(learning_rate= lr ) ,loss= 'binary_crossentropy', metrics=['accuracy' ])
    print ('Training supervised Model... ')
    history =model_supervised.fit (x_train, y_train ,batch_size=batch_size ,epochs=epochs,validation_data=(x_val ,y_val ))


    # evaluation
    train_accuracy =history.history['accuracy'][len (history.epoch ) -1]
    test_accuracy ,precision_true ,precision_fake ,recall_true ,recall_fake ,f1score_true ,f1score_fake, AUC = prec_rec_f1score \
        (y_test ,x_test ,model_supervised)
    # cm, test_accuracy, precision, recall, f1_score =Confusion_matrix(model_supervised,x_test,y_test,0.51, 'Supervised model')
    report_writing (args,'BiLstm-'+args.method ,lr, batch_size ,len (history.epoch) ,'NaN' ,'NaN', train_accuracy,
                    test_accuracy, precision_true, precision_fake, recall_true, recall_fake,f1score_true, f1score_fake ,AUC,args.data)
    return
