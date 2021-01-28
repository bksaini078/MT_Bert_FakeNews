from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import metrics
import tensorflow as tf

#TODO could your organize this, make it similar with DistillBERT evaluation, it is hard to follow score while writing to table.
def prec_rec_f1score(args,y_true,x_test,model):
    if args.model =='PI':
        y_hat1,y_hat2= model.predict(x_test)
        y_hat=(y_hat1+y_hat2)/2
        y_pred= tf.argmax(y_hat,1)
    else:
        y_hat= model.predict(x_test)
        y_pred=tf.argmax(y_hat,1)
    y_true=tf.argmax(y_true,1)

    tf.print(classification_report(y_true=y_true, y_pred=y_pred, digits=4))
    pr_re_f1score_perclass= precision_recall_fscore_support(y_true, y_pred, average=None)
    accuracy= accuracy_score(y_true,y_pred)
    #per class
    precision_true=pr_re_f1score_perclass[0][1]
    precision_fake=pr_re_f1score_perclass[0][0]
    recall_true=pr_re_f1score_perclass[1][1]
    recall_fake=pr_re_f1score_perclass[1][0]
    f1score_true= pr_re_f1score_perclass[2][1]
    f1score_fake= pr_re_f1score_perclass[2][0]


    # AUC
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred, pos_label=None)
    AUC= metrics.auc(fpr, tpr)

    metrices_name=['accuracy','precision_true','precision_fake','recall_true','recall_fake','f1score_true','f1score_fake', 'AUC']
    metrices_value=[accuracy, precision_true, precision_fake, recall_true, recall_fake, f1score_true, f1score_fake, AUC]
    i=0
    for item in metrices_name:
        tf.print(item +':' ,metrices_value[i])
        i+=1
    tf.print("F1-Macro")
    f1_macro= f1_score(y_true, y_pred, average='macro')
    tf.print(f1_score(y_true, y_pred, average='macro'))
    tf.print("F1-Micro")
    f1_micro= f1_score(y_true, y_pred, average='micro')
    tf.print(f1_score(y_true, y_pred, average='micro'))
    tf.print("F1-Weighted")
    f1_weighted= f1_score(y_true, y_pred, average='weighted')
    tf.print(f1_score(y_true, y_pred, average='weighted'))

    return accuracy, precision_true, precision_fake, recall_true, recall_fake, f1score_true, f1score_fake,f1_macro,f1_micro,f1_weighted,AUC