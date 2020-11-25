from sklearn.metrics import precision_recall_fscore_support
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import metrics


def prec_rec_f1score(y_true,x_test,model):
    y_hat= model.predict(x_test)
    y_pred=(np.greater_equal(y_hat,0.51)).astype(int)

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
    fpr, tpr, thresholds = metrics.roc_curve(y_true, y_hat, pos_label=None)
    AUC= metrics.auc(fpr, tpr)

    metrices_name=['accuracy','precision_true','precision_fake','recall_true','recall_fake','f1score_true','f1score_fake', 'AUC']
    metrices_value=[accuracy, precision_true, precision_fake, recall_true, recall_fake, f1score_true, f1score_fake, AUC]
    i=0
    for item in metrices_name:
        print(item +':' ,metrices_value[i])
        i+=1

    return accuracy, precision_true, precision_fake, recall_true, recall_fake, f1score_true, f1score_fake, AUC