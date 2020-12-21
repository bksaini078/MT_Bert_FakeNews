import datetime
import pandas as pd
from pathlib import Path

def report_writing(args,Model,lr,Batch_Size, Epoch,Alpha,Ratio, train_accuracy,test_accuracy,precision_true,precision_fake,recall_true,recall_fake,f1score_true,f1score_fake,AUC,comment):
    x = datetime.datetime.now()
    report_df = pd.DataFrame(columns=['Date', 'Model','Learning Rate','Batch_Size', 'Epoch','Alpha','Ratio','Train_Accuracy',
                                      'Test_Accuracy', 'Precision_True','Precision_Fake','Recall_True','Recall_Fake','F1_Score_True','F1_Score_Fake','AUC',
                                      'comment'])
    report_df = report_df.append({'Date' : x.strftime("%c"), 'Model' :Model,'Learning Rate':lr,'Batch_Size' : Batch_Size, 'Epoch': Epoch,'Alpha': Alpha,'Ratio': Ratio,'Train_Accuracy': train_accuracy,
                                  'Test_Accuracy': test_accuracy, 'Precision_True': precision_true,'Precision_Fake': precision_fake,'Recall_True': recall_true,'Recall_Fake': recall_fake,'F1_Score_True': f1score_true,'F1_Score_Fake': f1score_fake, 'AUC':AUC,'comment': comment}, ignore_index=True)  #my_file = Path(path+'/report_synonym_unlabelledDifference_0.99_max_len.csv')
    my_file = Path('Reports/'+args.data+'/' + args.model+'_'+args.method+'.csv')
    print(my_file)
    if my_file.exists():
        report_df.to_csv(my_file,mode='a', header= False , index = False)
    else:
        report_df.to_csv(my_file,mode='w', header= True , index= False)
    return 


# report_writing('Supervised-BiLstm', 124,10,34, 0.5, 0.99,0.90,0.90,0.90)