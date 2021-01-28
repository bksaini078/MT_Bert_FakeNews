#echo Fakehealth Full Label Mean Teacher
#python -m main \
#--data_folder Data/ExperimentFolds/3 \
#--data fakehealth \
#--model MT \
#--alpha 0.758 \
#--batch_size 4 \
#--epochs 3 \
#--max_len 300 \
#--ratio 0.5 \
#--pretrained_model distilbert-base-uncased \
#--lr 2e-5 \
#--loss_fn kl_divergence \
#--ratio_label 1.0 \
#--student_dropout 0.1 \
#--teacher_dropout 0.2
#
#echo Fakes Full Label Mean Teacher
#python -m main \
#--data_folder Data/ExperimentFolds/3 \
#--data fakes \
#--model MT \
#--alpha 0.758 \
#--batch_size 4 \
#--epochs 3 \
#--max_len 300 \
#--ratio 0.5 \
#--pretrained_model distilbert-base-uncased \
#--lr 2e-5 \
#--loss_fn kl_divergence \
#--ratio_label 1.0 \
#--student_dropout 0.1 \
#--teacher_dropout 0.2
#
#echo Gossipcop Full Label Mean Teacher
#python -m main \
#--data_folder Data/ExperimentFolds/3 \
#--data gossipcop \
#--model MT \
#--alpha 0.758 \
#--batch_size 4 \
#--epochs 3 \
#--max_len 300 \
#--ratio 0.5 \
#--pretrained_model distilbert-base-uncased \
#--lr 2e-5 \
#--loss_fn kl_divergence \
#--ratio_label 1.0 \
#--student_dropout 0.1 \
#--teacher_dropout 0.2
#
#echo Nela Full Label Mean Teacher
#python -m main \
#--data_folder Data/ExperimentFolds/3 \
#--data nela \
#--model MT \
#--alpha 0.758 \
#--batch_size 4 \
#--epochs 3 \
#--max_len 300 \
#--ratio 0.5 \
#--pretrained_model distilbert-base-uncased \
#--lr 2e-5 \
#--loss_fn kl_divergence \
#--ratio_label 1.0 \
#--student_dropout 0.1 \
#--teacher_dropout 0.2
#
#echo Covid Full Label Mean Teacher
#python -m main \
#--data_folder Data/ExperimentFolds/3 \
#--data covid \
#--model MT \
#--alpha 0.758 \
#--batch_size 4 \
#--epochs 3 \
#--max_len 300 \
#--ratio 0.5 \
#--pretrained_model distilbert-base-uncased \
#--lr 2e-5 \
#--loss_fn kl_divergence \
#--ratio_label 1.0 \
#--student_dropout 0.1 \
#--teacher_dropout 0.2


echo Fakehealth 0.5 Mean Teacher
python -m main \
--data_folder Data/ExperimentFolds/3 \
--data fakehealth \
--model MT \
--alpha 0.758 \
--batch_size 4 \
--epochs 3 \
--max_len 300 \
--ratio 0.5 \
--pretrained_model distilbert-base-uncased \
--lr 2e-5 \
--loss_fn kl_divergence \
--ratio_label 0.5 \
--student_dropout 0.1 \
--teacher_dropout 0.2

echo Fakes 0.5 Mean Teacher
python -m main \
--data_folder Data/ExperimentFolds/3 \
--data fakes \
--model MT \
--alpha 0.758 \
--batch_size 4 \
--epochs 3 \
--max_len 300 \
--ratio 0.5 \
--pretrained_model distilbert-base-uncased \
--lr 2e-5 \
--loss_fn kl_divergence \
--ratio_label 0.5 \
--student_dropout 0.1 \
--teacher_dropout 0.2

echo Gossipcop 0.5 Mean Teacher
python -m main \
--data_folder Data/ExperimentFolds/3 \
--data gossipcop \
--model MT \
--alpha 0.758 \
--batch_size 4 \
--epochs 3 \
--max_len 300 \
--ratio 0.5 \
--pretrained_model distilbert-base-uncased \
--lr 2e-5 \
--loss_fn kl_divergence \
--ratio_label 0.5 \
--student_dropout 0.1 \
--teacher_dropout 0.2

echo Nela 0.5 Mean Teacher
python -m main \
--data_folder Data/ExperimentFolds/3 \
--data nela \
--model MT \
--alpha 0.758 \
--batch_size 4 \
--epochs 3 \
--max_len 300 \
--ratio 0.5 \
--pretrained_model distilbert-base-uncased \
--lr 2e-5 \
--loss_fn kl_divergence \
--ratio_label 0.5 \
--student_dropout 0.1 \
--teacher_dropout 0.2

echo Covid 0.5 Mean Teacher
python -m main \
--data_folder Data/ExperimentFolds/3 \
--data covid \
--model MT \
--alpha 0.758 \
--batch_size 4 \
--epochs 3 \
--max_len 300 \
--ratio 0.5 \
--pretrained_model distilbert-base-uncased \
--lr 2e-5 \
--loss_fn kl_divergence \
--ratio_label 0.5 \
--student_dropout 0.1 \
--teacher_dropout 0.2


echo Fakehealth Full Label Mean Teacher
python -m main \
--data_folder Data/ExperimentFolds/3 \
--data fakehealth \
--model MT \
--alpha 0.758 \
--batch_size 4 \
--epochs 3 \
--max_len 300 \
--ratio 0.5 \
--pretrained_model distilbert-base-uncased \
--lr 2e-5 \
--loss_fn kl_divergence \
--ratio_label 0.1 \
--student_dropout 0.1 \
--teacher_dropout 0.2

echo Fakes Full Label Mean Teacher
python -m main \
--data_folder Data/ExperimentFolds/3 \
--data fakes \
--model MT \
--alpha 0.758 \
--batch_size 4 \
--epochs 3 \
--max_len 300 \
--ratio 0.5 \
--pretrained_model distilbert-base-uncased \
--lr 2e-5 \
--loss_fn kl_divergence \
--ratio_label 0.1 \
--student_dropout 0.1 \
--teacher_dropout 0.2

echo Gossipcop Full Label Mean Teacher
python -m main \
--data_folder Data/ExperimentFolds/3 \
--data gossipcop \
--model MT \
--alpha 0.758 \
--batch_size 4 \
--epochs 3 \
--max_len 300 \
--ratio 0.5 \
--pretrained_model distilbert-base-uncased \
--lr 2e-5 \
--loss_fn kl_divergence \
--ratio_label 0.1 \
--student_dropout 0.1 \
--teacher_dropout 0.2

echo Nela Full Label Mean Teacher
python -m main \
--data_folder Data/ExperimentFolds/3 \
--data nela \
--model MT \
--alpha 0.758 \
--batch_size 4 \
--epochs 3 \
--max_len 300 \
--ratio 0.5 \
--pretrained_model distilbert-base-uncased \
--lr 2e-5 \
--loss_fn kl_divergence \
--ratio_label 0.1 \
--student_dropout 0.1 \
--teacher_dropout 0.2

echo Covid Full Label Mean Teacher
python -m main \
--data_folder Data/ExperimentFolds/3 \
--data covid \
--model MT \
--alpha 0.758 \
--batch_size 4 \
--epochs 3 \
--max_len 300 \
--ratio 0.5 \
--pretrained_model distilbert-base-uncased \
--lr 2e-5 \
--loss_fn kl_divergence \
--ratio_label 0.1 \
--student_dropout 0.1 \
--teacher_dropout 0.2