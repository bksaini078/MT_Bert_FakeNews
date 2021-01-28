echo Fakehealth Full Label Mean Teacher
python -m src.main \
--pretrained_model distilbert-base-uncased \
--data_folder 'Data/ExperimentFolds/3' \
--data fakehealth \
--model_output_folder trained_models \
--model bert \
--max_len 300 \
--dropout 0.1 \
--batch_size 4 \
--lr 2e-5 \
--seed 42 \
--do_train \
--epochs 3 \
--ratio_label 1.0

echo NELA Full Label Mean Teacher
python -m src.main \
--pretrained_model distilbert-base-uncased \
--data_folder 'Data/ExperimentFolds/3' \
--data nela \
--model_output_folder trained_models \
--model bert \
--max_len 300 \
--dropout 0.1 \
--batch_size 4 \
--lr 2e-5 \
--seed 42 \
--do_train \
--epochs 3 \
--ratio_label 1.0

echo COVID Full Label Mean Teacher
python -m src.main \
--pretrained_model distilbert-base-uncased \
--data_folder 'Data/ExperimentFolds/3' \
--data covid \
--model_output_folder trained_models \
--model bert \
--max_len 300 \
--dropout 0.1 \
--batch_size 4 \
--lr 2e-5 \
--seed 42 \
--do_train \
--epochs 3 \
--ratio_label 1.0

echo Gossipcop Full Label Mean Teacher
python -m src.main \
--pretrained_model distilbert-base-uncased \
--data_folder 'Data/ExperimentFolds/3' \
--data gossipcop \
--model_output_folder trained_models \
--model bert \
--max_len 300 \
--dropout 0.1 \
--batch_size 4 \
--lr 2e-5 \
--seed 42 \
--do_train \
--epochs 3 \
--ratio_label 1.0

echo Fakes Full Label Mean Teacher
python -m src.main \
--pretrained_model distilbert-base-uncased \
--data_folder 'Data/ExperimentFolds/3' \
--data fakes \
--model_output_folder trained_models \
--model bert \
--max_len 300 \
--dropout 0.1 \
--batch_size 4 \
--lr 2e-5 \
--seed 42 \
--do_train \
--epochs 3 \
--ratio_label 1.0


echo Fakehealth 0.5 Label Mean Teacher
python -m src.main \
--pretrained_model distilbert-base-uncased \
--data_folder 'Data/ExperimentFolds/3' \
--data fakehealth \
--model_output_folder trained_models \
--model bert \
--max_len 300 \
--dropout 0.1 \
--batch_size 4 \
--lr 2e-5 \
--seed 42 \
--do_train \
--epochs 3 \
--ratio_label 0.5

echo NELA 0.5 Label Mean Teacher
python -m src.main \
--pretrained_model distilbert-base-uncased \
--data_folder 'Data/ExperimentFolds/3' \
--data nela \
--model_output_folder trained_models \
--model bert \
--max_len 300 \
--dropout 0.1 \
--batch_size 4 \
--lr 2e-5 \
--seed 42 \
--do_train \
--epochs 3 \
--ratio_label 0.5

echo COVID 0.5 Mean Teacher
python -m src.main \
--pretrained_model distilbert-base-uncased \
--data_folder 'Data/ExperimentFolds/3' \
--data covid \
--model_output_folder trained_models \
--model bert \
--max_len 300 \
--dropout 0.1 \
--batch_size 4 \
--lr 2e-5 \
--seed 42 \
--do_train \
--epochs 3 \
--ratio_label 0.5

echo Gossipcop 0.5 Mean Teacher
python -m src.main \
--pretrained_model distilbert-base-uncased \
--data_folder 'Data/ExperimentFolds/3' \
--data gossipcop \
--model_output_folder trained_models \
--model bert \
--max_len 300 \
--dropout 0.1 \
--batch_size 4 \
--lr 2e-5 \
--seed 42 \
--do_train \
--epochs 3 \
--ratio_label 0.5

echo Fakes 0.5 Mean Teacher
python -m src.main \
--pretrained_model distilbert-base-uncased \
--data_folder 'Data/ExperimentFolds/3' \
--data fakes \
--model_output_folder trained_models \
--model bert \
--max_len 300 \
--dropout 0.1 \
--batch_size 4 \
--lr 2e-5 \
--seed 42 \
--do_train \
--epochs 3 \
--ratio_label 0.5




echo Fakehealth 0.1 Mean Teacher
python -m src.main \
--pretrained_model distilbert-base-uncased \
--data_folder 'Data/ExperimentFolds/3' \
--data fakehealth \
--model_output_folder trained_models \
--model bert \
--max_len 300 \
--dropout 0.1 \
--batch_size 4 \
--lr 2e-5 \
--seed 42 \
--do_train \
--epochs 3 \
--ratio_label 0.1

echo NELA 0.1 Mean Teacher
python -m src.main \
--pretrained_model distilbert-base-uncased \
--data_folder 'Data/ExperimentFolds/3' \
--data nela \
--model_output_folder trained_models \
--model bert \
--max_len 300 \
--dropout 0.1 \
--batch_size 4 \
--lr 2e-5 \
--seed 42 \
--do_train \
--epochs 3 \
--ratio_label 0.1

echo COVID 0.1 Mean Teacher
python -m src.main \
--pretrained_model distilbert-base-uncased \
--data_folder 'Data/ExperimentFolds/3' \
--data covid \
--model_output_folder trained_models \
--model bert \
--max_len 300 \
--dropout 0.1 \
--batch_size 4 \
--lr 2e-5 \
--seed 42 \
--do_train \
--epochs 3 \
--ratio_label 0.1

echo Gossipcop 0.1 Mean Teacher
python -m src.main \
--pretrained_model distilbert-base-uncased \
--data_folder 'Data/ExperimentFolds/3' \
--data gossipcop \
--model_output_folder trained_models \
--model bert \
--max_len 300 \
--dropout 0.1 \
--batch_size 4 \
--lr 2e-5 \
--seed 42 \
--do_train \
--epochs 3 \
--ratio_label 0.1

echo Fakes 0.1 Mean Teacher
python -m src.main \
--pretrained_model distilbert-base-uncased \
--data_folder 'Data/ExperimentFolds/3' \
--data fakes \
--model_output_folder trained_models \
--model bert \
--max_len 300 \
--dropout 0.1 \
--batch_size 4 \
--lr 2e-5 \
--seed 42 \
--do_train \
--epochs 3 \
--ratio_label 0.1