#python -m Mean_Teacher.clf.main \
#--model bert \
#--data_folder 'Data/ExperimentFolds/3' \
#--data fakehealth \
#--model_output_folder trained_models \
#--pretrained_model bert-base-cased \
#--max_len 512 \
#--dropout 0.1 \
#--epochs 1 \
#--batch_size 1 \
#--lr 2e-5 \
#--seed 42 \
#--do_train
#
#python -m Mean_Teacher.clf.main \
#--model bert \
#--data_folder 'Data/ExperimentFolds/4' \
#--data fakehealth \
#--model_output_folder trained_models \
#--pretrained_model bert-base-cased \
#--max_len 512 \
#--dropout 0.1 \
#--epochs 100 \
#--batch_size 1 \
#--lr 2e-5 \
#--seed 42 \
#--do_train
#
#python -m Mean_Teacher.clf.main \
#--model bert \
#--data_folder 'Data/ExperimentFolds/5' \
#--data fakehealth \
#--model_output_folder trained_models \
#--pretrained_model bert-base-cased \
#--max_len 512 \
#--dropout 0.1 \
#--epochs 100 \
#--batch_size 1 \
#--lr 2e-5 \
#--seed 42 \
#--do_train
#
#python -m Mean_Teacher.clf.main \
#--model bert \
#--data_folder 'Data/ExperimentFolds/6' \
#--data fakehealth \
#--model_output_folder trained_models \
#--pretrained_model bert-base-cased \
#--max_len 512 \
#--dropout 0.1 \
#--epochs 100 \
#--batch_size 1 \
#--lr 2e-5 \
#--seed 42 \
#--do_train
#
#python -m Mean_Teacher.clf.main \
#--model bert \
#--data_folder 'Data/ExperimentFolds/7' \
#--data fakehealth \
#--model_output_folder trained_models \
#--pretrained_model bert-base-cased \
#--max_len 512 \
#--dropout 0.1 \
#--epochs 100 \
#--batch_size 1 \
#--lr 2e-5 \
#--seed 42 \
#--do_train
#
#python -m Mean_Teacher.clf.main \
#--model bert \
#--data_folder 'Data/ExperimentFolds/8' \
#--data fakehealth \
#--model_output_folder trained_models \
#--pretrained_model bert-base-cased \
#--max_len 512 \
#--dropout 0.1 \
#--epochs 100 \
#--batch_size 1 \
#--lr 2e-5 \
#--seed 42 \
#--do_train
#
#python -m Mean_Teacher.clf.main \
#--model bert \
#--data_folder 'Data/ExperimentFolds/9' \
#--data fakehealth \
#--model_output_folder trained_models \
#--pretrained_model bert-base-cased \
#--max_len 512 \
#--dropout 0.1 \
#--epochs 100 \
#--batch_size 1 \
#--lr 2e-5 \
#--seed 42 \
#--do_train

#python -m Mean_Teacher.clf.main \
#--model bert \
#--data_folder 'Data/ExperimentFolds/3' \
#--data gossipcop \
#--model_output_folder trained_models \
#--pretrained_model bert-base-cased \
#--max_len 512 \
#--dropout 0.1 \
#--epochs 1 \
#--batch_size 1 \
#--lr 2e-5 \
#--seed 42 \
#--do_train
#
#python -m Mean_Teacher.clf.main \
#--model bert \
#--data_folder 'Data/ExperimentFolds/4' \
#--data gossipcop \
#--model_output_folder trained_models \
#--pretrained_model bert-base-cased \
#--max_len 512 \
#--dropout 0.1 \
#--epochs 100 \
#--batch_size 1 \
#--lr 2e-5 \
#--seed 42 \
#--do_train
#
#python -m Mean_Teacher.clf.main \
#--model bert \
#--data_folder 'Data/ExperimentFolds/5' \
#--data gossipcop \
#--model_output_folder trained_models \
#--pretrained_model bert-base-cased \
#--max_len 512 \
#--dropout 0.1 \
#--epochs 100 \
#--batch_size 1 \
#--lr 2e-5 \
#--seed 42 \
#--do_train
#
#python -m Mean_Teacher.clf.main \
#--model bert \
#--data_folder 'Data/ExperimentFolds/6' \
#--data gossipcop \
#--model_output_folder trained_models \
#--pretrained_model bert-base-cased \
#--max_len 512 \
#--dropout 0.1 \
#--epochs 100 \
#--batch_size 1 \
#--lr 2e-5 \
#--seed 42 \
#--do_train
#
#python -m Mean_Teacher.clf.main \
#--model bert \
#--data_folder 'Data/ExperimentFolds/7' \
#--data gossipcop \
#--model_output_folder trained_models \
#--pretrained_model bert-base-cased \
#--max_len 512 \
#--dropout 0.1 \
#--epochs 100 \
#--batch_size 1 \
#--lr 2e-5 \
#--seed 42 \
#--do_train
#
#python -m Mean_Teacher.clf.main \
#--model bert \
#--data_folder 'Data/ExperimentFolds/8' \
#--data gossipcop \
#--model_output_folder trained_models \
#--pretrained_model bert-base-cased \
#--max_len 512 \
#--dropout 0.1 \
#--epochs 100 \
#--batch_size 1 \
#--lr 2e-5 \
#--seed 42 \
#--do_train
#
#python -m Mean_Teacher.clf.main \
#--model bert \
#--data_folder 'Data/ExperimentFolds/9' \
#--data gossipcop \
#--model_output_folder trained_models \
#--pretrained_model bert-base-cased \
#--max_len 512 \
#--dropout 0.1 \
#--epochs 100 \
#--batch_size 1 \
#--lr 2e-5 \
#--seed 42 \
#--do_train

python -m Mean_Teacher.clf.main \
--model bert \
--data_folder 'Data/ExperimentFolds/3' \
--data politifact \
--model_output_folder trained_models \
--pretrained_model bert-base-cased \
--max_len 512 \
--dropout 0.1 \
--epochs 1 \
--batch_size 1 \
--lr 2e-5 \
--seed 42 \
--do_train

python -m Mean_Teacher.clf.main \
--model bert \
--data_folder 'Data/ExperimentFolds/4' \
--data politifact \
--model_output_folder trained_models \
--pretrained_model bert-base-cased \
--max_len 512 \
--dropout 0.1 \
--epochs 100 \
--batch_size 1 \
--lr 2e-5 \
--seed 42 \
--do_train

python -m Mean_Teacher.clf.main \
--model bert \
--data_folder 'Data/ExperimentFolds/5' \
--data politifact \
--model_output_folder trained_models \
--pretrained_model bert-base-cased \
--max_len 512 \
--dropout 0.1 \
--epochs 100 \
--batch_size 1 \
--lr 2e-5 \
--seed 42 \
--do_train

python -m Mean_Teacher.clf.main \
--model bert \
--data_folder 'Data/ExperimentFolds/6' \
--data politifact \
--model_output_folder trained_models \
--pretrained_model bert-base-cased \
--max_len 512 \
--dropout 0.1 \
--epochs 100 \
--batch_size 1 \
--lr 2e-5 \
--seed 42 \
--do_train

python -m Mean_Teacher.clf.main \
--model bert \
--data_folder 'Data/ExperimentFolds/7' \
--data politifact \
--model_output_folder trained_models \
--pretrained_model bert-base-cased \
--max_len 512 \
--dropout 0.1 \
--epochs 100 \
--batch_size 1 \
--lr 2e-5 \
--seed 42 \
--do_train

python -m Mean_Teacher.clf.main \
--model bert \
--data_folder 'Data/ExperimentFolds/8' \
--data politifact \
--model_output_folder trained_models \
--pretrained_model bert-base-cased \
--max_len 512 \
--dropout 0.1 \
--epochs 100 \
--batch_size 1 \
--lr 2e-5 \
--seed 42 \
--do_train

python -m Mean_Teacher.clf.main \
--model bert \
--data_folder 'Data/ExperimentFolds/9' \
--data politifact \
--model_output_folder trained_models \
--pretrained_model bert-base-cased \
--max_len 512 \
--dropout 0.1 \
--epochs 100 \
--batch_size 1 \
--lr 2e-5 \
--seed 42 \
--do_train