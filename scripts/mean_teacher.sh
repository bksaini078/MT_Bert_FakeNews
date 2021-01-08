python -m src.main \
--model mean_bert \
--data_folder 'Data/ExperimentFolds/3' \
--data fakehealth \
--model_output_folder trained_models \
--pretrained_model distilbert-base-uncased \
--max_len 300 \
--dropout 0.1 \
--batch_size 8 \
--lr 2e-5 \
--do_train \
--seed 42 \
--epochs 3 \
--unlabel_batch 8 \
--loss_weight 0.5 \
--decay 0.99 \
--model_option teacher \
--use_unlabel_data \
--loss_type mse

python -m src.main \
--model mean_bert \
--data_folder 'Data/ExperimentFolds/4' \
--data fakehealth \
--model_output_folder trained_models \
--pretrained_model distilbert-base-uncased \
--max_len 300 \
--dropout 0.1 \
--batch_size 8 \
--lr 2e-5 \
--do_train \
--seed 42 \
--epochs 3 \
--unlabel_batch 8 \
--loss_weight 0.5 \
--decay 0.99 \
--model_option teacher \
--use_unlabel_data \
--loss_type mse
#
python -m src.main \
--model mean_bert \
--data_folder 'Data/ExperimentFolds/5' \
--data fakehealth \
--model_output_folder trained_models \
--pretrained_model distilbert-base-uncased \
--max_len 300 \
--dropout 0.1 \
--batch_size 8 \
--lr 2e-5 \
--do_train \
--seed 42 \
--epochs 3 \
--unlabel_batch 8 \
--loss_weight 0.5 \
--decay 0.99 \
--model_option teacher \
--use_unlabel_data \
--loss_type mse
#
python -m src.main \
--model mean_bert \
--data_folder 'Data/ExperimentFolds/6' \
--data fakehealth \
--model_output_folder trained_models \
--pretrained_model distilbert-base-uncased \
--max_len 300 \
--dropout 0.1 \
--batch_size 8 \
--lr 2e-5 \
--do_train \
--seed 42 \
--epochs 3 \
--unlabel_batch 8 \
--loss_weight 0.5 \
--decay 0.99 \
--model_option teacher \
--use_unlabel_data \
--loss_type mse
#
#
python -m src.main \
--model mean_bert \
--data_folder 'Data/ExperimentFolds/7' \
--data fakehealth \
--model_output_folder trained_models \
--pretrained_model distilbert-base-uncased \
--max_len 300 \
--dropout 0.1 \
--batch_size 8 \
--lr 2e-5 \
--do_train \
--seed 42 \
--epochs 3 \
--unlabel_batch 8 \
--loss_weight 0.5 \
--decay 0.99 \
--model_option teacher \
--use_unlabel_data \
--loss_type mse
#
python -m src.main \
--model mean_bert \
--data_folder 'Data/ExperimentFolds/8' \
--data fakehealth \
--model_output_folder trained_models \
--pretrained_model distilbert-base-uncased \
--max_len 300 \
--dropout 0.1 \
--batch_size 8 \
--lr 2e-5 \
--do_train \
--seed 42 \
--epochs 3 \
--unlabel_batch 8 \
--loss_weight 0.5 \
--decay 0.99 \
--model_option teacher \
--use_unlabel_data \
--loss_type mse
#
python -m src.main \
--model mean_bert \
--data_folder 'Data/ExperimentFolds/9' \
--data fakehealth \
--model_output_folder trained_models \
--pretrained_model distilbert-base-uncased \
--max_len 300 \
--dropout 0.1 \
--batch_size 8 \
--lr 2e-5 \
--do_train \
--seed 42 \
--epochs 3 \
--unlabel_batch 8 \
--loss_weight 0.5 \
--decay 0.99 \
--model_option teacher \
--use_unlabel_data \
--loss_type mse