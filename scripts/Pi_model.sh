#make sure by using fake_news_generate the unlabel data for each fold, if already done then move forward
python3 -m Mean_Teacher.main \
--model PI \
--method Attn \
--data_folder Data/ExperimentsFolds/3 \
--data fakehealth \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data fakehealth \
--alpha 0.99 \
--ratio 0.5

python3 -m Mean_Teacher.main \
--model PI \
--method Attn \
--data_folder Data/ExperimentsFolds/4 \
--data fakehealth \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data fakehealth \
--alpha 0.99 \
--ratio 0.5

python3 -m Mean_Teacher.main \
--model PI \
--method Attn \
--data_folder Data/ExperimentsFolds/6 \
--data fakehealth \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data fakehealth \
--alpha 0.99 \
--ratio 0.5

python3 -m Mean_Teacher.main \
--model PI \
--method Attn \
--data_folder Data/ExperimentsFolds/7 \
--data fakehealth \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data fakehealth \
--alpha 0.99 \
--ratio 0.5

python3 -m Mean_Teacher.main \
--model PI \
--method Attn \
--data_folder Data/ExperimentsFolds/8 \
--data fakehealth \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data fakehealth \
--alpha 0.99 \
--ratio 0.5

python3 -m Mean_Teacher.main \
--model PI \
--method Attn \
--data_folder Data/ExperimentsFolds/9 \
--data fakehealth \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data fakehealth \
--alpha 0.99 \
--ratio 0.5


# bert
python3 -m Mean_Teacher.main \
--model PI \
--method Bert \
--data_folder Data/ExperimentsFolds/3 \
--data fakehealth \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data fakehealth \
--alpha 0.99 \
--ratio 0.5

python3 -m Mean_Teacher.main \
--model PI \
--method Bert \
--data_folder Data/ExperimentsFolds/4 \
--data fakehealth \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data fakehealth \
--alpha 0.99 \
--ratio 0.5

python3 -m Mean_Teacher.main \
--model PI \
--method Bert \
--data_folder Data/ExperimentsFolds/6 \
--data fakehealth \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data fakehealth \
--alpha 0.99 \
--ratio 0.5

python3 -m Mean_Teacher.main \
--model PI \
--method Bert \
--data_folder Data/ExperimentsFolds/7 \
--data fakehealth \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data fakehealth \
--alpha 0.99 \
--ratio 0.5

python3 -m Mean_Teacher.main \
--model PI \
--method Bert \
--data_folder Data/ExperimentsFolds/8 \
--data fakehealth \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data fakehealth \
--alpha 0.99 \
--ratio 0.5

python3 -m Mean_Teacher.main \
--model PI \
--method Bert \
--data_folder Data/ExperimentsFolds/9 \
--data fakehealth \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data fakehealth \
--alpha 0.99 \
--ratio 0.5







