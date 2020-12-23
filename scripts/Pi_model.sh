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



#####GOSSIPCOP####
echo 'gossipcop'

python3 -m Mean_Teacher.main \
--model PI \
--method Attn \
--data_folder Data/ExperimentsFolds/3 \
--data gossipcop \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data gossipcop \


python3 -m Mean_Teacher.main \
--model PI \
--method Attn \
--data_folder Data/ExperimentsFolds/4 \
--data gossipcop \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data gossipcop \



python3 -m Mean_Teacher.main \
--model PI \
--method Attn \
--data_folder Data/ExperimentsFolds/6 \
--data gossipcop \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data gossipcop \



python3 -m Mean_Teacher.main \
--model PI \
--method Attn \
--data_folder Data/ExperimentsFolds/7 \
--data gossipcop \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data gossipcop \



python3 -m Mean_Teacher.main \
--model PI \
--method Attn \
--data_folder Data/ExperimentsFolds/8 \
--data gossipcop \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data gossipcop \



python3 -m Mean_Teacher.main \
--model PI \
--method Attn \
--data_folder Data/ExperimentsFolds/9 \
--data gossipcop \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data gossipcop \




# bert
python3 -m Mean_Teacher.main \
--model PI \
--method Bert \
--data_folder Data/ExperimentsFolds/3 \
--data gossipcop \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data gossipcop \



python3 -m Mean_Teacher.main \
--model PI \
--method Bert \
--data_folder Data/ExperimentsFolds/4 \
--data gossipcop \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data gossipcop \



python3 -m Mean_Teacher.main \
--model PI \
--method Bert \
--data_folder Data/ExperimentsFolds/6 \
--data gossipcop \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data gossipcop \



python3 -m Mean_Teacher.main \
--model PI \
--method Bert \
--data_folder Data/ExperimentsFolds/7 \
--data gossipcop \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data gossipcop \



python3 -m Mean_Teacher.main \
--model PI \
--method Bert \
--data_folder Data/ExperimentsFolds/8 \
--data gossipcop \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data gossipcop \



python3 -m Mean_Teacher.main \
--model PI \
--method Bert \
--data_folder Data/ExperimentsFolds/9 \
--data gossipcop \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data gossipcop \




#####GOSSIPCOP####
echo 'politifact'

python3 -m Mean_Teacher.main \
--model PI \
--method Attn \
--data_folder Data/ExperimentsFolds/3 \
--data politifact \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data politifact \



python3 -m Mean_Teacher.main \
--model PI \
--method Attn \
--data_folder Data/ExperimentsFolds/4 \
--data politifact \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data politifact \



python3 -m Mean_Teacher.main \
--model PI \
--method Attn \
--data_folder Data/ExperimentsFolds/6 \
--data politifact \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data politifact \



python3 -m Mean_Teacher.main \
--model PI \
--method Attn \
--data_folder Data/ExperimentsFolds/7 \
--data politifact \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data politifact \



python3 -m Mean_Teacher.main \
--model PI \
--method Attn \
--data_folder Data/ExperimentsFolds/8 \
--data politifact \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data politifact \



python3 -m Mean_Teacher.main \
--model PI \
--method Attn \
--data_folder Data/ExperimentsFolds/9 \
--data politifact \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data politifact \




# bert
python3 -m Mean_Teacher.main \
--model PI \
--method Bert \
--data_folder Data/ExperimentsFolds/3 \
--data politifact \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data politifact \



python3 -m Mean_Teacher.main \
--model PI \
--method Bert \
--data_folder Data/ExperimentsFolds/4 \
--data politifact \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data politifact \



python3 -m Mean_Teacher.main \
--model PI \
--method Bert \
--data_folder Data/ExperimentsFolds/6 \
--data politifact \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data politifact \



python3 -m Mean_Teacher.main \
--model PI \
--method Bert \
--data_folder Data/ExperimentsFolds/7 \
--data politifact \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data politifact \



python3 -m Mean_Teacher.main \
--model PI \
--method Bert \
--data_folder Data/ExperimentsFolds/8 \
--data politifact \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data politifact \



python3 -m Mean_Teacher.main \
--model PI \
--method Bert \
--data_folder Data/ExperimentsFolds/9 \
--data politifact \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data politifact \





