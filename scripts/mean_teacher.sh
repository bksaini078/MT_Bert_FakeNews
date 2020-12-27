#make sure by using fake_news_generate the unlabel data for each fold, if already done then move forward
python3 -m main \
--model MT \
--method Attn \
--data_folder Data/ExperimentFolds/3 \
--data fakehealth \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data fakehealth \
--alpha 0.99 \
--ratio 0.5

python3 -m main \
--model MT \
--method Attn \
--data_folder Data/ExperimentFolds/4 \
--data fakehealth \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data fakehealth \
--alpha 0.99 \
--ratio 0.5

python3 -m main \
--model MT \
--method Attn \
--data_folder Data/ExperimentFolds/6 \
--data fakehealth \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data fakehealth \
--alpha 0.99 \
--ratio 0.5

python3 -m main \
--model MT \
--method Attn \
--data_folder Data/ExperimentFolds/7 \
--data fakehealth \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data fakehealth \
--alpha 0.99 \
--ratio 0.5

python3 -m main \
--model MT \
--method Attn \
--data_folder Data/ExperimentFolds/8 \
--data fakehealth \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data fakehealth \
--alpha 0.99 \
--ratio 0.5

python3 -m main \
--model MT \
--method Attn \
--data_folder Data/ExperimentFolds/9 \
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
python3 -m main \
--model MT \
--method Bert \
--data_folder Data/ExperimentFolds/3 \
--data fakehealth \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data fakehealth \
--alpha 0.99 \
--ratio 0.5

python3 -m main \
--model MT \
--method Bert \
--data_folder Data/ExperimentFolds/4 \
--data fakehealth \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data fakehealth \
--alpha 0.99 \
--ratio 0.5

python3 -m main \
--model MT \
--method Bert \
--data_folder Data/ExperimentFolds/6 \
--data fakehealth \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data fakehealth \
--alpha 0.99 \
--ratio 0.5

python3 -m main \
--model MT \
--method Bert \
--data_folder Data/ExperimentFolds/7 \
--data fakehealth \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data fakehealth \
--alpha 0.99 \
--ratio 0.5

python3 -m main \
--model MT \
--method Bert \
--data_folder Data/ExperimentFolds/8 \
--data fakehealth \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data fakehealth \
--alpha 0.99 \
--ratio 0.5

python3 -m main \
--model MT \
--method Bert \
--data_folder Data/ExperimentFolds/9 \
--data fakehealth \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data fakehealth \
--alpha 0.99 \
--ratio 0.5

#####GOSSIPCOP####
echo 'gossipcop'

python3 -m main \
--model MT \
--method Attn \
--data_folder Data/ExperimentFolds/3 \
--data gossipcop \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data gossipcop \
--alpha 0.99 \
--ratio 0.5

python3 -m main \
--model MT \
--method Attn \
--data_folder Data/ExperimentFolds/4 \
--data gossipcop \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data gossipcop \
--alpha 0.99 \
--ratio 0.5

python3 -m main \
--model MT \
--method Attn \
--data_folder Data/ExperimentFolds/6 \
--data gossipcop \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data gossipcop \
--alpha 0.99 \
--ratio 0.5

python3 -m main \
--model MT \
--method Attn \
--data_folder Data/ExperimentFolds/7 \
--data gossipcop \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data gossipcop \
--alpha 0.99 \
--ratio 0.5

python3 -m main \
--model MT \
--method Attn \
--data_folder Data/ExperimentFolds/8 \
--data gossipcop \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data gossipcop \
--alpha 0.99 \
--ratio 0.5

python3 -m main \
--model MT \
--method Attn \
--data_folder Data/ExperimentFolds/9 \
--data gossipcop \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data gossipcop \
--alpha 0.99 \
--ratio 0.5


# bert
python3 -m main \
--model MT \
--method Bert \
--data_folder Data/ExperimentFolds/3 \
--data gossipcop \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data gossipcop \
--alpha 0.99 \
--ratio 0.5

python3 -m main \
--model MT \
--method Bert \
--data_folder Data/ExperimentFolds/4 \
--data gossipcop \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data gossipcop \
--alpha 0.99 \
--ratio 0.5

python3 -m main \
--model MT \
--method Bert \
--data_folder Data/ExperimentFolds/6 \
--data gossipcop \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data gossipcop \
--alpha 0.99 \
--ratio 0.5

python3 -m main \
--model MT \
--method Bert \
--data_folder Data/ExperimentFolds/7 \
--data gossipcop \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data gossipcop \
--alpha 0.99 \
--ratio 0.5

python3 -m main \
--model MT \
--method Bert \
--data_folder Data/ExperimentFolds/8 \
--data gossipcop \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data gossipcop \
--alpha 0.99 \
--ratio 0.5

python3 -m main \
--model MT \
--method Bert \
--data_folder Data/ExperimentFolds/9 \
--data gossipcop \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data gossipcop \
--alpha 0.99 \
--ratio 0.5


#####GOSSIPCOP####
echo 'politifact'

python3 -m main \
--model MT \
--method Attn \
--data_folder Data/ExperimentFolds/3 \
--data politifact \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data politifact \
--alpha 0.99 \
--ratio 0.5

python3 -m main \
--model MT \
--method Attn \
--data_folder Data/ExperimentFolds/4 \
--data politifact \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data politifact \
--alpha 0.99 \
--ratio 0.5

python3 -m main \
--model MT \
--method Attn \
--data_folder Data/ExperimentFolds/6 \
--data politifact \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data politifact \
--alpha 0.99 \
--ratio 0.5

python3 -m main \
--model MT \
--method Attn \
--data_folder Data/ExperimentFolds/7 \
--data politifact \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data politifact \
--alpha 0.99 \
--ratio 0.5

python3 -m main \
--model MT \
--method Attn \
--data_folder Data/ExperimentFolds/8 \
--data politifact \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data politifact \
--alpha 0.99 \
--ratio 0.5

python3 -m main \
--model MT \
--method Attn \
--data_folder Data/ExperimentFolds/9 \
--data politifact \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data politifact \
--alpha 0.99 \
--ratio 0.5


# bert
python3 -m main \
--model MT \
--method Bert \
--data_folder Data/ExperimentFolds/3 \
--data politifact \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data politifact \
--alpha 0.99 \
--ratio 0.5

python3 -m main \
--model MT \
--method Bert \
--data_folder Data/ExperimentFolds/4 \
--data politifact \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data politifact \
--alpha 0.99 \
--ratio 0.5

python3 -m main \
--model MT \
--method Bert \
--data_folder Data/ExperimentFolds/6 \
--data politifact \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data politifact \
--alpha 0.99 \
--ratio 0.5

python3 -m main \
--model MT \
--method Bert \
--data_folder Data/ExperimentFolds/7 \
--data politifact \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data politifact \
--alpha 0.99 \
--ratio 0.5

python3 -m main \
--model MT \
--method Bert \
--data_folder Data/ExperimentFolds/8 \
--data politifact \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data politifact \
--alpha 0.99 \
--ratio 0.5

python3 -m main \
--model MT \
--method Bert \
--data_folder Data/ExperimentFolds/9 \
--data politifact \
--model_output_folder trained_models \
--epochs 3 \
--lr 0.0001 \
--batch_size 1 \
--max_len 512 \
--data politifact \
--alpha 0.99 \
--ratio 0.5



