python -m main \
--data_folder Data/ExperimentFolds/3 \
--data fakehealth \
--model MT \
--alpha 0.753 \
--batch_size 8 \
--epochs 3 \
--max_len 300 \
--ratio 0.5 \
--pretrained_model distilbert-base-uncased \
--lr 2e-5 \
--loss_fn kl_divergence