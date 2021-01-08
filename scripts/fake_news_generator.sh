#python3 -m src.utils.news_generator \
#--data_folder Data/ExperimentFolds/3 \
#--data fakehealth \
#--data_type fake \
#--output_folder Data/fakehealth_3.jsonl \
#--generate
#
#python3 -m src.utils.news_generator \
#--data_folder Data/ExperimentFolds/4 \
#--data fakehealth \
#--data_type fake \
#--output_folder Data/fakehealth_4.jsonl \
#--generate
#
#python3 -m src.utils.news_generator \
#--data_folder Data/ExperimentFolds/5 \
#--data fakehealth \
#--data_type fake \
#--output_folder Data/fakehealth_5.jsonl \
#--generate
#
#python3 -m src.utils.news_generator \
#--data_folder Data/ExperimentFolds/6 \
#--data fakehealth \
#--data_type fake \
#--output_folder Data/fakehealth_6.jsonl \
#--generate
#
#python3 -m src.utils.news_generator \
#--data_folder Data/ExperimentFolds/7 \
#--data fakehealth \
#--data_type fake \
#--output_folder Data/fakehealth_7.jsonl \
#--generate
#
#python3 -m src.utils.news_generator \
#--data_folder Data/ExperimentFolds/8 \
#--data fakehealth \
#--data_type fake \
#--output_folder Data/fakehealth_8.jsonl \
#--generate
#
#python3 -m src.utils.news_generator \
#--data_folder Data/ExperimentFolds/9 \
#--data fakehealth \
#--data_type fake \
#--output_folder Data/fakehealth_9.jsonl \
#--generate

python3 -m src.utils.news_generator \
--output_folder Data/ExperimentFolds/3 \
--data fakehealth \
--data_folder Data/fakehealth_3_processed.jsonl \
--transform

python3 -m src.utils.news_generator \
--output_folder Data/ExperimentFolds/4 \
--data fakehealth \
--data_folder Data/fakehealth_4_processed.jsonl \
--transform

python3 -m src.utils.news_generator \
--output_folder Data/ExperimentFolds/5 \
--data fakehealth \
--data_folder Data/fakehealth_5_processed.jsonl \
--transform

python3 -m src.utils.news_generator \
--output_folder Data/ExperimentFolds/6 \
--data fakehealth \
--data_folder Data/fakehealth_6_processed.jsonl \
--transform

python3 -m src.utils.news_generator \
--output_folder Data/ExperimentFolds/7 \
--data fakehealth \
--data_folder Data/fakehealth_7_processed.jsonl \
--transform

python3 -m src.utils.news_generator \
--output_folder Data/ExperimentFolds/8 \
--data fakehealth \
--data_folder Data/fakehealth_8_processed.jsonl \
--transform

python3 -m src.utils.news_generator \
--output_folder Data/ExperimentFolds/9 \
--data fakehealth \
--data_folder Data/fakehealth_9_processed.jsonl \
--transform