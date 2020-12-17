import numpy as np
import tensorflow as tf
from cleantext import clean
from keras.utils import to_categorical
from tensorflow import keras
from tensorflow.keras import layers
from tqdm import tqdm
from transformers import AutoTokenizer, TFAutoModel
from transformers.modeling_tf_utils import get_initializer

from logger import logger


def clean_helper(text):
    return clean(text,
                 fix_unicode=True,  # fix various unicode errors
                 to_ascii=True,  # transliterate to closest ASCII representation
                 no_urls=True,  # replace all URLs with a special token
                 no_emails=True,
                 lower=True,
                 no_phone_numbers=True,
                 no_numbers=True,  # replace all numbers with a special token
                 no_digits=True,  # replace all digits with a special token
                 no_currency_symbols=True,
                 replace_with_url="<URL>",
                 replace_with_email="<EMAIL>",
                 replace_with_phone_number="<PHONE>",
                 replace_with_number="<NUMBER>",
                 replace_with_digit="<DIGIT>",
                 replace_with_currency_symbol="<CUR>",
                 lang="en")


class NewsExample:
    def __init__(self, title, content, label, max_len, tokenizer):
        self.title = title
        self.content = content
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.label = label
        self.skip = False
        self.label2id = {'fake': 0, 'true': 1}

    def preprocess(self):
        title = self.title
        content = self.content
        tokenizer = self.tokenizer
        max_len = self.max_len
        self.label = self.label2id[self.label]

        # Clean content and title
        content = clean_helper(content)
        title = clean_helper(title)

        # Tokenize title and content by adding SEP
        tokenized_text = tokenizer(title, content, add_special_tokens=True,
                                   max_length=max_len,
                                   padding='max_length',
                                   truncation=True,
                                   return_attention_mask=True,
                                   )
        self.input_ids = tokenized_text['input_ids']
        self.attention_mask = tokenized_text['attention_mask']

    @staticmethod
    def convert_label2id(label):
        return 'fake' if int(label) == 0 else 'true'


def create_news_examples(data, max_len, tokenizer):
    news_exps = []
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        news_exp = NewsExample(title=row["title"], content=row["content"], label=row["label"], max_len=max_len,
                               tokenizer=tokenizer)
        news_exp.preprocess()
        news_exps.append(news_exp)
    return news_exps


def create_inputs_targets(news_exps):
    dataset_dict = {
        "input_ids": [],
        "attention_mask": [],
        "label": []
    }
    for item in news_exps:
        if item.skip == False:
            for key in dataset_dict:
                dataset_dict[key].append(getattr(item, key))
    for key in dataset_dict:
        dataset_dict[key] = np.array(dataset_dict[key])

    x = [
        dataset_dict["input_ids"],
        dataset_dict["attention_mask"],
    ]
    y = to_categorical(dataset_dict["label"])
    return x, y


class BERT:
    def __init__(self, args):
        self.pretrained_model = args.pretrained_model
        self.max_len = args.max_len
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)
        self.dropout = tf.keras.layers.Dropout(args.dropout)
        self.classifier = tf.keras.layers.Dense(2, kernel_initializer=get_initializer(),
                                                name="classifier"
                                                )  # 2 labels classifier
        self.epochs = args.epochs
        self.batch_size = args.batch_size
        self.lr = args.lr
        self.model = None
        self.softmax_layer = tf.keras.layers.Softmax()

    def create_model(self):
        encoder = TFAutoModel.from_pretrained(self.pretrained_model)

        input_ids = layers.Input(shape=(self.max_len,), dtype=tf.int32)
        attention_mask = layers.Input(shape=(self.max_len), dtype=tf.int32)

        outputs = encoder(input_ids, attention_mask=attention_mask)[1]
        pooled_output = self.dropout(outputs, training=True)
        logits = self.classifier(pooled_output)

        model = keras.Model(inputs=(input_ids, attention_mask),
                            outputs=[logits])
        optimizer = keras.optimizers.Adam(lr=self.lr, clipvalue=1)
        model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def train(self, train_data):
        train_data = create_news_examples(train_data, self.max_len, self.tokenizer)
        x_train, y_train = create_inputs_targets(train_data)
        num_gpu = len(tf.config.experimental.list_physical_devices('GPU'))

        if num_gpu > 0:
            logger.info("GPU is found")
            with tf.device('/GPU:0'):
                self.model = self.create_model()
        else:
            logger.info("Training with CPU")
            self.model = self.create_model()

        logger.info("=======Model Summary======")
        logger.info(self.model.summary())

        self.model.fit(x_train, y_train, epochs=self.epochs, verbose=1, batch_size=self.batch_size)

    def predict(self, test_data):
        test_data = create_news_examples(test_data, self.max_len, self.tokenizer)
        x_test, y_test = create_inputs_targets(test_data)
        predictions = self.model.predict(x_test, batch_size=self.batch_size)

        probs = self.softmax_layer(predictions)
        labels = np.argmax(probs, axis=1)
        labels = [NewsExample.convert_label2id(label.item()) for label in labels]

        return {'probs': probs,
                'labels': labels}

    def save_weights(self, fname):
        self.model.save_weights(fname)

    def load_weights(self, fname):
        num_gpu = len(tf.config.experimental.list_physical_devices('GPU'))
        if num_gpu > 0:
            logger.info("GPU is found")
            with tf.device('/GPU:0'):
                self.model = self.create_model()
        else:
            logger.info("Training with CPU")
            self.model = self.create_model()
        self.model.load_weights(fname)
