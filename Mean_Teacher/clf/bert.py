import numpy as np
from cleantext import clean
from keras.utils import to_categorical
from tqdm import tqdm
from transformers import AutoTokenizer


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
    def __init__(self, title, content, max_len, tokenizer):
        self.title = title
        self.content = content
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.skip = False

    def preprocess(self):
        title = self.title
        content = self.content
        tokenizer = self.tokenizer
        max_len = self.max_len

        # Clean context, answer and question
        # TODO

        # Tokenize title
        tokenized_text = tokenizer(title, content, add_special_tokens=True,
                                   max_length=max_len,
                                   padding='max_length',
                                   truncation=True,
                                   return_attention_mask=True,
                                   )
        self.input_ids = tokenized_text['input_ids']
        self.attention_mask = tokenized_text['attention_mask']


def create_news_examples(data, max_len, tokenizer):
    news_exps = []
    for idx, row in tqdm(data.iterrows(), total=len(data)):
        news_exp = NewsExample(title=row["title"], content=row["content"], max_len=max_len, tokenizer=tokenizer)
        news_exp.preprocess()
        news_exps.append(news_exp)
    return news_exps


def create_inputs_targets(news_exps):
    dataset_dict = {
        "input_ids": [],
        "attention_mask": [],
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
    y = to_categorical(dataset_dict["labels"])
    return x, y


class BERT:
    def __init__(self, args):
        self.pretrained_model = args.pretrained_model
        self.max_len = args.max_len
        self.tokenizer = AutoTokenizer.from_pretrained(self.pretrained_model)

    def train(self, train_data):
        train_data = create_news_examples(train_data, self.max_len, self.tokenizer)
        x_train, y_train = create_inputs_targets(train_data)
        print(f"{len(x_train)} training points created.")
