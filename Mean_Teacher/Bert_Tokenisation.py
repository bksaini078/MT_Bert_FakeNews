import bert
import tensorflow_hub as hub
import numpy as np
import tensorflow as tf
BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",trainable=True)
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)
def bert_tokenization(articles,max_len):
    tokenized_articles=[]
    for article in articles:
        tk_article_temp= np.array(tokenizer.convert_tokens_to_ids(tokenizer.tokenize(article)[:max_len]))
        if len(tk_article_temp) < max_len :
            # padding with zeros in the end if the len of article is less than maxlen
            tk_article_temp= np.hstack((tk_article_temp, np.zeros(max_len-len(tk_article_temp))))
            tokenized_articles.append(tk_article_temp)
        else:
            tokenized_articles.append(tk_article_temp)
    return tf.convert_to_tensor(tokenized_articles),len(tokenizer.vocab),tokenizer