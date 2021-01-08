import tensorflow as tf

from keras import optimizers
from tensorflow.python.keras.engine import data_adapter
from src.clf.bert import BERT, create_news_examples, create_inputs_targets
import numpy as np
from logger import logger


class MTBERTModel(tf.keras.Model):
    def __init__(self, teacher, student, args, noise_data=None):
        super(MTBERTModel, self).__init__()
        self.teacher = teacher
        self.student = student
        self.loss_weight = args.loss_weight
        self.decay = args.decay
        self.noise_batch = args.noise_batch
        self.noisy_train = noise_data
        self.softmax = tf.keras.layers.Softmax()
        self.loss_type = args.loss_type
        self.split_ratio = args.split_ratio


    def compute_consistency_cost(self, logits_teacher, logits_student):
        if self.loss_type == 'kl_divergence':
            return self.kl_divergence(logits_teacher, logits_student)
        else:
            return self.mean_squared(logits_teacher, logits_student)

    def mean_squared(self, logits_teacher, logits_student):
        return tf.losses.mean_squared_error(logits_teacher, logits_student)

    def kl_divergence(self, logits_teacher, logits_student):
        return tf.losses.kl_divergence(logits_teacher, logits_student)

    def classification_costs(self, logits, labels):
        """Compute classification cost mean and classification cost per sample
        Assume unlabeled examples have label == -1. For unlabeled examples, cost == 0.
        Compute the mean over all examples.
        Note that unlabeled examples are treated differently in error calculation.
        """
        # labels_1D = tf.gather(labels, [0], axis=1)        
        applicable = tf.not_equal(labels, -1)
        labels = tf.where(applicable, labels, tf.zeros_like(labels))
        loss = tf.losses.categorical_crossentropy(logits, labels)
        per_sample = tf.where(applicable[:,1], loss, tf.zeros_like(loss))     

        return tf.reduce_mean(per_sample)

    def compute_overall_cost(self, student_classification_cost, consistency_cost):
        return (self.loss_weight * student_classification_cost) + ((1 - self.loss_weight) * consistency_cost)

    @tf.function()
    def train_step(self, input_data):
        input_data = data_adapter.expand_1d(input_data)
        input_ids, attentions, labels = tf.keras.utils.unpack_x_y_sample_weight(input_data)
        # input_ids, attentions, labels = input_data       

        with tf.GradientTape() as tape:
            augmented_data, augmented_labels = augment_data(input_ids, attentions, labels, self.noisy_train,
                                                            self.noise_batch)
            logits_student = self.student(augmented_data)            
            clf_loss = self.classification_costs(logits_student, augmented_labels)

            # we augment the data again
            augmented_data, augmented_labels = augment_data(input_ids, attentions, labels, self.noisy_train,self.noise_batch)
            # logits_student = self.student(augmented_data)
            logits_teacher = self.teacher(augmented_data)

            consistency_cost = self.compute_consistency_cost(logits_teacher, logits_student)
            loss = self.compute_overall_cost(clf_loss, consistency_cost)

        grads = tape.gradient(loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.student.trainable_variables))

        self.ema(self.student, self.teacher)
        logits_teacher = self.teacher([input_ids, attentions])

        self.compiled_metrics.update_state(labels, logits_teacher)
        metrics = {m.name: m.result() for m in self.metrics}
        metrics["loss"] = clf_loss
        return metrics

    def ema(self, from_model, to_model):
        """Exponential Moving Average of model's weights were stored as ema_model.
        reference: https://github.com/ryancheunggit/tensorflow2_model_zoo/blob/master/cifar10_cnn_ict.py
        """
        for from_val, to_val in zip(from_model.variables, to_model.variables):
            to_val.assign(self.decay * to_val + (1 - self.decay) * from_val)


def decompose_input(data):
    input_ids, attention_mask = data
    return input_ids, attention_mask

def unison_shuffled(x1,x2,y):
    indices = tf.range(start=0, limit=tf.shape(x1)[0], dtype=tf.int32)
    shuffled_indices = tf.random.shuffle(indices)
    x1 = tf.gather(x1, shuffled_indices)
    x2 = tf.gather(x2, shuffled_indices)
    y = tf.gather(y, shuffled_indices)   
    return [x1,x2],y


def augment_data(labeled_input_ids, labeled_attention_mask, y_train, x_unlabel, noise_batch):
    unlabeled_input_ids, unlabeled_attention_mask = decompose_input(x_unlabel)    
    len_unlabeled_samples = len(unlabeled_input_ids)
    p = np.random.permutation(len_unlabeled_samples)[:noise_batch]
    y_train_n = np.full((len(p), 2), -1)
    augmented_labels = tf.concat((y_train, y_train_n), axis=0)
    augmented_input_ids = tf.concat((labeled_input_ids, unlabeled_input_ids[p]), axis=0)
    augmented_attention_masks = tf.concat((labeled_attention_mask, unlabeled_attention_mask[p]), axis=0)   
    return unison_shuffled(augmented_input_ids, augmented_attention_masks, augmented_labels)


class MTBERT(BERT):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.student = None
        self.teacher = None
        self.model = None
        self.args = args


    def generator_batch(self,x_train,y_train,x_unlabel, y_unlabel) :
        unlabel_n = round( self.unlabel_ratio * self.batch_size )
        ini_unlabel_size=0
        label_n = self.batch_size - unlabel_n
        ini_label_size= 0
        while True :
            # ratio * batchsize            
            # print(y_unlabel)
            # print(x_unlabel[1][ini_unlabel_size:unlabel_n])
            ip_un=x_unlabel[0][ini_unlabel_size:unlabel_n]
            attn_un= x_unlabel[1][ini_unlabel_size:unlabel_n]
            label_un=y_unlabel[ini_unlabel_size:unlabel_n]       
            ini_unlabel_size= unlabel_n      
            unlabel_n+=unlabel_n
            if label_n<len(x_train[0]):
                # ip=x_train
                ip, attn, label= x_train[0][ini_label_size:label_n], x_train[1][ini_label_size:label_n], y_train[ini_label_size:label_n]
                ini_label_size=label_n
                label_n+=label_n
            # when labelled data is less then after one circle it will again send same data
            else:
                ini_label_size=0
                label_n = self.batch_size - round( self.unlabel_ratio * self.batch_size )
                ip, attn, label = x_train[0][ini_label_size:label_n], x_train[1][ini_label_size:label_n], y_train[ini_label_size:label_n]
                ini_label_size=label_n
                label_n+=label_n
            print(np.shape(ip_un),np.shape(ip))
            train_input_ids=tf.concat((ip, ip_un), axis=0)
            train_attention_mask = tf.concat((attn, attn_un), axis=0)
            train_labels = tf.concat((label, label_un), axis=0)
            yield((train_input_ids,train_attention_mask,train_labels))

    def train(self, train_data, noise_data):
        train_data = create_news_examples(train_data, self.max_len, self.tokenizer)
        x_train, y_train = create_inputs_targets(train_data)
        # print(y_train)

        #y_unlabel=-1 not real label and will not be considered in cost calculation
        # unlabel_train_data = create_news_examples(unlabel_train_data, self.max_len, self.tokenizer )
        # x_unlabel, y_unlabel = create_inputs_targets( unlabel_train_data)
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train[0], x_train[1], y_train))
        train_dataset = train_dataset.batch(self.batch_size)
        noise_data = create_news_examples(noise_data, self.max_len, self.tokenizer)
        x_noise_train, _ = create_inputs_targets(noise_data)

        num_gpu = len(tf.config.experimental.list_physical_devices('GPU'))
        if num_gpu > 0:
            logger.info("GPU is found")
            with tf.device('/GPU:0'):
                self.teacher = self.create_model(training=False)
                self.student = self.create_model(training=True)
                self.model = MTBERTModel(self.teacher, self.student, *self.args, x_noise_train)

                optimizer = optimizers.Adam(lr=self.lr)
                self.model.compile(optimizer=optimizer, metrics=['accuracy'])
                self.model.fit(train_dataset, epochs=self.epochs, verbose=1, batch_size=self.batch_size)
                # self.model.fit(self.generator_batch(x_train, y_train, x_unlabel, y_unlabel),epochs=self.epochs,verbose=0, batch_size=self.batch_size )
        else:
            logger.info("Training with CPU")
            self.teacher = self.create_model(training=False)
            self.student = self.create_model(training=True)
            self.model = MTBERTModel(self.teacher, self.student, *self.args, x_noise_train)

            optimizer = optimizers.Adam(lr=self.lr)
            self.model.compile(optimizer=optimizer, metrics=['accuracy'])
            # self.model.fit(x_train, y_train, epochs=self.epochs, verbose=0, batch_size=self.batch_size)
            self.model.fit(self.generator_batch(x_train, y_train, x_unlabel, y_unlabel),epochs=self.epochs,verbose=0, batch_size=self.batch_size )

    def save_weights(self, fname):
        self.model.student.save_weights(f"{fname}_student")
        del self.model.student
        self.model.teacher.save_weights(f"{fname}_teacher")
        del self.model.teacher
        del self.model

    def load_weights(self, fname):
        self.model = self.create_model(training=False)
        self.model.load_weights(f"{fname}_{self.model_option}")
