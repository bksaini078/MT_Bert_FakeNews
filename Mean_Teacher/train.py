import tensorflow as tf

tf.compat.v1.enable_eager_execution()
from Mean_Teacher.costfunction import Overall_Cost, EMA
from logger import logger
from src.clf.bert import BERT
from sklearn.metrics import classification_report, roc_curve, auc

CONSISTENCY_LOSS_FN = {
    "mse": tf.keras.losses.MSE,
    "kl_divergence": tf.keras.losses.kl_divergence
}


class MeanTeacher:
    def __init__(self, args):
        self.student = None
        self.teacher = None
        self.args = args
        self.epochs = args.epochs
        self.loss_fn = CONSISTENCY_LOSS_FN[self.args.loss_fn]
        self.model_output_folder = args.model_output_folder
        self.student_model_name = f"{self.model_output_folder}/{self.args.pretrained_model}_{self.args.model}_seed_{self.args.seed}_portion_{self.args.data}_alpha_{self.args.alpha}_{self.args.ratio_label}_student"
        self.student_results_name = f"{self.model_output_folder}/{self.args.pretrained_model}_{self.args.model}_seed_{self.args.seed}_portion_{self.args.data}_{self.args.alpha}_{self.args.ratio_label}_student_results.tsv"
        self.teacher_model_name = f"{self.model_output_folder}/{self.args.pretrained_model}_{self.args.model}_seed_{self.args.seed}_portion_{self.args.data}_{self.args.alpha}_{self.args.ratio_label}_teacher"
        self.teacher_results_name = f"{self.model_output_folder}/{self.args.pretrained_model}_{self.args.model}_seed_{self.args.seed}_portion_{self.args.data}_{self.args.alpha}_{self.args.ratio_label}_teacher_results.tsv"

    def fit(self, X_train, y_train):
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train[0], X_train[1], y_train)).batch(
            self.args.batch_size)

        optimizer = tf.keras.optimizers.Adam(learning_rate=self.args.lr)

        logger.info("Initializing Student Model")
        self.student = BERT(args=self.args, hidden_dropout_prob=self.args.student_hidden_dropout_prob,
                            attention_prob_drop=self.args.student_attention_prob_dropout).create_model(
            training=True)

        logger.info("Initializing Teacher Model")
        self.teacher = BERT(args=self.args, hidden_dropout_prob=self.args.teacher_hidden_dropout_prob,
                            attention_prob_drop=self.args.teacher_attention_prob_dropout).create_model(
            training=False)

        # declaring metrics
        train_metrics = tf.keras.metrics.BinaryAccuracy(name='Binary_Accuracy')
        progbar = tf.keras.utils.Progbar(len(train_dataset), stateful_metrics=['Accuracy', 'Overall_Loss'])

        for epoch in range(self.epochs):
            for step, (inputs, attention, y_batch_train) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    # TODO please lower the function name and instead of using args parameter, use the real params. it would be very confusing to find the bugs if "args" is seen.
                    overall_cost = Overall_Cost(self.args, [inputs, attention], y_batch_train,
                                                self.student, self.teacher, self.loss_fn)

                grads = tape.gradient(overall_cost, self.student.trainable_weights)
                optimizer.apply_gradients(
                    (grad, var) for (grad, var) in zip(grads, self.student.trainable_weights) if grad is not None)

                # applying student weights to teacher
                teacher = EMA(self.student, self.teacher, alpha=self.args.alpha)

                # calculating training accuracy
                logits_t = teacher([inputs, attention])
                train_acc = train_metrics(tf.argmax(y_batch_train, 1), tf.argmax(logits_t, 1))

                # TODO: Overall cost gives always NaN
                progbar.update(step, values=[('Accuracy', train_acc), ('Overall_Loss', overall_cost)])

        self.save_weights()

        tf.keras.backend.clear_session()

    def eval(self, X_test, y_test):
        self.load_weights()

        logger.info("Student Evaluation")
        y_hat = self.student.predict(X_test)
        y_pred = tf.argmax(y_hat, 1)
        y_true = tf.argmax(y_test, 1)

        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=None)
        logger.info(classification_report(y_true=y_true, y_pred=y_pred, digits=4))
        logger.info(f"AUC {auc(fpr, tpr)}")

        logger.info("Teacher Evaluation")
        y_hat = self.teacher.predict(X_test)
        y_pred = tf.argmax(y_hat, 1)
        y_true = tf.argmax(y_test, 1)

        fpr, tpr, thresholds = roc_curve(y_true, y_pred, pos_label=None)
        logger.info(classification_report(y_true=y_true, y_pred=y_pred, digits=4))
        logger.info(f"AUC {auc(fpr, tpr)}")

    def save_weights(self):
        self.student.save_weights(self.student_model_name)
        del self.student
        self.teacher.save_weights(self.teacher_model_name)
        del self.teacher

    def load_weights(self):
        self.student = BERT(args=self.args, hidden_dropout_prob=self.args.student_hidden_dropout_prob,
                            attention_prob_drop=self.args.student_attention_prob_dropout).create_model(
            training=False)
        self.student.load_weights(self.student_model_name)
        self.teacher = BERT(args=self.args, hidden_dropout_prob=self.args.teacher_hidden_dropout_prob,
                            attention_prob_drop=self.args.teacher_attention_prob_dropout).create_model(
            training=False)
        self.teacher.load_weights(self.teacher_model_name)
