# declaring loss function
import tensorflow as tf

from Mean_Teacher.augmentation import augment_data


# ref:https://github.com/CuriousAI/mean-teacher/tree/master/tensorflow/mean_teacher  updated according to our need .
# @Bhupender method names should be always lowerized, see: https://www.python.org/dev/peps/
def compute_clf_loss(logits, labels):
    """ Commputing classification cost , after removing labels -1 of unlabelled data and then calculating
    the binary cross entropy .
    """

    # Change -1s to zeros to make cross-entropy computable
    applicable = tf.not_equal(labels, -1)

    labels = tf.where(applicable, labels, tf.zeros_like(labels))
    logits = tf.where(applicable, logits, tf.zeros_like(logits))
    loss = tf.keras.losses.categorical_crossentropy(labels, logits)

    # mask Nan values, makes them zero
    loss = tf.where(tf.math.is_nan(loss), tf.zeros_like(loss), loss)

    # compute the loss only from labeled samples
    mask = tf.not_equal(loss, 0.)
    masked = tf.boolean_mask(loss, mask)
    loss = tf.reduce_mean(masked)

    return loss


def compute_consistency_loss(student_output, teacher_output, loss_fn):
    return tf.reduce_mean(loss_fn(student_output, teacher_output))


def Overall_Cost(args, x_train, y_train,student, teacher, loss_fn):
    '''Calculating overall cost using classification cost and consistency cost'''
    # including noise data in train data

    #TODO when I checked the code, it seems that you are always using same data distribution, you are not in fact different data distribution
    # x_train_n, y_train_n = augment_data(x_train, y_train, x_unlabel_tar, args)

    # student prediction
    logit_student = student(x_train)

    # including different noise data in train data
    # x_train_n1, _ = augment_data(x_train, y_train, x_unlabel_tar, args)

    # calculating classification cost
    classification_cost = compute_clf_loss(logit_student, y_train)

    # teacher prediction
    logit_teacher = teacher(x_train)

    # calculating consistency cost
    consistency_cost = compute_consistency_loss(logit_student, logit_teacher, loss_fn)

    return (args.ratio * classification_cost) + ((1 - args.ratio) * consistency_cost)


# function for consistency cost
def EMA(student_model, teacher_model, alpha):
    # alpha = min(1 - 1 / (global_step + 1), alpha)
    # taking weights
    student_weights = student_model.get_weights()
    teacher_weights = teacher_model.get_weights()

    # length must be equal otherwise it will not work
    assert len(student_weights) == len(
        teacher_weights), 'length of student and teachers weights are not equal Please check. \n Student: {}, \n Teacher:{}'.format(
        len(student_weights), len(teacher_weights))
    new_layers = []
    for i, layers in enumerate(student_weights):
        new_layer = alpha * (teacher_weights[i]) + (1 - alpha) * layers
        new_layers.append(new_layer)
    teacher_model.set_weights(new_layers)
    return teacher_model
