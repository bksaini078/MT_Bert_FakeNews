# declaring loss function
import tensorflow as tf
import numpy as np
from src.noise_creator import  augment_data
# ref:https://github.com/CuriousAI/mean-teacher/tree/master/tensorflow/mean_teacher  updated according to our need .
def Classification_costs(logits, labels) :
    """ Commputing classification cost , after removing labels -1 of unlabelled data and then calculating
    the binary cross entropy .
    """
    applicable = tf.not_equal(labels, -1)
    # Change -1s to zeros to make cross-entropy computable
    labels = tf.where(applicable, labels, tf.zeros_like(labels))
    per_sample= tf.keras.losses.categorical_crossentropy(labels, logits)
    # Retain costs only for labeled
    per_sample=tf.where(applicable[:,1], per_sample, tf.zeros_like(per_sample))
    # Take mean over all examples, not just labeled examples.
    loss = tf.math.divide(tf.reduce_mean(tf.reduce_sum( per_sample ) ), np.shape ( per_sample )[0] )
    
    return loss

def Overall_Cost(args,x_train,y_train,x_unlabel_tar, student, teacher):

    x_train_n, y_train_n = augment_data(x_train, y_train, x_unlabel_tar, args)
    x_train_n1, _ = augment_data(x_train, y_train, x_unlabel_tar, args)
    logits = student(x_train_n)
    classification_cost = Classification_costs(logits, y_train)
    tar_student = student(x_train_n1)
    tar_teacher = teacher(x_train_n1)
    consistency_cost = Consistency_Cost( tar_student, tar_teacher)

    return (args.loss_weight * classification_cost) + ((1 - args.loss_weight) * consistency_cost)


# function for consistency cost
def Consistency_Cost(student_output, teacher_output) :
    return tf.losses.mean_squared_error(teacher_output, student_output )


def EMA(student_model, teacher_model, alpha) :
    # taking weights
    student_weights = student_model.get_weights()
    teacher_weights = teacher_model.get_weights()
    # length must be equal otherwise it will not work
    assert len(student_weights ) == len(teacher_weights ), 'length of student and teachers weights are not equal Please check. \n Student: {}, \n Teacher:{}'.format (
        len(student_weights ), len (teacher_weights ) )
    new_layers = []
    for i, layers in enumerate ( student_weights ) :
        new_layer = alpha * (teacher_weights[i]) + (1 - alpha) * layers
        new_layers.append(new_layer)
    teacher_model.set_weights(new_layers)
    return teacher_model
