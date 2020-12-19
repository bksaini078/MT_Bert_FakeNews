# declaring loss function
import tensorflow as tf
import numpy as np
# ref:https://github.com/CuriousAI/mean-teacher/tree/master/tensorflow/mean_teacher  updated according to our need .
def Classification_costs(logits, labels) :
    """ Commputing classification cost , after removing labels -1 of unlabelled data and then calculating
    the binary cross entropy .
    """
    # logits=tf.argmax(logits,1)
    # print(logits)
    # labels=tf.argmax(labels,1)
    # print(np.shape(labels),np.shape( logits))
    applicable = tf.not_equal(labels, -1)

    # Change -1s to zeros to make cross-entropy computable
    labels = tf.where(applicable, labels, tf.zeros_like(labels))

    # This will now have incorrect values for unlabeled examples
    # per_sample = tf.keras.losses.binary_crossentropy( labels, logits )
    # print(np.shape(labels),np.shape( logits))
    per_sample= tf.keras.losses.categorical_crossentropy(labels, logits)
    # Retain costs only for labeled
    # per_sample = tf.where(applicable, per_sample, tf.zeros_like(per_sample))
    per_sample=tf.where(applicable[:,1], per_sample, tf.zeros_like(per_sample))
    # Take mean over all examples, not just labeled examples.

    loss = tf.math.divide(tf.reduce_mean(tf.reduce_sum( per_sample ) ), np.shape ( per_sample )[0] )
    
    return loss


# custom loss function
def Overall_Cost(classification_cost, consistency_cost, ratio=0.5) :
    return (ratio * classification_cost) + ((1 - ratio) * consistency_cost)


# function for consistency cost
def Consistency_Cost(teacher_output, student_output) :
    return tf.losses.mean_squared_error( teacher_output, student_output )


def EMA(student_model, teacher_model, alpha) :

    # taking weights
    student_weights = student_model.get_weights ()
    teacher_weights = teacher_model.get_weights ()

    # length must be equal otherwise it will not work
    assert len ( student_weights ) == len (teacher_weights ), 'length of student and teachers weights are not equal Please check. \n Student: {}, \n Teacher:{}'.format (
        len ( student_weights ), len ( teacher_weights ) )

    new_layers = []
    for i, layers in enumerate ( student_weights ) :
        new_layer = alpha * (teacher_weights[i]) + (1 - alpha) * layers
        new_layers.append ( new_layer )
    teacher_model.set_weights ( new_layers )
    return teacher_model
