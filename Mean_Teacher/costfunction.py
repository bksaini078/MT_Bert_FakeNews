# declaring loss function
import tensorflow as tf
import numpy as np
from Mean_Teacher.noise_creator import  instant_noise_bert
# ref:https://github.com/CuriousAI/mean-teacher/tree/master/tensorflow/mean_teacher  updated according to our need .
def Classification_costs(logits, labels) :
    """ Commputing classification cost , after removing labels -1 of unlabelled data and then calculating
    the binary cross entropy .
    """   
    # print(applicable)
    # Change -1s to zeros to make cross-entropy computable
    applicable = tf.not_equal(labels, -1)
    labels = tf.where(applicable, labels, tf.zeros_like(labels))
    logits = tf.where(applicable, logits, tf.zeros_like(logits))    
    # labels = tf.nn.softmax(labels )
    # logits= tf.nn.softmax(logits)
    loss = tf.keras.losses.categorical_crossentropy(labels,logits)
    loss= tf.reduce_sum(loss)/2  # 2 is number of class    
    return loss
def Consistency_Cost(student_output, teacher_output) :
    # student_output = tf.nn.log_softmax(student_output)
    # teacher_output = tf.nn.softmax(teacher_output)
    #doing kl_divergence returns array , thats why reduce_sum
    return tf.reduce_sum(tf.keras.losses.kl_divergence(student_output,teacher_output ))

def Overall_Cost(args,x_train,y_train,x_unlabel_tar, student, teacher):
    x_train_n, y_train_n = instant_noise_bert(x_train, y_train,x_unlabel_tar, args)
    x_train_n1, _ = instant_noise_bert(x_train, y_train, x_unlabel_tar, args)
    logits = student(x_train_n)
    classification_cost = Classification_costs(logits, y_train_n)
    # print('classification cost', classification_cost)
    tar_teacher = teacher(x_train_n1)
    consistency_cost = Consistency_Cost(logits, tar_teacher)
    # print('consistency cost', consistency_cost)
    return (args.ratio * classification_cost) + ((1 - args.ratio) * consistency_cost)


# function for consistency cost
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
