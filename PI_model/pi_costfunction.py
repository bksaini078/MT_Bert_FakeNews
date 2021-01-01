import tensorflow as tf
import math
import numpy as np

def pi_model_loss(args,X_train_labeled, y_train_labeled, X_train_unlabeled,pi_model, unsupervised_weight):
    """ Gets the Loss Value for SSL Pi Model
    Arguments:
        X_train_labeled {tensor} -- train images
        y_train_labeled {tensor} -- train labels
        X_train_unlabeled {tensor} -- unlabeled train images
        pi_model {tf.keras.Model} -- model to be trained
        unsupervised_weight {float} -- weight
    Returns:
        {tensor} -- loss value
    """
    z_labeled = pi_model(X_train_labeled)
    z_labeled_i = pi_model(X_train_labeled)
    # print('unlabel_size:', np.shape(X_train_unlabeled))
    p = np.random.permutation(args.batch_size)
    X_train_unlabeled[0]= X_train_unlabeled[0][p]
    X_train_unlabeled[1] = X_train_unlabeled[1][p]
    X_train_unlabeled[2] = X_train_unlabeled[2][p]

    z_unlabeled = pi_model(X_train_unlabeled)
    z_unlabeled_i = pi_model(X_train_unlabeled)

    # Loss = supervised loss + unsup loss of labeled sample + unsup loss unlabeled sample
    return tf.compat.v1.losses.softmax_cross_entropy(
        y_train_labeled, z_labeled) + unsupervised_weight * (
            tf.losses.mean_squared_error(z_labeled, z_labeled_i) +
            tf.losses.mean_squared_error(z_unlabeled, z_unlabeled_i))


def pi_model_gradients(X_train_labeled, y_train_labeled, X_train_unlabeled,
                       pi_model, unsupervised_weight):
    """ Returns the loss and the gradients for eager Pi Model
    Arguments:
        X_train_labeled {tensor} -- train images
        y_train_labeled {tensor} -- train labels
        X_train_unlabeled {tensor} -- unlabeled train images
        pi_model {tf.keras.Model} -- model to be trained
        unsupervised_weight {float} -- weight
    Returns:
        {tensor} -- loss value
        {tensor} -- gradients for each model variables
    """
    with tf.GradientTape() as tape:
        loss_value = pi_model_loss(X_train_labeled, y_train_labeled, X_train_unlabeled,
                                   pi_model, unsupervised_weight)
    return loss_value, tape.gradient(loss_value, pi_model.trainable_weights)


def ramp_up_function(epoch, epoch_with_max_rampup=2):
    """ Ramps the value of the weight and learning rate according to the epoch
        according to the paper
    Arguments:
        {int} epoch
        {int} epoch where the rampup function gets its maximum value
    Returns:
        {float} -- rampup value
    """

    if epoch < epoch_with_max_rampup:
        p = max(0.0, float(epoch)) / float(epoch_with_max_rampup)
        p = 1.0 - p
        print ( 'p', p )
        print('ramp_up_function:',math.exp(-p*p*5.0) )
        return math.exp(-p*p*5.0)
    else:
        return 1.0


def ramp_down_function(epoch, num_epochs):
    """    Arguments:
        {int} current epoch
        {int} total epochs to train
    Returns:
        {float} -- rampup value
    """
    epoch_with_max_rampdown =1


    if epoch >= (num_epochs - epoch_with_max_rampdown):
        ep = (epoch - (num_epochs - epoch_with_max_rampdown)) * 0.2
        # print('ep',ep)
        print ( 'ramp_down_function:', math.exp ( -(ep * ep) / epoch_with_max_rampdown ) )
        return math.exp(-(ep * ep) / epoch_with_max_rampdown)
    else:
        return 1.0
if __name__=='__main__':
    # calculate steps
    training_size= 100
    batch_size = 1
    steps= training_size/batch_size
    epochs=5
    max_supervised_weight= 100 * 2000*1000
    for num_epochs in range(epochs):
        print(ramp_down_function(epochs,num_epochs))


