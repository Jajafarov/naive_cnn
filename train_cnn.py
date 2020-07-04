###
# Author: Almar Jafarov
# A simple convolutional neural network implementation
# Implemented to learn how CNN's work under the hood
# 02-06-2020
##

import scipy.io as spio
from cupyx.scipy.ndimage import correlate as correlate_image
import numpy as np
import cupy as cp
from itertools import product


def initialise_weights_ur(nn_structure):
    # Uniform random weight initialisation
    weights = []
    weights.append(cp.random.uniform(0, 1, (nn_structure[0].shape)))
    for layer in range(1, len(nn_structure) - 1):
        weights.append(cp.random.uniform(
            0, 1, (nn_structure[layer+1], nn_structure[layer])))
    return weights


def normalise_weights(weights):
    for layer in range(1, len(weights)):
        weights[layer] = weights[layer] / \
            weights[layer].sum(axis=1)[:, cp.newaxis]
    return weights


def initialise_bias(nn_structure):
    bias = []
    bias.append(cp.zeros(int(nn_structure[0].shape[-1])))
    for layer in range(2, len(nn_structure)):
        bias.append(cp.zeros((nn_structure[layer])))
    return bias


def relu(x):
    return cp.maximum(x, 0)


def softmax(x):
    """
    Softmax function for cupy arrays
    Shifts the values by the max of the array
    to avoid nan values.
    """
    shifted = cp.exp(x - cp.max(x))
    res = shifted/cp.sum(shifted)
    return res


def correlate(image, kernel):
    """
    Valid correlation (i.e. no padding) function for use in the convolution layer
    as cupy doesn't implement scipy.signal.correlate
    """
    row_offset = int(kernel.shape[0]/2)
    column_offset = int(kernel.shape[1]/2)
    if kernel.shape[0] % 2 == 0 and kernel.shape[1] % 2 == 0:
        return correlate_image(image, kernel)[row_offset:-row_offset+1, column_offset:-column_offset+1]
    else:
        return correlate_image(image, kernel)[row_offset:-row_offset, column_offset:-column_offset]


def heaviside(x):
    """
    Heaviside function using cupy
    """
    return cp.sign(x)


def maxpool(x):
    """
    Function to perform 2x2 max pooling with stride = 2
    Params:
    x - cupy array - data to do max pooling on, should be 3 dimensional, with the last dimension
    being the number of channels after convolution

    Returns:
    pool_res - cupy array - data after pooling
    maxpool_mask - cupy array - 1D array to specify the index of the
    feature that was propagated from the pooling
    """
    r, c, d = x.shape
    im = im2col(x, 2, 2)
    maxpool_mask = cp.argmax(im, axis=0)
    pool_res = im[maxpool_mask, cp.arange(
        maxpool_mask.size)]
    # .reshape(int(r/2), int(c/2), d)

    return pool_res, maxpool_mask


def im2col(im, k_size, stride):
    """
    Function to convert each block in a multi-channel image
    to a column of a 2D matrix, currently only works with square blocks,
    similar to the MATLAB function with the same name
    Params:
    im - cupy array, the image to convert
    k_size - int, block size
    stride - int

    Returns:
    cols - cupy array containing the columns
    """
    if not im.shape[0] % k_size == 0:
        im = cp.pad(im, (0, 1, 0), 'edge')
    cols = cp.empty((0, int(im.shape[0]*im.shape[1]/(k_size**2)*im.shape[2])))

    for i, j in product(range(k_size), repeat=2):
        cols = cp.concatenate(
            (cols, im[i::stride, j::stride, :].reshape((1, int(im.shape[0]/k_size)**2*im.shape[2]))))

    return cols


def col2im(col, pad, stride, block, x_shape, channels):
    """
    Function to convert a column matrix back to the multi-channel image
    Currently not used
    """
    im = cp.zeros(x_shape)
    for k, (i, j) in enumerate(product(range(block), repeat=2)):
        im[j::stride, i::stride, :] = cp.transpose(
            col[k, :].reshape(block, block, channels), axes=(1, 0, 2))
    if pad:
        return cp.transpose(im, axes=(1, 0, 2))[:-1, :-1, :]
    else:
        return cp.transpose(im, axes=(1, 0, 2))


def train_nn(epochs, batch_size, nn_structure, weights, biases,
             train_samples, train_labels, act_func, eta):
    """
    Trains a convolutional neural network over a given data set
    Params:
    epochs - int - number of epochs to train for
    batch_size - int

    nn_structure - list of ints specifiying how many layers and neurons in each layer,
    the first object has to be an empty numpy array with 3 dimensions,
    first two are filter/kernel size and the last is number of filters

    weights - list of cupy arrays - initial weights including the filter weights
    bias - list of cupy arrays - initial biases including filter biases
    train_samples - cupy array of training data
    train_labels - cupy array of training data labels, one-hot encoded
    act_func - function - relu, the activation function used
    eta - float - learning rate

    Returns:
    weights - list of numpy arrays - final weights of the network
    bias - list of numpy arrays - final biases of the networks
    """

    w_layer_num = len(weights)
    n_samples = train_labels.shape[0]
    error = [0]*epochs
    weight_deltas = [0]*w_layer_num
    bias_deltas = [0]*w_layer_num

    for i in range(epochs):

        # Shuffle samples
        shuffled_train = cp.random.permutation(train_samples.shape[0])

        for batch in range(int(train_samples.shape[0]/batch_size)):

            # Initialise weights
            for layer in range(w_layer_num):
                weight_deltas[layer] = cp.zeros_like(weights[layer])
                bias_deltas[layer] = cp.zeros_like(biases[layer])

            for sample in range(batch_size):

                outputs = []
                input_sample = train_samples[shuffled_train[batch_size *
                                                            batch + sample]].reshape(3, 32, 32).T
                sample_label = train_labels[shuffled_train[batch_size * batch + sample]]

                # Convolve with first three filters
                convolutions = correlate(
                    input_sample, weights[0][:, :, 0:3])

                # Convolution with the rest of the filters
                for kernel in range(3, nn_structure[0].shape[2], 3):
                    convolutions = cp.concatenate((convolutions, correlate(
                        input_sample, weights[0][:, :, kernel:kernel+3])), axis=2)

                # Maxpooling, activation function and flattening the output
                maxpool_res, maxpool_mask = maxpool(
                    convolutions + bias[0])
                maxpool_res = act_func(maxpool_res).flatten()

                # Standard forward propagation
                outputs.append(maxpool_res)
                outputs.append(
                    act_func(cp.dot(weights[1], maxpool_res) + bias[1]))

                # Back prop for final layer
                # used to store previous delta
                previous_delta = softmax(outputs[-1]) - sample_label
                weight_deltas[-1] += cp.outer(previous_delta, outputs[-2])
                bias_deltas[-1] += previous_delta

                previous_delta = heaviside(outputs[0]) * cp.dot(
                    weights[1].T, previous_delta)

                # Back prop for convolutional layer which is just a convolution between
                # previous delta and input layer
                previous_delta[-maxpool_mask] = 0

                for kernel in cp.arange(0, (n_of_kernels - 1) * 3 + 1, 3):
                    weight_deltas[0][:, :, kernel:kernel+3] += correlate(
                        input_sample, previous_delta.reshape(
                            (15, 15, n_of_kernels*3)).repeat(2, axis=0).repeat(2, axis=1)[:, :, kernel:kernel+3])

                bias_deltas[0] += cp.sum(previous_delta.reshape(15,
                                                                15, n_of_kernels*3), axis=(0, 1))

                # Store error
                error[i] -= cp.log(cp.dot(sample_label,
                                          softmax(outputs[-1])))/n_samples

            # Update weights
            for layer in range(w_layer_num):
                weights[layer] -= eta * weight_deltas[layer]/batch_size
                biases[layer] -= eta * bias_deltas[layer]/batch_size

                # Weight and bias clipping to combat exploding gradients
                if cp.linalg.norm(weights[layer]) > 5:
                    weights[layer] = (5 * weights[layer]) / \
                        cp.linalg.norm(weights[layer])
                if cp.linalg.norm(biases[layer]) > 5:
                    biases[layer] = (5 * biases[layer]) / \
                        cp.linalg.norm(biases[layer])

            # pdb.set_trace()

        print("Epoch ", i+1, ": error = ", error[i])

    return weights, bias


def test_nn(nn_structure, weights, bias, test_set, testlabels, act_func):
    """
    Trains a neural network over a given data set
    Params:
    nn_structure - list of ints specifiying how many layers and neurons in each layer,
    the first object has to be an empty numpy array with 3 dimensions,
    first two are filter/kernel size and the last is number of filters

    weights - list of numpy arrays - weights of the network to test
    bias - list of numpy arrays - biases of the network to test
    test_set - numpy array of test data
    testlabels - numpy array of test data labels
    act_func - function - either sigmoid or relu, the activation function used

    Returns:
    proportion of the test data correctly predicted
    """

    predicted_correct = 0
    w_layer_num = len(weights)

    for i in range(0, test_set.shape[0]):
        outputs = []
        input_sample = test_set[i].reshape(3, 32, 32).T
        correct_label = testlabels[i]

        # Colvolve with first 3 filters
        convolutions = correlate(input_sample, weights[0][:, :, 0:3])

        # Convolution with all other filters
        for kernel in range(3, nn_structure[0].shape[2], 3):
            convolutions = cp.concatenate((convolutions, correlate(
                input_sample, weights[0][:, :, kernel:kernel+3])), axis=2)

        # Max pooling and flattening the image
        maxpool_res, _ = maxpool(convolutions + bias[0])
        maxpool_res = act_func(maxpool_res).flatten()

        # Proceed with standard forward propagation
        outputs.append(act_func(
            cp.dot(weights[1], maxpool_res) + bias[1]))

        # Find the neuron with the highest probability output
        predicted_label = cp.argmax(outputs[-1])
        if predicted_label == correct_label:
            predicted_correct += 1

    print(predicted_correct)
    return predicted_correct/testlabels.shape[0]


# Load the training dataset
mat = spio.loadmat('train_data.mat', squeeze_me=True)
train = cp.reshape(cp.array(mat['x_train']), (50000, 3, 1024))  # training data
train_means = cp.mean(train, axis=(0, 2))[:, cp.newaxis]
train_std = cp.std(train, axis=(0, 2))[:, cp.newaxis]

# Normalised training data
train_normalised = ((train - train_means) / train_std).reshape(50000, 3072)
trainlabels = cp.asarray(mat['x_train_labs']-1)  # labels
# one-hot encoded training labels
trainlabels_ohe = cp.eye(10)[cp.asarray(trainlabels)]

mat2 = spio.loadmat('test_data.mat', squeeze_me=True)
test = cp.reshape(cp.array(mat2['x_test']), (10000, 3, 1024))  # test data

# Normalised training data
test_normalised = ((test - train_means) / train_std).reshape(10000, 3072)
testlabels = cp.asarray(mat2['x_test_labs']-1)  # labels


n_of_kernels = 6  # number of filters

kernel_size = 3  # square filter size

# network structure
nn_structure = [cp.empty((kernel_size, kernel_size, n_of_kernels*3)),
                int((((32-kernel_size+1)/2)**2) * n_of_kernels * 3), 10]

# Initiate weights and biases
weights = normalise_weights(initialise_weights_ur(nn_structure))
bias = initialise_bias(nn_structure)


trained_weights, trained_bias = train_nn(10, 1, nn_structure, weights, bias,
                                         train_normalised, trainlabels_ohe, relu, 0.001)

number_correct = test_nn(nn_structure, trained_weights,
                         trained_bias, test_normalised, testlabels, relu)
print(number_correct)
