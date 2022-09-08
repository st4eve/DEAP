import numpy as np
import tensorflow as tf

def bisect_min(array, val):
    """
    Given a sorted array, return the index of an existing element
    that has the closest absolute distance to val.
    """
    array = array
    index = np.searchsorted(array, val, side='left')

    if index == array.size:
        return array.size - 1
    elif index == 0:
        return 0

    if index == array.size - 1:
        offsetArray = tf.constant([
            array[index - 1],
            array[index]], dtype=tf.float32)
    else:
        offsetArray = tf.constant([
            array[index - 1],
            array[index],
            array[index + 1]],dtype=tf.float32)

    return tf.argmin(tf.abs(val - offsetArray)) - 1 + index


def getOutputShape(inputShape, kernelShape, padding=0, stride=1):
    """
    Given an input shape, kernel shape, padding and stride, return the
    dimensions of the convoled image. Will throw an error if the
    dimensionalities do not match.

    The inputShape can have 2 or 3 dimensions.
    The kernelShape can 3 or 4 dimensions

    """
    assert kernelShape[0] == kernelShape[1]

    filterSize = kernelShape[0]
    outputWidth = (inputShape[1] - filterSize + 2 * padding) / stride + 1
    outputHeight = (inputShape[0] - filterSize + 2 * padding) / stride + 1
    outputWidth = int(outputWidth)
    outputHeight = int(outputHeight)

    if len(kernelShape) == 4:
        outputDepth = kernelShape[3]
    else:
        outputDepth = 1

    return (outputHeight, outputWidth, outputDepth)
