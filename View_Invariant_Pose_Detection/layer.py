import tensorflow as tf
import numpy as np

def conv_layer(input_data,variable,bias):
        #filt = get_conv_filter(name)
        conv = tf.nn.conv2d(input_data,variable, [1, 1, 1, 1], padding='SAME')
        #conv_biases = get_bias(name)

        bias = tf.nn.bias_add(conv, bias)
        relu = tf.nn.relu(bias)
        return relu
def conv_layer3D(input_data,variable,bias):
        #filt = get_conv_filter(name)
        print(variable.get_shape(),bias.get_shape())
        conv = tf.nn.conv3d(input_data,variable, [1, 1, 1, 1,1], padding='SAME')
        #print(conv.get_shape(),'######################')
          #conv_biases = get_bias(name)
        bias = tf.nn.bias_add(conv, bias)
        relu = tf.nn.relu(bias)
        return relu
def avg_pool3D( bottom, name):
        return tf.nn.avg_pool3d(bottom, ksize=[1,1, 2, 2, 1], strides=[1,1, 2, 2, 1], padding='SAME', name=name)

def max_pool3D(bottom, name):
        return tf.nn.max_pool3d(bottom, ksize=[1,1, 2, 2, 1], strides=[1,1, 2, 2, 1], padding='SAME', name=name)

def conv_layer_pad1(input_data,variable,bias):
        #filt = get_conv_filter(name)
        conv = tf.nn.conv2d(input_data,variable, [1, 1, 1, 1], padding='SAME')
        #conv_biases = get_bias(name)
        bias = tf.nn.bias_add(conv, bias)
        relu = tf.nn.relu(bias)
        return relu
def avg_pool( bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
def max_pool(bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

