# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 11:42:10 2019

@author: askin
"""

import tensorflow as tf
tf.compat.v1.enable_eager_execution()
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense


is_correct_tf_version = '1.14.' in tf.__version__
assert is_correct_tf_version, "Wring tensorflow version {} installed".format(tf.__version__)

is_eager_enabled = tf.executing_eagerly()
assert is_eager_enabled, "Tensorflow eager mode is not enabled"

# First example, basic graph
"""
a = tf.constant(15, name="a")
b = tf.constant(61, name="b")


def graph(a,b):
    c = tf.add(a,b, name="c")
    d = tf.subtract(b,1, name="d")
    e = tf.multiply(c,d, name="e")
    
    return e

a, b = 1.5, 2.5
e_out = graph(a,b)
print(e_out)
"""


# Second exmaple, perceptron manually
"""
# n_in: number of inputs
# n_out: number of outputs
def our_dense_layer(x, n_in, n_out):
    # Define and initialize parameters, a weight matrix W and biases b
    W = tf.Variable(tf.ones((n_in, n_out)))
    b = tf.Variable(tf.zeros((1, n_out)))
    
    z = tf.add(tf.matmul(x, W), b, name="z")
    
    out = tf.sigmoid(z, name="out")
    
    return out

x_input = tf.constant([1,2.], shape=(1,2), name="x_input")
print(our_dense_layer(x_input, n_in=2, n_out=3))
"""


# API accelarated processing...
"""
# Define the number of inputs and outputs
n_input_nodes = 2
n_output_nodes = 3

# First define the model
model = Sequential()

dense_layer = Dense(n_output_nodes, input_shape=(n_input_nodes,), activation='sigmoid')

model.add(dense_layer)

x_input = tf.constant([[1,2.]], shape=(1,2))
print(model(x_input))
"""


# Automatic differentiation - Finding minimum of (x-1) ** 2
"""
x = tf.Variable([tf.random.normal([1])])
print("Initilizing x={}".format(x.numpy()))
learning_rate = 1e-2
history = []

for i in range(500):
    with tf.GradientTape() as tape:
        y = (x - 1) ** 2 #record the forward pass on the tape
        
    grad = tape.gradient(y,x) # computere the gradient of y wrt x
    new_x = x - learning_rate*grad # sgd updated
    x.assign(new_x) # update the value of x
    history.append(x.numpy()[0])
    
plt.plot(history)
plt.plot([0,500],[1,1])
plt.legend(('Predicted', 'True'))
plt.xlabel('Iteration')
plt.ylabel('x value')
plt.show()
"""

# Exmaple of eager control flow use...
"""
a = tf.constant(12)
counter = 0
while not tf.equal(a,1):
    if tf.equal(a % 2, 0):
        a = a / 2
    else:
        a = 3 * a + 1
    print(a)
"""

