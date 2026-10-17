import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x)) # Logistic function

def neuron(weights, inputs, bias):
    total = (w1 *x1) + (w2 *x2) + bias #activation function
    return sigmoid(total)

w1 = 0 # weight for input 1
w2 = 1 # weight for input 2
bias = 4 # bias
x1 = 2 # input 1
x2 = 3 # input 2
print(neuron(weights=[w1, w2], inputs=[x1, x2], bias=bias))