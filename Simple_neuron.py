import numpy as np

def sigmoid(x): #activation function
    return 1 / (1 + np.exp(-x))

class SimpleNeuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        
    def forward(self, inputs): #Using the activation function to calculate the output of the neuron
        total = np.dot(self.weights, inputs) +self.bias
        return sigmoid(total)
    
weights = np.array([0, 1]) #w1 = 0, w2 = 1
bias = 4
n = SimpleNeuron(weights, bias)

x = np.array([2, 3]) #x1 = 2, x2 = 3
print(n.forward(x)) #Final output of the neuron after applying the activation function
    
