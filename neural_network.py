import numpy as np

def sigmoid(x): #activation function
    return 1 / (1 + np.exp(-x))

class SimpleNeuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias
        
    def feedforward(self, inputs): #Using the activation function to calculate the output of the neuron
        total = np.dot(self.weights, inputs) +self.bias
        return sigmoid(total)
    
    
weights = np.array([0, 1]) #w1 = 0, w2 = 1
bias = 4
n = SimpleNeuron(weights, bias)

x = np.array([2, 3]) #x1 = 2, x2 = 3
print(n.feedforward(x)) #Final output of the neuron after applying the activation function
    
class NeuralNetwork:
    def __init__(self):
        weights = np.array([0,1])
        bias = 0
        self.h1 = SimpleNeuron(weights, bias)
        self.h2 = SimpleNeuron(weights, bias)
        self.output = SimpleNeuron(weights, bias)

    def feedforward(self, x):
        output_h1 = self.h1.feedforward(x)
        output_h2 = self.h2.feedforward(x)
        output = self.output.feedforward(np.array([output_h1, output_h2]))
        return output
    
network = NeuralNetwork()
x = np.array([2, 3])
print(network.feedforward(x)) #Final output of the neural network after applying the activation function
    
def mse(y1, y2): # y1 is the actual value, y2 is the predicted value.
    return (np.mean((y1 - y2) ** 2)) #Mean Squared Error

y1 = np.array([0.1, 1, 0, 1])
y2 = np.array([0, 1, 0, 1])
print(mse(y1, y2)) #Final output of the Mean Squared Error
