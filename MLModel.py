import sys

import numpy as np
import math
import random as rn


class ActorDimensions:
    def __init__(self, eye, layerOne, layerTwo, output):
        self.dimensions = [eye, layerOne, layerTwo, output]


class NeuralNetwork:
    def __init__(self):
        self.score = 0
        self.layers = []
        self.weights = []
        self.output = 0
        self.activation = 'RELU'
        self.training = False
        self.mutationRate = 0.001

    # Adds a new layer to the network
    # This will automatically create an array of weights of the correct size
    def add_layer(self, size):
        self.layers.append(np.zeros((1, size)))
        if len(self.layers) > 1:
            self.weights.append(np.zeros((len(self.layers[-2][0, :]), len(self.layers[-1][0, :]))))
    # Initializes the network with random weights for each connection
    # This process in done for each layer, at a chosen magnitude
    def initialize_random_weights(self, maxWeight):
        for layer in self.weights:
            for i in range(len(layer[:, 0])):
                for j in range(len(layer[0, :])):
                    layer[i, j] = (rn.random() - 0.5) / maxWeight

    # Spits out the parameters of the Neural Network
    # The first bit denotes the number of layers, N
    # The following 2N bits are the sizes of these layers
    # Following this, all the connection weights are outputted
    # WARNING:: THE SPIT_MODEL METHOD IS DEPRECATED
    def spit_model(self):
        model = []
        # Tabulates number of Layers
        model.append(len(self.layers))
        for layer in self.layers:
            # Records size of each layer
            model.append(len(layer))
        # Records the weight of each connecting layer
        for layer in self.weights:
            for row in layer:
                for weight in row:
                    model.append(weight)
        return model

    # Creates a Neural Network following parameters created by spit_model
    # Any previous parameters and layers held by the network will be deleted
    # WARNING:: THE EAT_MODEL METHOD IS DEPRECATED
    def eat_model(self, model):
        self.layers = []
        self.weights = []
        head = 1
        for i in range(int(model[0])):
            self.add_layer(int(model[head]))
            head += 1
        for layer in self.weights:
            for i in range(len(layer[:, 0])):
                for j in range(len(layer[0, :])):
                    layer[i, j] = float(model[head])
                    head += 1

    # The Activation function of the model
    def activation_function(self, value):
        return max(0, value)

    def activation_derivative(self, layer):
        layerDerivative = np.zeros((1, len(layer[0, :])))
        for i in range(len(layer[0, :])):
            layerDerivative[0, i] = 1
            # if layer[0, i] != 0:
            #     layerDerivative[0, i] = 1
        return layerDerivative


    # Takes in input and provides output
    # The input must match the size of the first layer of the model
    # Calculation is done through simple matrix multiplication, followed by activation for each layer
    def get_output(self, inputArray):
        # Check to see if input array is appropriate size
        if len(self.layers[0]) != len(inputArray):
            print('Error: Incorrect Size of Input for Model')
            sys.exit()
        self.layers[0] = inputArray
        # For each connecting layer multiply that layer by the previous layer
        # This gives the values for the next layer in the model
        # This new layer is the subject to the activation function
        for i in range(0, len(self.weights)):
            self.layers[i + 1] = np.matmul(self.layers[i], self.weights[i])
            for j in range(len(self.layers[i + 1][0, :])):
                self.layers[i + 1][0, j] = self.activation_function(self.layers[i + 1][0, j])
        return self.layers[-1]

    def backpropagation(self, input_array, output_array):
        self.get_output(input_array)
        layerError = [output_array - self.layers[-1]]
        # layerDelta = [layerError[0] * self.activation_derivative(self.layers[-1])]
        layerDelta = [layerError[0]]
        for i in range(len(self.weights)):
            layerError.append(np.dot(layerDelta[i], self.weights[-(i + 1)].T))
            layerDelta.append(layerError[i + 1] * self.activation_derivative(self.layers[-(i + 2)]))
        for i in range(len(self.weights)):
            self.weights[-(i + 1)] += np.dot(self.layers[-(i+2)].T, layerDelta[i]) * self.mutationRate

    # The error function used in backpropagation
    # Currently the model is expected to have a single output value so a simple
    # Square of distance is used
    def error_function(self, output_array):
        return math.pow((output_array[0] - self.layers[-1][0]), 2)


# The actor class is a deprecated version of the Neural_Network class
class Actor:
    def __init__(self, dimensions):
        self.score = 0
        self.inputWeights = np.zeros((dimensions[1], dimensions[0]))
        self.layerOne = np.zeros(dimensions[1])
        self.oneWeights = np.zeros((dimensions[2], dimensions[1]))
        self.layerTwo = np.zeros(dimensions[2])
        self.twoWeights = np.zeros((dimensions[3], dimensions[2]))
        self.output = np.zeros(dimensions[3])
        self.nodeCount = dimensions[0] * dimensions[1] + dimensions[1] * dimensions[2] + dimensions[2] * dimensions[3]

    def initialize_weights(self, parent, maxChange):
        # maxChange = math.sqrt(math.sqrt(self.nodeCount) * 2)
        for i in range(len(self.inputWeights[:, 0])):
            for j in range(len(self.inputWeights[0, :])):
                self.inputWeights[i, j] = parent.inputWeights[i, j] + (rn.random() - 0.5) / maxChange
        for i in range(len(self.oneWeights[:, 0])):
            for j in range(len(self.oneWeights[0, :])):
                self.oneWeights[i, j] = parent.oneWeights[i, j] + (rn.random() - 0.5) / maxChange
        for i in range(len(self.twoWeights[:, 0])):
            for j in range(len(self.twoWeights[0, :])):
                self.twoWeights[i, j] = parent.twoWeights[i, j] + (rn.random() - 0.5) / maxChange

    # Defines the activation function used throughout the model
    # Here the sigmoid function is being used
    def activation(self, value):
        # if value < 0:
        #     return 0
        # else:
        #     return value
        return max(0, value)
        # return 1 / (1 + math.exp(-value))

    # Going from input to output is repeated matrix multiplication
    # The nodes are subject to the activation function after each step
    def get_output(self, eye):
        eye = eye.flatten()
        self.layerOne = np.matmul(self.inputWeights, eye)
        for i in range(len(self.layerOne)):
            self.layerOne[i] = self.activation(self.layerOne[i])
        self.layerTwo = np.matmul(self.oneWeights, self.layerOne)
        for i in range(len(self.layerTwo)):
            self.layerTwo[i] = self.activation(self.layerTwo[i])
        self.output = np.matmul(self.twoWeights, self.layerTwo)
        for i in range(len(self.output)):
            self.output[i] = 1 / (1 + math.exp(-self.output[i]))
        return self.output

    def spit_model(self):
        model = []
        model.append(len(self.inputWeights[0, :]))
        model.append(len(self.inputWeights[:, 0]))
        model.append(len(self.twoWeights[0, :]))
        model.append(len(self.twoWeights[:, 0]))
        for row in self.inputWeights:
            for weight in row:
                model.append(weight)
        for row in self.oneWeights:
            for weight in row:
                model.append(weight)
        for row in self.twoWeights:
            for weight in row:
                model.append(weight)
        return model

    def eat_model(self, model):
        self.inputWeights = np.zeros((int(model[1]), int(model[0])))
        self.layerOne = []
        self.oneWeights = np.zeros((int(model[2]), int(model[1])))
        self.layerTwo = []
        self.twoWeights = np.zeros((int(model[3]), int(model[2])))
        self.output = []
        for i in range(int(model[1])):
            for j in range(int(model[0])):
                self.inputWeights[i][j] = float(model[j + 4 + int(model[0]) * i])
        for i in range(int(model[2])):
            for j in range(int(model[1])):
                self.oneWeights[i][j] = float(model[j + 4 + int(model[1]) * i + int(model[1]) * int(model[0])])
        for i in range(int(model[3])):
            for j in range(int(model[2])):
                self.twoWeights[i][j] = float(model[j + 4 + int(model[2]) * i + int(model[1]) * int(model[0]) + int(model[2]) * int(model[1])])


# testMat = np.zeros((3, 5))
# testVec = np.zeros(5)
# print(np.matmul(testMat, testVec))

# dim = ActorDimensions(12, 15, 15, 104)
# protoActor = Actor(dim.dimensions)
# newActor = Actor(dim.dimensions)
# newActor.initialize_weights(protoActor)
# testState = np.zeros(12)
# for i in range(4):
#     testState[i] = math.ceil(rn.random() * 104)
#     testState[i + 4] = math.ceil(rn.random() * 5)
#     testState[i + 8] = math.ceil(rn.random() * 12)
#
# print(testState)
# newActor.get_output(testState)
# print(newActor.output)
