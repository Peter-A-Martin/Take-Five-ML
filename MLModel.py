import numpy as np
import math
import random as rn


class ActorDimensions:
    def __init__(self, eye, layerOne, layerTwo, output):
        self.dimensions = [eye, layerOne, layerTwo, output]

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
        return 1 / (1 + math.exp(-value))

    # Going from input to output is repeated matrix multiplication
    # The nodes are subject to the activation function after each step
    def get_output(self, eye):
        self.layerOne = np.matmul(self.inputWeights, eye)
        for i in range(len(self.layerOne)):
            self.layerOne[i] = self.activation(self.layerOne[i])
        self.layerTwo = np.matmul(self.oneWeights, self.layerOne)
        for i in range(len(self.layerTwo)):
            self.layerTwo[i] = self.activation(self.layerTwo[i])
        self.output = np.matmul(self.twoWeights, self.layerTwo)
        for i in range(len(self.output)):
            self.output[i] = self.activation(self.output[i])
        return self.output


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
