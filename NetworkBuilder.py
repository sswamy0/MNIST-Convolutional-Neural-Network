from Layer import Layer
from MaxPoolLayer import MaxPoolLayer
from ConvolutionLayer import ConvolutionLayer
from FullyConnectedLayer import FullyConnectedLayer
from NeuralNetwork import NeuralNetwork


class NetworkBuilder():
    def __init__(self, inputRows: int, inputCols : int, scaleFactor : float):

        self.inputRows = inputRows
        self.inputCols = inputCols
        self.layers : list[Layer] = []
        self.scaleFactor = scaleFactor


    def addConvolutionalLayer(self, numFilters : int, filterSize: int, stepSize: int, learningRate: float, seed:int):
        if not self.layers:
            self.layers.append(ConvolutionLayer(filterSize, stepSize, 1, self.inputRows, self.inputCols, seed, numFilters, learningRate ))
        else:
            prev : Layer = self.layers[len(self.layers)-1]
            self.layers.append(ConvolutionLayer(filterSize, stepSize, prev.getOutputLength(), prev.getOutputRows(), prev.getOutputCols(), seed, numFilters, learningRate ))


    def addMaxPoolLayer(self, windowSize : int, stepSize: int):
        if not self.layers:
            self.layers.append(MaxPoolLayer(stepSize, windowSize, 1, self.inputRows, self.inputCols))
        else:
            prev : Layer = self.layers[len(self.layers)-1]
            self.layers.append(MaxPoolLayer(stepSize, windowSize, prev.getOutputLength(),prev.getOutputRows(), prev.getOutputCols()))


    def addFullyConnectedLayer(self, outLength : int, learningRate : float, seed: int):
        if not self.layers:
            self.layers.append(FullyConnectedLayer(self.inputCols*self.inputRows, outLength, seed, learningRate))

        else:
            prev : Layer = self.layers[len(self.layers)-1]
            self.layers.append(FullyConnectedLayer(prev.getOutputElements(), outLength, seed, learningRate))


    def build(self)->NeuralNetwork:
        net : NeuralNetwork = NeuralNetwork(self.layers, self.scaleFactor)
        return net