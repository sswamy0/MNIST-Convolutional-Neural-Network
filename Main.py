import random
from Image import DataReader
from Image import Image
from NeuralNetwork import NeuralNetwork

from NetworkBuilder import NetworkBuilder

class Main():
    def __init__(self):
        self.seed = 123
        print("Starting data loading...")
        data_reader = DataReader()
        imagesTest : list[Image] = data_reader.readData("data/mnist_test.csv")
        imagesTrain : list[Image] = data_reader.readData("data/mnist_train.csv")

        print("Images Train size", len(imagesTrain))
        print("Images Test size", len(imagesTest))

        builder : NetworkBuilder = NetworkBuilder(28,28,256*100)
        builder.addConvolutionalLayer(8,5,1,0.01,self.seed)
        builder.addMaxPoolLayer(3, 2)
        builder.addFullyConnectedLayer(10,0.01, self.seed)

        net : NeuralNetwork = builder.build()

        rate : float = net.test(imagesTest)
        print(f"Pre training success rate: {rate}")

        epochs = 3
        for i in range(epochs):
            random.shuffle(imagesTrain)
            net.train(imagesTrain, i)
            rate = net.test(imagesTest)
            print(f"Success rate after round {i}: {rate}")

if __name__ == "__main__":
    Main()