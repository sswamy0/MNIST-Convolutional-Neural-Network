from Layer import Layer
from MatrixOperations import MatrixOperations
from Image import Image
from Image import DataReader
class NeuralNetwork(MatrixOperations, Image):
    def __init__(self, layers: list[Layer], scaleFactor : float):
        self.layers = layers

        self.scaleFactor = scaleFactor
        self.linkLayers()


    def linkLayers(self):
        if len(self.layers) <=1:
            return

        for i in range(len(self.layers)):
            if i == 0:
                self.layers[i].setNextLayer(self.layers[i+1])
            elif i == len(self.layers)-1:
                self.layers[i].setPrevLayer(self.layers[i-1])
            else:
                self.layers[i].setPrevLayer(self.layers[i-1])
                self.layers[i].setNextLayer(self.layers[i+1])


    def getErrors(self, networkOutput: list[float], correctAnswer: int)->list[float]:
        numClasses : int = len(networkOutput)
        expected : list[float] = [0.0]*numClasses

        expected[correctAnswer] =1

        return self.addVectors(networkOutput, self.multiplyVectorByScalar(expected, -1))

    def getMaxIndex(self, ind:list[float])->int:
        max = 0.0
        index = 0

        for i in range(len(ind)):
            if ind[i]>=max:
                max = ind[i]
                index = i

        return index

    def guess(self, image:Image )->int:
        inList : list[list[list[float]]] = []
        inList.append(self.multiplyMatrixByScalar(image.getData(),(1.0/self.scaleFactor)))

        out:list[float] = self.layers[0].getMatrixOutput(inList)

        guess : int =  self.getMaxIndex(out)
        
        

        return guess

    def test(self, images: list[Image])->float:
        correct = 0.0
        for image in images:
            guess = self.guess(image)

            if guess == image.getLabel():
                correct+=1
        return correct/len(images)

    def train(self, images : list[Image], epoch):
        for image in images:
            inList : list[list[list[float]]] = []
            inList.append(self.multiplyMatrixByScalar(image.getData(),1.0/self.scaleFactor))


            out:list[float] = self.layers[0].getMatrixOutput(inList)
            dldO = self.getErrors(out,image.getLabel())

            self.layers[len(self.layers)-1].backPropagationVector(dldO)
            

