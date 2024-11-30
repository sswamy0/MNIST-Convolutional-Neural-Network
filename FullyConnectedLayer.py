
import random
from Layer import Layer
from typing import Optional

class FullyConnectedLayer(Layer):

    def __init__(self, inputLength : int, outputLength: int, seed: int, learningRate : float):
        super().__init__()
        self.inputLength = inputLength
        self.outputLength = outputLength
        self.learningRate = learningRate
        self.seed = seed

        self.weights : list[list[float]] = [[0.0 for _ in range(outputLength)] for _ in range(inputLength)]

        self.lastZ : list[float] = [0.0] * outputLength
        self.lastX : list[float] = [0.0] * inputLength
        self.leak : float = 0.01
        self.setRandomWeights()




    def setRandomWeights(self):
        random.seed(self.seed)
        for i in range(self.inputLength):
            for j in range(self.outputLength):
                self.weights[i][j]= random.gauss(0,1)

    def reLU(self,input:float)->float:
        if input<=0:
            return 0
        else:
            return input

    def derivativeReLU(self,input:float)->float:
        if input<=0:
            return self.leak
        else:
            return 1



    def fullyConnectedForwardPass(self, input: list[float]) ->list[float]:
        self.lastX = input
        z :list[float] = [0.0] * self.outputLength
        output : list[float] = [0.0] *self.outputLength

        for i in range(self.inputLength):
            for j in range(self.outputLength):
                z[j]+=input[i]*self.weights[i][j]

        self.lastZ = z;

        for j in range(self.outputLength):
            output[j] = self.reLU(z[j])
        return output



    def getMatrixOutput(self,input:list[list[list[float]]])->list[float]:
        vector : list[float] = self.matrixToVector(input)
        return self.getVectorOutput(vector)


    def getVectorOutput(self,input:list[float])->list[float]:
        forwardPass : list[float] = self.fullyConnectedForwardPass(input)
        if self.nextLayer is not None:
            return self.nextLayer.getVectorOutput(forwardPass)
        else:
            return forwardPass




    def backPropagationVector(self,dldO:list[float]):
        dldx : list[float] = [0.0] * self.inputLength
        dOdz : float
        dzdw : float
        dldw : float
        dzdx : float
        for i in range(self.inputLength):
            dldx_sum : float = 0.0
            for j in range(self.outputLength):
                dOdz = self.derivativeReLU(self.lastZ[j])
                dzdw = self.lastX[i]
                dldw = dldO[j]*dOdz*dzdw
                dzdx = self.weights[i][j]

                self.weights[i][j] -= dldw*self.learningRate
                dldx_sum +=dldO[j]*dOdz*dzdx

            dldx[i] = dldx_sum
        if self.prevLayer is not None:
            self.prevLayer.backPropagationVector(dldx)


    def backPropagationMatrix(self,dldO:list[list[list[float]]]):
        vector : list[float] = self.matrixToVector(dldO)
        self.backPropagationVector(vector)



    def getOutputLength(self)->int:
        return 0

    def getOutputRows(self)->int:
        return 0

    def getOutputCols(self)->int:
        return 0

    def getOutputElements(self)->int:
        return self.outputLength