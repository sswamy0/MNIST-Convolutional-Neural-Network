

from Layer import Layer
class MaxPoolLayer(Layer):

    def __init__(self, stepSize: int, windowSize:int, inLength:int, inRows:int, inCols: int):
        super().__init__()
        self.stepSize = stepSize
        self.windowSize = windowSize

        self.inLength = inLength
        self.inRows = inRows
        self.inCols = inCols

        self.lastMaxRow : list[list[list[int]]] = []
        self.lastMaxCol : list[list[list[int]]] = []

    def maxPoolForwardPass(self, input:list[list[float]])->list[list[float]]:
        output :list[list[float]] = []
        self.lastMaxCol
        self.lastMaxRow
        for i in range(len(input)):
            output.append(self.pool(input[i]))

        return output

    def pool(self, input:list[list[float]])->list[list[float]]:
        output :list[list[float]] = [[0.0 for _ in range(self.getOutputCols())] for _ in range(self.getOutputRows())]

        maxRows : list[list[int]] = [[0 for _ in range(self.getOutputCols())] for _ in range(self.getOutputRows())]
        maxCols : list[list[int]] = [[0 for _ in range(self.getOutputCols())] for _ in range(self.getOutputRows())]

        for r in range(0,self.getOutputRows(),self.stepSize):
            for c in range(0,self.getOutputCols(), self.stepSize):
                max : float = 0.0
                maxRows[r][c] = -1
                maxCols[r][c] = -1
                for k in range(self.windowSize):
                    for l in range(self.windowSize):
                        if max<input[r+k][c+l]:
                            max = input[r+k][c+l]
                            maxRows[r][c] = r+k
                            maxCols[r][c] = c+l


                output[r][c] = max

        self.lastMaxRow.append(maxRows)
        self.lastMaxCol.append(maxCols)
        return output




    def getMatrixOutput(self,input:list[list[list[float]]])->list[float]:
        outputPool : list[list[list[float]]] = self.maxPoolForwardPass(input)

        return self.nextLayer.getMatrixOutput(outputPool)


    def getVectorOutput(self,input:list[float])->list[float]:
        matrixList : list[list[list[float]]] = self.vectorToMatrix(input,self.inLength,self.inRows,self.inCols)
        return self.getMatrixOutput(matrixList)



    def backPropagationVector(self,dldO:list[float]):
        matrixList : list[list[list[float]]] = self.vectorToMatrix(dldO,self.getOutputLength(),self.getOutputRows(),self.getOutputCols())
        self.backPropagationMatrix(matrixList)


    def backPropagationMatrix(self,dldO:list[list[list[float]]]):

        dxdl : list[list[list[float]]] = []
        l : int = 0
        for array in dldO:
            error : list[list[float]] = [[0.0 for _ in range(self.inCols)] for _ in range(self.inRows)]

            for r in range(self.getOutputRows()):
                for c in range(self.getOutputCols()):
                    max_i : int = self.lastMaxRow[l][r][c]
                    max_j : int = self.lastMaxCol[l][r][c]

                    if max_i != -1:
                        error[max_i][max_j] +=array[r][c]

            dxdl.append(error)
            l+=1

        if self.prevLayer is not None:
            self.prevLayer.backPropagationMatrix(dxdl)

    def getOutputLength(self)->int:
        return self.inLength

    def getOutputRows(self)->int:
        return (self.inRows-self.windowSize)//self.stepSize + 1

    def getOutputCols(self)->int:
        return (self.inCols-self.windowSize)//self.stepSize + 1

    def getOutputElements(self)->int:
        return self.inLength*self.getOutputCols()*self.getOutputRows()