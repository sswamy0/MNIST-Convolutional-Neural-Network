import random
from MatrixOperations import MatrixOperations
from Layer import Layer

class ConvolutionLayer(Layer, MatrixOperations):
    def __init__(self, filterSize: int, stepSize:int, inLength:int, inRows:int, inCols:int, seed:int, numFilters : int, learningRate : float):
        super().__init__()
        self.filterSize = filterSize
        self.stepSize = stepSize
        self.inLength = inLength
        self.inRows = inRows
        self.inCols = inCols
        self.seed = seed
        self.learningRate = learningRate

        self.generateRandomFilters(numFilters)
        self.filters : list[list[list[float]]] = []
        self.lastInput : list[list[list[float]]] = None
        self.generateRandomFilters(numFilters)



    def generateRandomFilters(self, numFilters:int):
        random.seed(self.seed)
        filters :list[list[list[float]]] = []
        for n in range(numFilters):
            newFilter : list[list[float]] = [[0.0 for _ in range(self.filterSize)]for _ in range(self.filterSize)]
            for i in range(self.filterSize):
                for j in range(self.filterSize):
                    value = random.gauss(0,1)
                    newFilter[i][j]= value

            filters.append(newFilter)
        self.filters = filters


    def convolve(self, input: list[list[float]], filter: list[list[float]], stepSize: int):
        outRows = (len(input)-len(filter))//stepSize +1
        outCols = (len(input[0])-len(filter))//stepSize +1
        inRows = len(input)
        inCols = len(input[0])
        fRows = len(filter)
        fCols = len(filter[0])

        output : list[list[float]] = [[0.0 for _ in range(outCols)] for _ in range(outRows)]

        outRow = 0


        for i in range(0, inRows - fRows + 1, stepSize):
            outCol = 0

            for j in range(0, inCols-fCols +1, stepSize):

                sum = 0.0

                #apply filter around this position
                for x in range(fRows):
                    for y in range(fCols):
                        inputRowIndex = i +x
                        inputColIndex = j+y

                        value = filter[x][y]*input[inputRowIndex][inputColIndex]
                        sum+=value
                output[outRow][outCol] = sum
                outCol+=1
            outRow+=1

        return output

    def spaceArray(self, input:list[list[float]])->list[list[float]]:
        if self.stepSize == 1:
            return input
        outRows = (len(input)-1)*self.stepSize+1
        outCols = (len(input[0])-1)*self.stepSize+1

        output : list[list[float]] = [[0.0 for _ in range(outCols)]for _ in range(outRows)]
        for i in range(len(input)):
            for j in range(len(input[0])):
                output[i*self.stepSize][j*self.stepSize] = input[i][j]

        return output

    def convolutionForwardPass(self, list : list[list[list[float]]])->list[list[list[float]]]:
        self.lastInput=list
        output : list[list[list[float]]] = []
        for i in range(len(list)):
            for filter in self.filters:
                output.append(self.convolve(list[i],filter,self.stepSize))

        return output



    def getMatrixOutput(self,input:list[list[list[float]]])->list[float]:
        output :list[list[list[float]]] = self.convolutionForwardPass(input)
        return self.nextLayer.getMatrixOutput(output)



    def getVectorOutput(self,input:list[float])->list[float]:
        matrixInput : list[list[list[float]]] = self.vectorToMatrix(input,self.inLength, self.inRows,self.inCols)
        return self.getMatrixOutput(matrixInput)



    def backPropagationVector(self,dldO:list[float]):
        matrixInput : list[list[list[float]]] = self.vectorToMatrix(dldO,self.inLength, self.inRows,self.inCols)
        #print("Length of matrixInput", len(matrixInput[0]))
        self.backPropagationMatrix(matrixInput)


    def backPropagationMatrix(self,dldO:list[list[list[float]]]):
        filtersDelta : list[list[list[float]]] = []
        dldOPreviousLayer : list[list[list[float]]] = []

        for h in range(len(self.filters)):
            filtersDelta.append([[0.0] * self.filterSize for _ in range(self.filterSize)])


        for i in range(len(self.lastInput)):

            errorForInput : list[list[float]] = [[0.0 for _ in range(self.inCols)] for _ in range(self.inRows)]

            for f in range(len(self.filters)):
                currFilter : list[list[float]] = self.filters[f]
                #print("currFilter: ",len(currFilter), " x ", len(currFilter[0]))
                
                error : list[list[float]] = dldO[i*(len(self.filters))+f]
                #print("Length of error", len(error), " x ", len(error[0]))

                spacedError : list[list[float]] = self.spaceArray(error)
                #print("Length of spacedError", len(spacedError), " x ", len(spacedError[0]))
                
                dldF : list[list[float]] = self.convolve(self.lastInput[i],spacedError,1)
                #print("dldF: ", len(dldF), " x ", len(dldF[0]))
                
                
                delta : list[list[float]] = self.multiplyMatrixByScalar(dldF, self.learningRate*-1)
                #print("delta: ", len(delta), " x ", len(delta[0]))
                
                newTotalDelta : list[list[float]] = self.addMatrices(filtersDelta[f],delta)
                #print("newTotalDelta: ", len(newTotalDelta), " x ", len(newTotalDelta[0]))
                
                filtersDelta[f] =  newTotalDelta
                #print("filtersDeltaF: ", len(filtersDelta[f]), " x ", len(filtersDelta[0]))

                flippedError : list[list[float]] = self.flipArrayHorizontal(self.flipArrayVertical(spacedError))
                #print("flippedError length: ", len(flippedError), " x ", len(flippedError[0]))
               
                convResult = self.fullConvolve(currFilter, flippedError)
                #print(f"fullConvolve result dimensions: {len(convResult)}x{len(convResult[0])}")

                # Perform addition
                #print("errorForInput: ", len(errorForInput), " x ", len(errorForInput[0]))
                
                errorForInput = self.addMatrices(errorForInput, convResult)

            dldOPreviousLayer.append(errorForInput)


        for f in range(len(self.filters)):
            modified : list[list[float]] = self.addMatrices(filtersDelta[f], self.filters[f])
            self.filters[f]=modified

        if(self.prevLayer is not None):
            #print("Length of dldOPreviousLayer", len(dldOPreviousLayer[0]))
            self.prevLayer.backPropagationMatrix(dldOPreviousLayer)




    def flipArrayHorizontal(self, array:list[list[float]])->list[list[float]]:
        rows = len(array)
        cols = len(array[0])
        output : list[list[float]] = [[0.0 for _ in range(cols)] for _ in range(rows)]

        for i in range(rows):
            for j in range(cols):
                output[rows-i-1][j]=array[i][j]

        return output

    def flipArrayVertical(self, array:list[list[float]])->list[list[float]]:
        rows = len(array)
        cols = len(array[0])
        output : list[list[float]] = [[0.0 for _ in range(cols)] for _ in range(rows)]

        for i in range(rows):
            for j in range(cols):
                output[i][cols-j-1]=array[i][j]

        return output

    def fullConvolve(self, input: list[list[float]], filter: list[list[float]]):
        outRows = (len(input)+len(filter)) +1
        #print("outRows: " ,outRows)
        outCols = (len(input[0])+len(filter[0])) +1
        #print("outCols: ", outCols)

        inRows = len(input)
        inCols = len(input[0])

        fRows = len(filter)
        fCols = len(filter[0])

        output : list[list[float]] = [[0.0 for _ in range(outCols)] for _ in range(outRows)]

        outRow = 0


        for i in range(-1*fRows +1, inRows, 1):
            outCol = 0

            for j in range(-1*fCols+1, inCols, 1):

                sum = 0.0

                #apply filter around this position
                for x in range(fRows):
                    for y in range(fCols):
                        inputRowIndex = i +x
                        inputColIndex = j+y

                        if inputRowIndex >=0 and inputColIndex >=0 and inputRowIndex < inRows and inputColIndex < inCols:
                            value = filter[x][y]*input[inputRowIndex][inputColIndex]
                            sum+=value
                output[outRow][outCol] = sum
                outCol+=1
            outRow+=1

        return output



    def getOutputLength(self)->int:
        return len(self.filters)*self.inLength


    def getOutputRows(self)->int:
        return (self.inRows -self.filterSize)//self.stepSize+1


    def getOutputCols(self)->int:
        return (self.inCols - self.filterSize)//self.stepSize+1


    def getOutputElements(self)->int:
        return self.getOutputCols()*self.getOutputRows()*self.getOutputLength()
