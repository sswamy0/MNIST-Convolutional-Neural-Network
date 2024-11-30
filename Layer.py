
from typing import Optional


class Layer():
    def __init__(self):
        self.nextLayer: Optional["Layer"] = None  # Use _nextLayer to indicate it's a private attribute
        self.prevLayer: Optional["Layer"] = None  # Similarly for _prevLayer


    # Getter for nextLayer
    def getNextLayer(self) -> Optional["Layer"]:
        return self.nextLayer

    # Setter for nextLayer
    def setNextLayer(self, nextLayer: Optional["Layer"]) -> None:
        self.nextLayer = nextLayer

    # Getter for prevLayer
    def getPrevLayer(self) -> Optional["Layer"]:
        return self.prevLayer

    # Setter for prevLayer
    def setPrevLayer(self, prevLayer: Optional["Layer"]) -> None:
        self.prevLayer = prevLayer



    def matrixToVector(self,input:list[list[list[float]]]):
        length:int = len(input)
        rows:int =  len(input[0])
        cols:int = len(input[0][0])

        vector:list[float] = []

        for l in range(length):
            for r in range(rows):
                for c in range(cols):
                    vector.append(input[l][r][c])

        return vector

    def vectorToMatrix(self, input:list[float],length:int,rows:int,cols:int):
        out:list[list[list[float]]] =[]
        i:int = 0
        for l in range(length):
            matrix:list[list[float]] = [[0 for _ in range(cols)] for _ in range(rows)]
            for r in range(rows):
                for c in range(cols):
                    matrix[r][c] = input[i]
                    i+=1
            out.append(matrix)
        return out