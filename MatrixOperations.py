class MatrixOperations():


    def addMatrices(self, a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
        

        result = [[0.0 for _ in range(len(a[0]))] for _ in range(len(a))]
        for i in range(len(a)):
            for j in range(len(a[0])):
                result[i][j] = a[i][j]+ b[i][j]
        return result

    def addVectors(self, a: list[float], b: list[float]) -> list[float]:
        result = [0.0] * (len(a))
        for i in range(len(a)):
            result[i] = a[i] + b[i]
        return result

    def multiplyMatrixByScalar(self, a: list[list[float]], scalar: float) -> list[list[float]]:
        result = [[0.0 for _ in range(len(a[0]))] for _ in range(len(a))]
        for i in range(len(a)):
            for j in range(len(a[0])):
                result[i][j] = a[i][j] * scalar
        return result

    def multiplyVectorByScalar(self, a: list[float], scalar: float) -> list[float]:
        result = [0.0] * (len(a))
        for i in range(len(a)):
            result[i] = a[i]*scalar
            
        return result
        
