
class Image():
    def __init__(self, data,label):
        self.data = data
        self.label = label

    def getData(self):
        return self.data

    def getLabel(self):
        return self.label

    def toString(self):
        s = str(self.label) + ", \n"  # Convert label to string before concatenation
        for i in range(len(self.data)):
            for j in range(len(self.data[0])):
                s += self.data[i][j] + ", "# Convert pixel value to string
            s += "\n"
        return s

class DataReader:
    def __init__(self):
        self.rows= 28
        self.cols = 28

    def readData(self, path):
        images:list = []
        try:
            with open(path, 'r') as file:
                # Read each line
                for line in file:
                    lineItems = line.strip().split(",")  # Split the line by commas


                    # The first element is the label
                    label = int(lineItems[0])
                    # The rest are the pixel values, reshape them into a 28x28 array
                    data = [[float(lineItems[i * self.cols + j + 1]) for j in range(self.cols)] for i in range(self.rows)]

                    # Append the created Image object to the list
                    images.append(Image(data, label))

        except FileNotFoundError:
            raise FileNotFoundError(f"File not found: {path}")
        except Exception as e:
            raise RuntimeError(f"Error reading file {path}: {e}")

        return images


class Main:
    def __init__(self):
        self.dataReader = DataReader()
        images = self.dataReader.readData("data/mnist_test.csv")
        if images:
            print(images[0].toString())
if __name__ == "__main__":
    main = Main()