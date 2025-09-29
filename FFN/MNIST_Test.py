import numpy as np
import random
from mnist_loader import datas
from Network import Network
    
if __name__ == '__main__':
    training_data, validation_data, test_data = datas()
    # print(training_data)
    net = Network([784,40,40,10])
    net.SGD(training_data, 30, 10, 3.0, evaluation_data=validation_data,
            monitor_training_accuracy=True,
            monitor_evaluation_accuracy=True)