from nn_model import *
from input_batch_generator import *
from nn_input import *

def train_model():
    files_to_use = range(1,3000)
    files_to_use_validation = range(3000,3629)

    print "Number of solutions with training data: {}".format(len(files_to_use))

    nn = NeuralNet(files_to_use,files_to_use_validation, epochs=10)
    nn.train()


if __name__ == "__main__":
    train_model()