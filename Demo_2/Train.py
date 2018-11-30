from nn_model import *
from input_batch_generator import *
from nn_input import *

def train_model():
    files_to_use = range(68)

    print "Number of solutions with training data: {}".format(len(files_to_use))

    nn = NeuralNet(files_to_use, epochs=100)
    nn.train()


if __name__ == "__main__":
    train_model()