import keras
import numpy as np
from nn_input import *
from __init__ import *

class BatchGenerator(keras.utils.Sequence):
    def __init__(self, id,  batch_size=32, shuffle=True):
        """Initialization"""
        self.id = id
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.id))
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self.id) / float(self.batch_size)))

    def __getitem__(self, index):
        """Generate one batch of data"""
        batch_keys = [self.id[ID_] for ID_ in self.indexes[index*self.batch_size:(index+1)*self.batch_size]]
        # print 'batch_keys'
        # print batch_keys
        X_main = None
        X_side = None
        Y = None

        # (sol_X_main, sol_X_side ), sol_Y = nn_input(1)

        # X_main = np.zeros((len(batch_keys), sol_X_main.shape[0], sol_X_main.shape[1], sol_X_main.shape[2]))
        # X_side = np.zeros((len(batch_keys), len(sol_X_side)))
        # Y = np.zeros(len(batch_keys))
        

        for index, key in enumerate(batch_keys):
            
            if index == 0:
                (sol_X_main, sol_X_side ), sol_Y = nn_input(key)
                
                X_main = np.zeros((len(batch_keys), sol_X_main.shape[0], sol_X_main.shape[1], sol_X_main.shape[2]))
                X_side = np.zeros((len(batch_keys), len(sol_X_side)))
                Y = np.zeros(len(batch_keys))

                X_main[index] = sol_X_main
                X_side[index] = sol_X_side
                Y[index] = sol_Y

            else:
                (X_main[index], X_side[index]), Y[index] = nn_input(key)

        X_main = X_main.astype(np.float32)
        X_side = X_side.astype(np.float32)
        Y = Y.astype(np.float32)

        return [X_main, X_side], Y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(np.arange(len(self.id)))

class BatchGenerator_Validation(keras.utils.Sequence):
    def __init__(self, id,  batch_size=32, shuffle=False):
        """Initialization"""
        self.id = id
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.id))
        self.on_epoch_end()

    def __len__(self):
        """Denotes the number of batches per epoch"""
        return int(np.ceil(len(self.id) / float(self.batch_size)))

    def __getitem__(self, index):
        """Generate one batch of data"""
        batch_keys = [self.id[ID_] for ID_ in self.indexes[index*self.batch_size:(index+1)*self.batch_size]]
        # print 'batch_keys'
        # print batch_keys
        X_main = None
        X_side = None
        Y = None

        # (sol_X_main, sol_X_side ), sol_Y = nn_input(1)

        # X_main = np.zeros((len(batch_keys), sol_X_main.shape[0], sol_X_main.shape[1], sol_X_main.shape[2]))
        # X_side = np.zeros((len(batch_keys), len(sol_X_side)))
        # Y = np.zeros(len(batch_keys))
        

        for index, key in enumerate(batch_keys):
            
            if index == 0:
                (sol_X_main, sol_X_side ), sol_Y = nn_input(key)
                
                X_main = np.zeros((len(batch_keys), sol_X_main.shape[0], sol_X_main.shape[1], sol_X_main.shape[2]))
                X_side = np.zeros((len(batch_keys), len(sol_X_side)))
                Y = np.zeros(len(batch_keys))

                X_main[index] = sol_X_main
                X_side[index] = sol_X_side
                Y[index] = sol_Y

            else:
                (X_main[index], X_side[index]), Y[index] = nn_input(key)

        X_main = X_main.astype(np.float32)
        X_side = X_side.astype(np.float32)
        Y = Y.astype(np.float32)

        return [X_main, X_side], Y

    def on_epoch_end(self):
        """Updates indexes after each epoch"""
        if self.shuffle:
            self.indexes = np.random.permutation(np.arange(len(self.id)))


