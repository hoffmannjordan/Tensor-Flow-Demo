import keras
from keras import Model
from keras.layers import Input, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.layers.merge import concatenate
from input_batch_generator import *

from __init__ import *

def nn_model(input_shape):
    print 'Defining NN model'
    conv_kernel = (3, 3)
    conv_channels = 16
    pool_size = (2, 2)
    n_hidden = 100

    # Main input 
    main_input = Input(shape=input_shape[0])
    x = Conv2D(conv_channels, conv_kernel, activation="relu")(main_input)
    x = MaxPooling2D(pool_size=pool_size)(x)
    x = Flatten()(x)

    # Create some side fetures
    side_input = Input(shape=(input_shape[1],))
    merged_layer = concatenate([x, side_input], axis=-1)

    # Make the Final prediction
    x = Dense(n_hidden, activation="relu")(merged_layer)
    x = Dropout(0.5)(x)
    y = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=[main_input, side_input], outputs=y)

    return model


class NeuralNet:
    def __init__(self, IDS, epochs=10):
        # training_data_lookup =
        window_size = (81,81)
        self.nn_model = nn_model(((window_size[0], window_size[1], FEATURES), SIDE_FEATURES))
        self.batch_generator = BatchGenerator(IDS)
        self.epochs = epochs

    def train(self):
        print 'Starting training'
        optimizer = keras.optimizers.RMSprop(lr=0.0005)


        self.nn_model.compile(loss='mean_squared_error',
                              optimizer=optimizer,
                              metrics=['mean_squared_error'])

        self.nn_model.fit_generator(self.batch_generator,
                                    epochs=self.epochs)




