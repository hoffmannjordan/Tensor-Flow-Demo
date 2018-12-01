import keras
from keras import Model
from keras.layers import Input, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.layers.merge import concatenate
from input_batch_generator import *
import numpy as np
from __init__ import *

def nn_model(input_shape):
    print 'Defining NN model'
    conv_kernel = (3, 3)
    conv_channels = 4
    conv_channels_2 = 8
    conv_channels_3 = 8
    pool_size = (2, 2)
    n_hidden_1 = 100
    n_hidden_2 = 10
    # Main input 
    main_input = Input(shape=input_shape[0])
    x = Conv2D(conv_channels, conv_kernel, activation="relu")(main_input)
    x = MaxPooling2D(pool_size=pool_size)(x)
    x = Conv2D(conv_channels_2, conv_kernel, activation="relu")(x)
    x = MaxPooling2D(pool_size=pool_size)(x)
    x = Conv2D(conv_channels_3, conv_kernel, activation="relu")(x)
    x = MaxPooling2D(pool_size=pool_size)(x)
    x = Conv2D(conv_channels_3, conv_kernel, activation="relu")(x)
    x = MaxPooling2D(pool_size=pool_size)(x)
    x = Flatten()(x)

    # Create some side fetures
    side_input = Input(shape=(input_shape[1],))
    merged_layer = concatenate([x, side_input], axis=-1)

    # Make the Final prediction
    x = Dense(n_hidden_1, activation="relu")(merged_layer)
    x = Dropout(0.5)(x)
    x = Dense(n_hidden_2, activation="relu")(x)
    x = Dropout(0.5)(x)
    y = Dense(1, activation="linear")(x)

    model = Model(inputs=[main_input, side_input], outputs=y)

    return model


class NeuralNet:
    def __init__(self, IDS, IDS_VALIDATION, epochs=10):
        # training_data_lookup =
        window_size = (81,81)
        self.nn_model = nn_model(((window_size[0], window_size[1], FEATURES), SIDE_FEATURES))
        self.batch_generator = BatchGenerator(IDS)
        self.batch_generator_validation = BatchGenerator_Validation(IDS_VALIDATION)
        self.epochs = epochs

    def train(self):
        print 'Starting training'
        

        #optimizer = keras.optimizers.RMSprop(lr=0.0005)


        self.nn_model.compile(loss='mean_squared_error',
                              #optimizer=optimizer,
                              optimizer='adam',
                              metrics=['mean_squared_error'])

        self.nn_model.fit_generator(self.batch_generator,
                                    epochs=self.epochs,validation_data=self.batch_generator_validation)

        print 'PREDICT'
        prediction = self.nn_model.predict_generator(self.batch_generator_validation, steps=None, max_queue_size=10, workers=1, use_multiprocessing=False, verbose=1)
        np.savetxt('./prediction.csv',prediction)
        print np.array(prediction).flatten()
        np.savetxt('./prediction2.csv',prediction)




