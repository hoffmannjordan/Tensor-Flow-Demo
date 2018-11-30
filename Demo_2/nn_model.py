import keras
from keras import Model
from keras.layers import Input, Dropout, Flatten, Dense, Conv2D, MaxPooling2D
from keras.layers.merge import concatenate


def nn_model(input_shape):
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

    def __init__(self, solution_keys, epochs=10):

        training_data_lookup =

        self.nn_model = nn_model(((window_size[0], window_size[1], NUMBER_OF_FEATURE_GRIDS), NUMBER_OF_SIDE_FEATURES))
        self.batch_generator = BatchGenerator(solution_keys)
        self.epochs = epochs

    def train(self):
        optimizer = keras.optimizers.RMSprop(lr=0.0005)


        self.nn_model.compile(loss='mean_squared_error',
                              optimizer=optimizer,
                              metrics=['mean_squared_error'])

        self.nn_model.fit_generator(self.batch_generator,
                                    epochs=self.epochs)

