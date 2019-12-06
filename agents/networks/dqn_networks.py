from keras import Sequential
from keras.layers import Conv2D, Flatten, Dense
from keras.optimizers import Adam


def build_dqn_network(input_shape, output_shape, learning_rate):
    model = Sequential()
    # 2 layers of convolutional networks
    # padding is added so that information is not lost when the kernel size is smaller
    model.add(Conv2D(32, kernel_size=(8, 8), strides=(4, 4), padding='valid', activation='relu',
                     input_shape=input_shape, data_format='channels_first'))
    model.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu',))
    model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu',))
    # convert image from 2D to 1D
    model.add(Flatten())

    # hidden layer takes a pre-processed frame as input, and has 200 units
    #  fibre channel layer 1
    model.add(Dense(units=512, activation='relu', kernel_initializer='glorot_uniform'))

    # output layer
    model.add(Dense(units=output_shape, activation='softmax', kernel_initializer='RandomNormal'))

    # compile the self.model using traditional Machine Learning losses and optimizers
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
    # model.summary()
    return model


def build_dqn_cartpole_network(input_shape, output_shape, learning_rate):
    model = Sequential()
    # input layer
    model.add(Dense(24, input_shape=input_shape, activation="relu", kernel_initializer='he_uniform'))
    model.add(Dense(24, activation='relu', kernel_initializer='he_uniform'))
    # output layer
    model.add(Dense(output_shape, activation='linear', kernel_initializer='he_uniform'))
    model.compile(optimizer=Adam(lr=learning_rate), loss='mse')
    # model.summary()
    return model


# see https://github.com/rohitgirdhar/Deep-Q-Networks
def build_simple_convoluted_net(input_shape, output_shape, learning_rate):
    model = Sequential()
    model.add(Conv2D(16, 8, strides=(4, 4), activation='relu', input_shape=input_shape, data_format='channels_first'))
    model.add(Conv2D(32, 4, strides=(2, 2), activation='relu'))
    model.add(Flatten())

    model.add(Dense(256, activation='relu'))
    model.add(Dense(output_shape, activation=None))

    model.compile(optimizer=Adam(lr=learning_rate), loss='mse')
    return model


