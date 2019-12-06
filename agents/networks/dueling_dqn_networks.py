import keras
from keras.models import Model
from keras.layers import Dense, Flatten, Conv2D, Lambda, Input
from keras.optimizers import Adam
from keras import backend as K


def build_dueling_dqn_network(obs_space, action_space, learning_rate):
    # see https://github.com/UoA-RL/Gym-T4-Testbed/blob/henry_test/models.py
    state_input = Input(shape=obs_space)
    x = Conv2D(16, kernel_size=(8, 8), strides=(4, 4), activation='relu',
               data_format='channels_first')(state_input)
    x = Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation='relu')(x)
    x = Flatten()(x)

    # state value tower - V
    state_value = Dense(256, activation='relu')(x)
    state_value = Dense(1, init='uniform')(state_value)
    state_value = Lambda(lambda s: K.expand_dims(s[:, 0], axis=-1), output_shape=(action_space,))(state_value)

    # action advantage tower - A
    action_advantage = Dense(256, activation='relu')(x)
    action_advantage = Dense(action_space)(action_advantage)
    action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True),
                              output_shape=(action_space,))(action_advantage)

    # merge to state-action value function Q
    state_action_value = keras.layers.add([state_value, action_advantage])

    model = Model(input=state_input, output=state_action_value)
    # model.compile(rmsprop(lr=learning_rate), "mse")
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    # model.summary()
    return model


def build_dueling_cartpole_network(obs_space, action_space, learning_rate):
    state_input = Input(shape=obs_space)
    x = Dense(24, activation='relu')(state_input)
    x = Dense(24, activation='relu')(x)

    state_value = Dense(12, activation='relu')(x)
    state_value = Dense(1, init='uniform')(state_value)
    state_value = Lambda(lambda s: K.expand_dims(s[:, 0], axis=-1), output_shape=(action_space,))(state_value)

    action_advantage = Dense(12, activation='relu')(x)
    action_advantage = Dense(action_space)(action_advantage)
    action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True),
                              output_shape=(action_space,))(action_advantage)

    state_action_value = keras.layers.add([state_value, action_advantage])
    model = Model(input=state_input, output=state_action_value)
    model.compile(loss='mse', optimizer=Adam(lr=learning_rate))
    # model.summary()
    return model
