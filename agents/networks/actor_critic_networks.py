from keras import Sequential
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam


# see https://github.com/rlcode/reinforcement-learning/blob/master/2-cartpole/4-actor-critic/cartpole_a2c.py
def build_actor_network(obs_space, action_space, learning_rate):
    actor = Sequential()
    actor.add(Conv2D(32, kernel_size=(8, 8), strides=(4, 4), activation='relu', input_shape=obs_space,
                     kernel_initializer='he_uniform', data_format='channels_first'))
    actor.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='relu', input_shape=obs_space))
    actor.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu',  input_shape=obs_space))
    # convert image from 2D to 1D
    actor.add(Flatten())

    actor.add(Dense(units=512, activation='relu', kernel_initializer='he_uniform'))

    # output layer
    actor.add(Dense(units=action_space, activation='softmax', kernel_initializer='he_uniform'))

    # compile the self.model using traditional Machine Learning losses and optimizers
    actor.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate))
    # actor.summary()
    return actor


# see https://github.com/rlcode/reinforcement-learning/blob/master/2-cartpole/4-actor-critic/cartpole_a2c.py
def build_critic_network(obs_space, value_size, learning_rate):
    critic = Sequential()
    critic.add(Conv2D(32, kernel_size=(8, 8), strides=(4, 4), padding='valid', activation='relu',
                      input_shape=obs_space, kernel_initializer='he_uniform', data_format='channels_first'))
    critic.add(Conv2D(64, kernel_size=(4, 4), strides=(2, 2), padding='valid', activation='relu',
                      input_shape=obs_space, kernel_initializer='he_uniform', data_format='channels_first'))
    critic.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu',
                      input_shape=obs_space, kernel_initializer='he_uniform', data_format='channels_first'))
    # convert image from 2D to 1D
    critic.add(Flatten())

    critic.add(Dense(units=512, activation='relu'))

    # output layer
    critic.add(Dense(units=value_size, activation='linear'))

    # compile the self.model using traditional Machine Learning losses and optimizers
    critic.compile(loss="mse", optimizer=Adam(lr=learning_rate))
    # model.summary()
    return critic


# see https://github.com/simoninithomas/reinforcement-learning-1/blob/master/2-cartpole/3-reinforce/cartpole_reinforce.py
def build_actor_cartpole_network(obs_space, action_space, learning_rate):
    actor = Sequential()
    actor.add(Dense(24, input_dim=obs_space[0], activation='relu', kernel_initializer='he_uniform'))
    actor.add(Dense(action_space, activation='softmax'))
    # actor.summary()
    # Using categorical crossentropy as a loss is a trick to easily implement the policy gradient.
    # Categorical cross entropy is defined as (p, q) = sum(p_i * log(q_i)).
    # For the action taken, a, you set p_a = advantage.
    # q_a is the output of the policy network, which is the probability of taking the action a, i.e. policy(s, a).
    # All other p_i are zero, thus we have H(p, q) = A * log(policy(s, a))
    actor.compile(loss='categorical_crossentropy', optimizer=Adam(lr=learning_rate))
    return actor


# see https://github.com/rlcode/reinforcement-learning/blob/master/2-cartpole/4-actor-critic/cartpole_a2c.py
def build_critic_cartpole_network(obs_space, value_size, learning_rate):
    critic = Sequential()
    critic.add(Dense(24, input_dim=obs_space[0], activation='relu', kernel_initializer='he_uniform'))
    critic.add(Dense(value_size, activation='linear'))
    # critic.summary()
    critic.compile(loss="mse", optimizer=Adam(lr=learning_rate))
    return critic
