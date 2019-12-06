from keras import Input, Model
from keras.layers import Dense, Conv2D, Flatten
from keras.optimizers import Adam
import keras.backend as K


# def build_ppo_actor_network(state_space, action_space, learning_rate, clipping_loss_ratio, entropy_loss_ratio):
def build_ppo_actor_network(state_space, action_space, learning_rate, clipping_loss_ratio):
    state_input = Input(shape=state_space)
    advantage = Input(shape=(1,))
    old_prediction = Input(shape=(action_space,))

    x = Conv2D(32, kernel_size=(8, 8), strides=(4, 4), activation='tanh', data_format='channels_first')(state_input)
    x = Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='tanh')(x)
    x = Flatten()(x)
    x = Dense(units=512, activation='tanh')(x)

    out_actions = Dense(action_space, activation='softmax', name='output')(x)

    model = Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])
    model.compile(optimizer=Adam(lr=learning_rate),
                  loss=[proximal_policy_optimization_loss(
                      advantage=advantage,
                      old_prediction=old_prediction,
                      clipping_loss_ratio=clipping_loss_ratio,)])  # entropy_loss_ratio=entropy_loss_ratio)])
    return model


def build_ppo_critic_network(state_space, learning_rate):
    state_input = Input(shape=state_space)

    x = Conv2D(32, kernel_size=(8, 8), strides=(4, 4), activation='tanh', data_format='channels_first')(state_input)
    x = Conv2D(64, kernel_size=(4, 4), strides=(2, 2), activation='tanh')(x)
    x = Flatten()(x)
    x = Dense(units=512, activation='tanh')(x)

    out_value = Dense(1)(x)

    model = Model(inputs=[state_input], outputs=[out_value])
    model.compile(optimizer=Adam(lr=learning_rate), loss='mse')
    return model


# def build_ppo_actor_cartpole_network(state_space, action_space, learning_rate, clipping_loss_ratio, entropy_loss_ratio):
def build_ppo_actor_cartpole_network(state_space, action_space, learning_rate, clipping_loss_ratio):
    state_input = Input(shape=state_space)
    advantage = Input(shape=(1,))
    old_prediction = Input(shape=(action_space,))

    x = Dense(128, activation='tanh')(state_input)
    x = Dense(128, activation='tanh')(x)

    out_actions = Dense(action_space, activation='softmax', name='output')(x)

    model = Model(inputs=[state_input, advantage, old_prediction], outputs=[out_actions])
    model.compile(optimizer=Adam(lr=learning_rate),
                  loss=[proximal_policy_optimization_loss(
                      advantage=advantage,
                      old_prediction=old_prediction,
                      clipping_loss_ratio=clipping_loss_ratio)])  # entropy_loss_ratio=entropy_loss_ratio)])
    return model


def build_ppo_critic_cartpole_network(state_space, learning_rate):
    state_input = Input(shape=state_space)
    x = Dense(128, activation='tanh')(state_input)
    x = Dense(128, activation='tanh')(x)
    out_value = Dense(1)(x)

    model = Model(inputs=[state_input], outputs=[out_value])
    model.compile(optimizer=Adam(lr=learning_rate), loss='mse')

    return model


# def proximal_policy_optimization_loss(advantage, old_prediction, clipping_loss_ratio, entropy_loss_ratio):
def proximal_policy_optimization_loss(advantage, old_prediction, clipping_loss_ratio):
    def loss(y_true, y_prediction):
        prob = y_true * y_prediction
        old_prob = y_true * old_prediction
        r = prob / (old_prob + 1e-10)
        # TODO: test this, compare to equation (9),
        #  see https://github.com/coreystaten/deeprl-ppo/blob/master/ppo.py (clip_loss, entropy_loss, value_loss)
        # return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - clipping_loss_ratio,
        #                                                max_value=1 + clipping_loss_ratio) * advantage)
        #                + entropy_loss_ratio * -(prob * K.log(prob + 1e-10)))
        return -K.mean(K.minimum(r * advantage, K.clip(r, min_value=1 - clipping_loss_ratio,
                                                       max_value=1 + clipping_loss_ratio) * advantage))

    return loss
