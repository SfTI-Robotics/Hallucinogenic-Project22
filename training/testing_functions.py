import numpy as np

from agents.image_input import AbstractBrain
from utils.summary import Summary


test_episodes = 10


def test(learner: AbstractBrain, env, config, processor, filename, path, episode=None):

    # add episode number to filename
    if episode is not None:
        filename += 'episode' + str(episode)

    test_summary = Summary(['sumiz_step', 'sumiz_reward'],
                           name=filename,
                           save_path=path,
                           min_reward=processor.reward_min,
                           max_reward=processor.reward_max)

    temp_epsilon = learner.epsilon
    learner.epsilon = 0.0

    test_rewards = []
    test_steps = []

    for e in range(test_episodes):
        test_state = env.reset()
        test_state = processor.process_state_for_memory(test_state, True)

        test_sum = 0
        test_step = 0
        while True:
            test_action, _ = learner.choose_action(processor.process_state_for_network(test_state))
            test_next_state, test_reward, test_done, _ = env.step(test_action)
            test_next_state = processor.process_state_for_memory(test_next_state, False)
            test_reward = processor.process_reward(test_reward, reward_clipping=config['reward_clipping'])

            if config['environment'] == 'CartPole-v1':
                # punish if terminal state reached
                if test_done:
                    test_reward = -test_reward

            test_state = test_next_state

            test_sum += test_reward
            test_step += 1

            if test_done:
                print('Test Episode = ' + str(e), ' epsilon =', "%.4f" % learner.epsilon,
                      ', steps = ', test_step,
                      ", total reward = ", test_sum)
                test_rewards.append(test_sum)
                test_steps.append(test_step)
                break

    # plot test-results
    test_summary.summarize(step_counts=test_steps, reward_counts=test_rewards)
    print('Sum Rewards = ', np.sum(test_rewards),
          ' Mean Reward = ', np.mean(test_rewards),
          ' Median Rewards = ', np.median(test_rewards),
          ' Standard Deviation = ', np.std(test_rewards))

    learner.epsilon = temp_epsilon
