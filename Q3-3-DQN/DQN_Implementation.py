#!/usr/bin/env python
import os
import sys
import time
import argparse
import os.path as osp

import gym
import keras
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from datetime import datetime
from importlib import import_module
from collections import deque
from keras.optimizers import Adam
from keras.layers import Dense
from keras.models import load_model


class QNetwork():
    r"""This class essentially defines the network architecture.
    The network should take in state of the world as an input,
    and output Q values of the actions available to the agent as the output.
    """

    def __init__(self, env, cfg):
        """Define your network architecture here. It is also a good idea to
        define any training operations and optimizers here, initialize your
        variables, or alternately compile your model here.
        """

        # Model configuration
        inp_dim = env.observation_space.shape[0]
        out_dim = env.action_space.n
        self.model = keras.models.Sequential([
            Dense(cfg.HIDDEN_1, activation=cfg.ACTIVATION,
                  input_shape=(inp_dim,)),
            Dense(cfg.HIDDEN_2, activation=cfg.ACTIVATION),
            Dense(out_dim)
        ])
        self.loss = 'mse'
        self.alpha = cfg.LR
        self.optimizer = Adam(lr=self.alpha)
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=['accuracy'])
        self.weights_direc = cfg.WEIGHTS
        if not osp.exists(self.weights_direc):
            os.makedirs(self.weights_direc)

    def load_model(self, model_file):
        """Helper function to load an existing model.
        """
        self.model = load_model(model_file)

    def load_model_weights(self, weight_file):
        """Helper funciton to load model weights.
        NOTE: we will use this function instead of load_model
        """
        self.model.load_weights(weight_file)


class Replay_Memory():

    def __init__(self, memory_size=20000, burn_in=10000):
        r"""The memory essentially stores transitions recorder from the agent
        taking actions in the environment.
        Burn in episodes define the number of episodes that are written
        into the memory from the randomly initialized agent. Memory size is
        the maximum size after which old elements in the memory are replaced.
        A simple (if not the most efficient) was to implement the memory is
        as a list of transitions.
        """

        self.memory_size = memory_size
        self.burn_in = burn_in
        self.memory = deque(maxlen=memory_size)

    def sample_batch(self, batch_size=32):
        """This function returns a batch of randomly sampled transitions
        - i.e. state, action, reward, next state, terminal flag tuples.
        You will feed this to your model to train.
        """

        idx_batch = np.random.choice(np.arange(len(self.memory)),
                                     size=batch_size)
        assert(len(idx_batch) == batch_size)
        batch = [self.memory[i] for i in idx_batch]
        return batch

    def append(self, transition):
        """Appends transition to the memory.
        """

        self.memory.append(transition)


class DQN_Agent():
    r"""In this class, we will implement functions to do the following.
    - Create an instance of the Q Network class.
    - Create a function that constructs a policy from the Q values
        predicted by the Q Network.
        (a) Epsilon Greedy Policy.
        (b) Greedy Policy.
    - Create a function to train the Q Network, by interacting with the
        environment.
    - Create a function to test the Q Network's performance on the environment.
    - Create a function for Experience Replay.
    """

    def __init__(self, cfg, render=False):
        """Create an instance of the network itself, as well as the memory.
        Here is also a good place to set environmental parameters,
        as well as training parameters - number of episodes/iterations, etc.
        """

        self.env = gym.make(cfg.ENV)
        # self.env = gym.wrappers.Monitor(self.env, 'recording')
        self.q_network = QNetwork(self.env, cfg)
        self.r_memory = Replay_Memory(memory_size=cfg.BUFFER_SIZE)
        self.train_episodes = cfg.NUM_TRAINING_EPISODES
        self.test_episodes = cfg.NUM_TEST_EPISODES
        self.epsilon = cfg.EPSILON
        self.epsilon_min = cfg.EPSILON_MIN
        self.decay_step = (cfg.EPSILON - cfg.EPSILON_MIN) / cfg.DECAY_PERIOD
        self.gamma = cfg.GAMMA
        self.batch_size = cfg.BATCH_SIZE
        self.epochs = cfg.EPOCHS
        self.interval = cfg.INTERVAL
        self.weights_direc = cfg.WEIGHTS
        self.test_rewards = list()
        self.td_errors = list()
        self.early_stopping = deque(maxlen=cfg.EARLY_STOPPING)

    def epsilon_greedy_policy(self, q_values):
        """Return an action based on the epsilon greedy policy
        """

        if np.random.uniform() < self.epsilon:
            return np.random.randint(0, self.env.action_space.n)
        else:
            return np.argmax(q_values)

    def save_model_weights(self, directory, episode):
        """Save the weights of a model
        """
        filepath = osp.join(directory, 'ckpt_{}.h5'.format(episode))
        self.q_network.model.save_weights(filepath)

    def train(self):
        """In this function, we will train our network.
        If training without experience replay_memory, then you will interact
        with the environment in this function, while also updating your
        network parameters.
        When using replay memory, you should interact with environment here,
        and store these transitions to memory, while also updating your model.
        """

        def get_run_id():
            # A unique ID for a training session
            dt = datetime.now()
            run_id = dt.strftime('%m_%d_%H_%M')
            return run_id

        print('*' * 10, 'TRAINING', '*' * 10)
        # Create directory for this run_id
        run_id = get_run_id()
        self.model_weights = osp.join(self.weights_direc, run_id)
        if not osp.exists(self.model_weights):
            os.makedirs(self.model_weights)

        start = time.time()
        # Initialize replay memory
        self.burn_in_memory()
        episode_rewards = list()
        best_reward = 0
        training = True
        for episode in range(self.train_episodes):
            r = list()  # cummulative reward for current episode
            done = False
            state = self.env.reset()
            td_error = 0
            it = 0
            while not done:
                it += 1
                prev_state = state
                # Step 1: Get action based on epsilon greedy policy
                q_values = self.q_network.model.predict(
                    np.array(prev_state).reshape(1, -1))
                q_values = q_values[0]
                action = self.epsilon_greedy_policy(q_values)
                state, reward, done, _ = self.env.step(action)
                r.append(reward)
                # Step 2: Add data for replay buffer
                transition = [prev_state, action, reward, state, done]
                self.r_memory.memory.append(transition)
                # Step 3: Sample batch for training from the replay buffer
                X = self.r_memory.sample_batch(self.batch_size)
                p_states = np.array([x[0] for x in X])
                n_states = np.array([x[3] for x in X])
                actions = [x[1] for x in X]
                rewards = [x[2] for x in X]
                dones = np.asarray([x[4] for x in X])
                dones = 1 - dones  # for handling terminal states
                # Step 4: target q (current) = r + gamma * max q (next)
                q_net = self.q_network.model.predict(p_states)
                q_target = self.q_network.model.predict(n_states)
                q_target = np.max(q_target, axis=1)  # greedy policy
                q_target *= (self.gamma * dones)
                q_target += rewards
                q_curr = q_net[range(self.batch_size), actions]
                q_net[range(self.batch_size), actions] = q_target
                td_error += np.mean(np.absolute(q_target - q_curr))
                if training:
                    history = self.q_network.model.fit(
                        p_states, q_net, epochs=self.epochs,
                        batch_size=self.batch_size, verbose=0)
            # epsilon decay
            if self.epsilon > self.epsilon_min:
                self.epsilon -= self.decay_step
            # TD errors
            self.td_errors.append(td_error / it)
            end = time.time()
            sum_of_reward = sum(r)
            # Early stopping criterion
            self.early_stopping.append(sum_of_reward)
            if len(self.early_stopping) == self.early_stopping.maxlen:
                if int(np.mean(self.early_stopping)) == 200:
                    training = False
            episode_rewards.append(sum_of_reward)
            print('Reward for episode {}: {:.2f} | Time elapsed: {:.2f}'
                  ' mins'.format(episode, sum_of_reward, (end - start) / 60))
            # Create video at every (train_episodes/3)th episode
            if episode % (self.train_episodes // 3) == 0:
                videopath = osp.join(self.model_weights,
                                     'video_{}'.format(episode))
                if not osp.exists(videopath):
                    os.makedirs(videopath)
                test_video(self.q_network.model, self.env, videopath)
            # Save weights if highest reward so far and evaluate
            if best_reward < sum_of_reward or episode % self.interval == 0:
                best_reward = sum_of_reward
                self.save_model_weights(self.model_weights, episode)
                # Evaluate after every cfg.INTERVALth episode
                if episode % self.interval == 0:
                    self.test()
                # Plot TD error
                plt.figure()
                plt.plot(self.td_errors)
                plt.xlabel('#episodes')
                plt.ylabel('td error')
                tdpath = osp.join(self.model_weights, 'td_error.png')
                plt.savefig(tdpath, dpi=400, bbox_inches='tight')
                plt.close()

    def test(self):
        """Evaluate the performance of your agent over 100 episodes, by
        calculating cummulative rewards for the 100 episodes. Here you
        need to interact with the environment, irrespective of whether
        you are using a memory.
        """

        print('*'*10, 'EVALUATION', '*'*10)
        rewards = []
        for episode in range(self.test_episodes):
            state = self.env.reset()
            done = False
            r = list()
            while not done:
                q_values = self.q_network.model.predict(
                    np.array(state).reshape(1, -1))
                action = np.argmax(q_values)  # greedy policy
                state, reward, done, _ = self.env.step(action)
                r.append(reward)
            rewards.append(sum(r))
        self.test_rewards.append(np.mean(rewards))
        filepath = osp.join(self.model_weights, 'test_rewards.npy')
        np.save(filepath, self.test_rewards)
        # Plot test rewards
        plt.figure()
        episodes = np.arange(len(self.test_rewards)) * 100
        plt.plot(episodes, self.test_rewards)
        plt.xlabel('#episodes')
        plt.ylabel('reward')
        imagepath = osp.join(self.model_weights, 'eval.png')
        plt.savefig(imagepath, bboxes_inches='tight', dpi=400)
        plt.close()

    def burn_in_memory(self):
        """Initialize your replay memory with a burn_in number
        of episodes/transitions.
        """

        while len(self.r_memory.memory) < self.r_memory.burn_in:
            state = self.env.reset()
            done = False
            while not done:
                prev_state = state
                # Randomly choose an action
                action = np.random.choice(self.env.action_space.n)
                state, reward, done, _ = self.env.step(action)
                transition = [prev_state, action, reward, state, done]
                self.r_memory.memory.append(transition)
                if len(self.r_memory.memory) == self.r_memory.burn_in:
                    break
        assert(len(self.r_memory.memory) == self.r_memory.burn_in)


# NOTE: if you have problems creating video captures on servers without GUI,
# you could save and relaod model to create videos on your laptop.
def test_video(model, env, save_path):
    env = gym.wrappers.Monitor(env, save_path)
    reward_total = list()
    state = env.reset()
    done = False
    while not done:
        env.render()
        q_values = model.predict(np.array(state).reshape(1, -1))
        action = np.argmax(q_values)
        state, reward, done, _ = env.step(action)
        reward_total.append(reward)
    print("reward_total: {}".format(np.sum(reward_total)))
    env.close()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network '
                                     'Argument Parser')
    parser.add_argument('--env', dest='env',
                        help='Configuration file of the environment')
    return parser.parse_args()


def main(args):
    args = parse_arguments()
    filename = args.env[:-3]
    cfg = import_module(filename)
    # Setting the session to allow growth, so it doesn't allocate all GPU
    # memory.
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    # Setting this as the default tensorflow session.
    keras.backend.tensorflow_backend.set_session(sess)

    agent = DQN_Agent(cfg)
    agent.train()


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    main(sys.argv)
