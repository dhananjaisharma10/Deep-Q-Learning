#!/usr/bin/env python
import os
import sys
import copy
import argparse

import gym
import keras
import numpy as np
import tensorflow as tf

from collections import deque
from keras.models import Model, load_model
from keras.layers import Input, Dense


class QNetwork():
    r"""This class essentially defines the network architecture.
    The network should take in state of the world as an input,
    and output Q values of the actions available to the agent as the output.
    """

    def __init__(self, env):
        """Define your network architecture here. It is also a good idea to
        define any training operations and optimizers here, initialize your
        variables, or alternately compile your model here.
        """

        # A neural network with 3 fully-connected layers should suffice for
        # the low-dimensional environments that we are working with.

        inp_dim = env.observation_space.shape[0]
        out_dim = env.action_space.n
        self.x = Input(shape=(inp_dim,))
        self.l1 = Dense(20, activation='relu')(self.x)
        self.l2 = Dense(10, activation='relu')(self.l1)
        self.l3 = Dense(out_dim, activation='relu')(self.l2)
        self.model = Model(inputs=self.x, outputs=self.l3)
        self.loss = 'mean_squared_error'
        self.optimizer = 'adam'
        self.model.compile(optimizer=self.optimizer,
                           loss=self.loss,
                           metrics=['accuracy'])
        self.weights_direc = './weights'
        if not os.path.exists(self.weights_direc):
            os.makedirs(self.weights_direc)

    def save_model_weights(self, suffix):
        """Helper function to save your model/weights.
        """
        self.model.save_weights('ckpt_{}.h5'.format(suffix))

    def load_model(self, model_file):
        """Helper function to load an existing model.
        """
        self.model = load_model(model_file)

    def load_model_weights(self, weight_file):
        """Helper funciton to load model weights.
        NOTE: we will use this function instead of load_model
        """
        self.model.load_weights(weights_file)


class Replay_Memory():

    def __init__(self, memory_size=50000, burn_in=10000):
        r"""The memory essentially stores transitions recorder from the agent
        taking actions in the environment.
        Burn in episodes define the number of episodes that are written
        into the memory from the randomly initialized agent. Memory size is
        the maximum size after which old elements in the memory are replaced.
        A simple (if not the most efficient) was to implement the memory is
        as a list of transitions.

        Hint: use collections.deque(maxlen=memory_size)
        """

        self.memory_size = memory_size
        self.burn_in = burn_in
        self.memory = deque(maxlen=memory_size)

    def sample_batch(self, batch_size=32):
        """This function returns a batch of randomly sampled transitions
        - i.e. state, action, reward, next state, terminal flag tuples.
        You will feed this to your model to train.
        """

        idx_batch = np.random.choice(range(len(self.memory)))
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

    def __init__(self, env, render=False):
        """Create an instance of the network itself, as well as the memory.
        Here is also a good place to set environmental parameters,
        as well as training parameters - number of episodes/iterations, etc.
        """

        self.env = env
        self.q_network = QNetwork(env)
        self.r_memory = Replay_Memory()
        self.num_episodes = 50
        self.epsilon = 0.5
        self.gamma = 0.99
        self.batch_size = 32
        self.epochs = 10

    def epsilon_greedy_policy(self, q_values):
        """Creating epsilon greedy probabilities to sample from.
        """

        max_idx = np.argmax(q_values)
        probs = np.full_like(q_values, self.epsilon / len(q_values))
        probs[max_idx] += 1 - self.epsilon
        return probs

    def greedy_policy(self, q_values):
        """Creating greedy policy for test time.
        """

        policy = np.argmax(q_values, axis=1)
        return policy

    def train(self):
        """In this function, we will train our network.
        If training without experience replay_memory, then you will interact
        with the environment in this function, while also updating your
        network parameters.

        When use replay memory, you should interact with environment here,
        and store these transitions to memory, while also updating your model.
        """

        # Initialize replay memory D to capacity burn
        self.burn_in_memory()
        for episode in range(self.num_episodes):
            # Initialize your state
            state = self.env.reset()
            done = False
            while not done:
                prev_state = state
                # Step 1: Get action based on epsilon greedy policy
                q_values = self.q_network.model.predict(
                    np.array(state).reshape(1, -1))
                probs = self.epsilon_greedy_policy(q_values)
                action = np.random.choice(np.arange(len(q_values)), p=probs)
                state, reward, done, _ = self.env.step(action)
                # Step 2: Add data for replay buffer
                transition = [np.array(prev_state).reshape(1, -1), action,
                              reward, np.array(state).reshape(1, -1)]
                self.r_memory.memory.append(transition)
                # Step 3: Sample batch for training from the replay buffer
                X = self.r_memory.sample_batch(self.batch_size)
                X = np.array(X)
                p_states = X[:, 0]
                n_states = X[:, 3]
                rewards = X[:, 2]
                q_values = self.q_network.model.predict(n_states)
                y = np.max(q_values, axis=1)  # maximum q_value
                y += rewards  # add rewards
                history = self.q_network.model.fit(p_states, y,
                                                   epochs=self.epochs,
                                                   batch_size=self.batch_size,
                                                   verbose=1)
                

    def test(self, model_file=None):
        """Evaluate the performance of your agent over 100 episodes, by
        calculating cummulative rewards for the 100 episodes. Here you
        need to interact with the environment, irrespective of whether
        you are using a memory.
        """
        num_episodes = 100
        for episode in range(num_episodes):
            print(episode)

    def burn_in_memory(self):
        """Initialize your replay memory with a burn_in number
        of episodes/transitions.
        """

        # Refer to Piazza post #226 for more details.
        while len(self.r_memory.memory) < self.r_memory.burn_in:
            state = self.env.reset()
            done = False
            while not done:
                prev_state = state
                q_values = self.q_network.model.predict(
                    np.array(state).reshape(1, -1))
                action = np.random.choice
                state, reward, done, _ = self.env.step(action)
                transition = [np.array(prev_state).reshape(1, -1), action,
                              reward, np.array(state).reshape(1, -1)]
                if len(self.r_memory.memory) == self.r_memory.burn_in:
                    break
                self.r_memory.memory.append(transition)


# NOTE: if you have problems creating video captures on servers without GUI,
# you could save and relaod model to create videos on your laptop.
def test_video(agent, env, epi):
    # Usage:
    # 	you can pass the arguments within agent.train() as:
    # 		if episode % int(self.num_episodes/3) == 0:
    #       	test_video(self, self.environment_name, episode)
    save_path = "./videos-%s-%s" % (env, epi)
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    # To create video
    env = gym.wrappers.Monitor(agent.env, save_path, force=True)
    reward_total = []
    state = env.reset()
    done = False
    while not done:
        env.render()
        action = agent.epsilon_greedy_policy(state, 0.05)
        next_state, reward, done, info = env.step(action)
        state = next_state
        reward_total.append(reward)
    print("reward_total: {}".format(np.sum(reward_total)))
    agent.env.close()


def parse_arguments():
    parser = argparse.ArgumentParser(description='Deep Q Network '
                                     'Argument Parser')
    parser.add_argument('--env', dest='env', type=str)
    parser.add_argument('--render', dest='render', type=int, default=0)
    parser.add_argument('--train', dest='train', type=int, default=1)
    parser.add_argument('--model', dest='model_file', type=str)
    return parser.parse_args()


def main(args):
    args = parse_arguments()
    environment_name = args.env
    env = gym.make(environment_name)

    # Setting the session to allow growth, so it doesn't allocate all GPU
    # memory.
    gpu_ops = tf.GPUOptions(allow_growth=True)
    config = tf.ConfigProto(gpu_options=gpu_ops)
    sess = tf.Session(config=config)

    # Setting this as the default tensorflow session.
    keras.backend.tensorflow_backend.set_session(sess)

    # You want to create an instance of the DQN_Agent class here,
    # and then train / test it.


if __name__ == '__main__':
    main(sys.argv)
