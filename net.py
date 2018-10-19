#python3

import random

import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout


class Experiences:
    """
    A class for managing an experience replay buffer.
    """
    def __init__(self, size):
        self.buf = []
        self.max = size

    def add(self, exp):
        """
        Add exp to the buffer, evicting another item if the buffer is full.
        """
        if (len(self.buf) < self.max):
            self.buf.append(exp)
        else:
            pos = random.randrange(0, len(self.buf))
            self.buf[pos] = exp

    def get(self, num):
        """
        Get num elements from the buffer. num should probably be less than size.
        """
        return random.sample(self.buf, num)


class Env:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.state = self.env.reset()
        self.score = 0

    def actions(self):
        return self.env.action_space

    def next(self, action):
        observation, reward, done, _info = self.env.step(action)
        self.score += reward # TODO Discounted reward?

        exp = (self.state, action, self.score, observation)
        self.state = observation

        if (done):
            self.state = self.env.reset()
            self.score = 0

        return exp


class NN:
    """
    The model: Given the current state, predict the maximum future reward for both possible actions.
    """
    def __init__(self, dropout_rate):
        self.model = Sequential()
        self.model.add(Dense(10, activation='relu', input_dim=4))
        self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dropout(dropout_rate))
        self.model.add(Dense(2, activation='softmax'))
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, data, labels):
        """
        data: shape (x, 4)
        labels: shape (x, 2)
        x must be the same for both and is the number of samples to fit.
        """
        return self.model.fit(data, labels, epochs=10, batch_size=data.shape[1])

    def predict(self, data):
        """
        data: shape (x, 4) where x is the number of samples to predict.
        """
        return self.model.predict(data)


# Hyperparameters
epoch_size = 32
epoch_num = 1024
epoch_updates = 4
experience_size = 1024
learning_rate = 0.99
dropout_rate = 0.5
exploration_rate = 0.9


if __name__ == '__main__':
    env = Env()
    model = NN(dropout_rate)
    exp = Experiences(experience_size)

    # Prime the exp replay buffer
    for i in range(epoch_size):
        pass

    for epoch in range(epoch_num):
        # Create new experiences and add them to the buffer
        for i in range(epoch_updates):
            # TODO Print these in a human readable way too
            pass

        data, labels = exp.get(epoch_size)
        model.fit(data, labels)
