#python3

import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation


class Experiences:
    """
    A class for managing an experience replay buffer
    TODO Implement this
    """
    def __init__(self, size):
        pass

    def add(self, exp):
        pass

    def get(self, num):
        pass


class Env:
    def __init__(self):
        self.env = gym.make('CartPole-v1')
        self.state = self.env.reset()
        self.score = 0

    def actions(self):
        return self.env.action_space

    def next(self, action):
        # TODO Should we take random actions occasionally to further explore?

        observation, reward, done, _info = self.env.step(action)
        self.score += reward # TODO Discounted reward?

        exp = (self.state, action, self.score, observation)
        self.state = observation

        if (done):
            self.state = self.env.reset()
            self.score = 0

        return exp


class NN:
    def __init__(self):
        """
        The model:
        Given the current state, predict the maximum future reward for both possible actions
        """
        self.model = Sequential()
        self.model.add(Dense(10, activation='relu', input_dim=4)) # TODO Input dims?
        self.model.add(Dense(10, activation='relu'))
        self.model.add(Dense(2, activation='softmax'))
        self.model.compile(optimizer='adam', loss='mse')

    def fit(self, data, labels):
        return self.model.fit(data, labels, epochs=10, batch_size=data.shape[1])

    def predict(self, data):
        return self.model.predict(data)


epoch_size = 32
epoch_num = 1024
epoch_updates = 4
experience_size = 1024

if __name__ == '__main__':
    env = Env()
    model = NN()
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
