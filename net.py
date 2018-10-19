#python3

import random

import gym
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout


# Hyperparameters
epoch_size = 128
epoch_num = 1024
epoch_updates = 16
experience_size = 1024
learning_rate = 0.95
# discount_rate = 0.98
dropout_rate = 0.5
# exploration_rate = 0.9


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
        self.stats = []

    def actions(self):
        return self.env.action_space

    def next(self, action):
        observation, reward, done, _info = self.env.step(action)
        # self.env.render()
        self.score += reward # TODO Discounted reward?

        exp = (self.state, [action], [self.score], observation)
        self.state = observation

        if (done):
            self.stats.append(self.score)
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

    def train(self, data, labels):
        """
        data: shape (x, 4)
        labels: shape (x, 2)
        x must be the same for both and is the number of samples to fit.
        """
        return self.model.train_on_batch(data, labels)

    def evaluate(self, data, labels):
        """
        The same arguments as for fit, but we evaluate the model instead of training it.
        """
        return self.model.evaluate(data, labels, verbose=0)

    def predict(self, data):
        """
        data: shape (x, 4) where x is the number of samples to predict.
        """
        return self.model.predict(data, verbose=0)


def new_sample(env, model, exp):
    state = env.state
    action = np.argmax(model.predict(np.array([state])))
    update = env.next(action)
    exp.add(update)
    return update


def calc_labels(start, action, reward, end, model):
    """
    start: shape (x,4)
    action: shape (x, 1)
    reward: shape (x, 1)
    end: shape (x, 4)
    returns: data, labels, the pair used to train the model.
    """
    target = model.predict(start) # TODO Batch these two together
    Qp = model.predict(end)
    value = reward + learning_rate * np.max(Qp, axis=1, keepdims=True)
    np.put_along_axis(target, action, value, axis=1)

    return start, target


def test(env, model, exp):
    # Create new experiences, test with them, add them to the buffer
    updates = [new_sample(env, model, exp) for i in range(epoch_updates)]
    start, action, reward, end = map(np.array, zip(*updates))
    data, labels = calc_labels(start, action, reward, end, model)
    return model.evaluate(data, labels)


def train(env, model, exp):
    start, action, reward, end = map(np.array, zip(*exp.get(epoch_size)))
    data, labels = calc_labels(start, action, reward, end, model)
    model.train(data, labels)


if __name__ == '__main__':
    accuracy = []
    env = Env()
    model = NN(dropout_rate)
    exp = Experiences(experience_size)

    # Prime the exp replay buffer
    for i in range(epoch_size):
        action = env.actions().sample()
        exp.add(env.next(action))

    for epoch in range(epoch_num):
        train(env, model, exp)
        acc = test(env, model, exp)
        print("Accuracy on epoch %d: %f" % (epoch, acc))
        accuracy.append(acc)

    print(env.stats)