import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np
import random
from torch import nn
import torch.nn.functional as F
import statistics
import timeit


from minipong import Minipong
pong = Minipong(level=3, size=5)

def plot_res(values, title='', save=False, i=0):
    ''' Plot the reward curve and histogram of results over time.'''

    # Define the figure
    f, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    f.suptitle(title)
    ax[0].plot(values, label='score per run')
    ax[0].axhline(300, c='red', ls='--', label='goal')
    ax[0].set_xlabel('Episodes')
    ax[0].set_ylabel('Reward')
    x = range(len(values))
    ax[0].legend()
    # Calculate the trend
    try:
        z = np.polyfit(x, values, 1)
        p = np.poly1d(z)
        ax[0].plot(x, p(x), "--", label='trend')
    except:
        pass

    # Plot the histogram of results
    ax[1].hist(values[-50:])
    ax[1].axvline(300, c='red', label='goal')
    ax[1].set_xlabel('Scores per Last 500 Episodes')
    ax[1].set_ylabel('Frequency')
    ax[1].legend()
    if save:
        plt.savefig('perf.png', dpi=300)
    if i % 500 == 0:
        plt.show()
    plt.close()


class DQN(nn.Module):
    """Deep Q Neural Network class. """

    def __init__(self, state_dim, action_dim, hidden_dim=64, lr=0.0001):
        super(DQN, self).__init__()

        # define architecture
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)

        self.criterion = torch.nn.MSELoss()
        self.optimiser = torch.optim.Adam(self.parameters(), lr)

    def forward(self, x):
        # forward propogation
        # x --> fc1 --> leaky relu --> fc2 --> y
        x = F.leaky_relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def update(self, state, q_values):
        """Update the weights of the network given a training sample. """

        q_value_pred = self(torch.Tensor(state))  # put state in and get q-value to determine actions
        loss = self.criterion(q_value_pred, Variable(torch.Tensor(q_values)))
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

    def predict(self, state):
        """ Compute the output of the network given a state. """
        with torch.no_grad():
            return self(torch.Tensor(state))



def q_learning(pong, model, episodes, gamma, epsilon, eps_decay, title='DQL', verbose=True):
    """Deep Q Learning algorithm using the DQN. """
    final = []
    for i, episode in enumerate(range(episodes), start=1):
        # Reset state
        state = pong.reset()

        done = False
        total = 0
        while not done:
            # Implement greedy search policy to explore the state space
            if random.random() < epsilon:
                action = pong.sampleaction()
            else:
                q_values = model.predict(state)
                action = torch.argmax(q_values).item()

            # Take action and add reward to total
            next_state, reward, done = pong.step(action)

            # Update total reward
            total += reward
            q_values = model.predict(state).tolist()

            if done:
                q_values[action] = reward
                model.update(state, q_values)
                break

            q_values_next = model.predict(next_state)
            q_values[action] = reward + gamma * torch.max(q_values_next).item()
            model.update(state, q_values)
            # Update network weights
            # end of your code here
            state = next_state
        # Update epsilon
        epsilon = max(epsilon * eps_decay, 0.01)
        final.append(total)
        plot_res(final, title, i=i)

        if verbose:
            print("episode: {}, total reward: {}".format(i, total))
    return final


# Number of states
n_state = pong.observationspace()
# Number of actions
n_action = 3
# Number of episodes
episodes = 500
# Number of hidden nodes in the DQN
n_hidden = 150
# Learning rate
lr = 0.001

start = timeit.default_timer()

dqn = DQN(n_state, n_action, n_hidden, lr)
model = q_learning(pong, dqn, episodes, gamma=0.95, epsilon=0.6, eps_decay=0.99)

stop = timeit.default_timer()
timeTaken = stop - start
print('\nFinished Training')
print('Training time: ', timeTaken)

rewards = []
def simulate(pong, model):
    for _ in range(50):
        done = False
        state = pong.reset()

        total = 0

        while done is False:
            q_values = model.predict(state)
            action = torch.argmax(q_values).item()

            _, reward, done = pong.step(action)
            total += reward
            #update total reward
        rewards.append(total)
    print(len(rewards))
    test_average = sum(rewards)/ len(rewards)
    test_std_dev = statistics.stdev(rewards)
    print('Test Average:',test_average,'Test Standard Deviation:',test_std_dev)


simulate(pong=pong, model=dqn)





