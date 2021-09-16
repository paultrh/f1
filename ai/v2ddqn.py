from random import random
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from ai.SumTree import SumTree



class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims,
                 n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions
        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, observation):
        state = T.Tensor(observation).to(self.device)
        #observation = observation.view(-1)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)
        return actions

class AgentV2(object):
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=1000000, eps_end=0.05, eps_dec=0.0001):

        self.losses = []
        self.GAMMA = gamma
        self.EPSILON = epsilon
        self.EPS_MIN = eps_end
        self.EPS_DEC = eps_dec
        self.lr = lr
        self.action_space = [i for i in range(n_actions)]
        self.n_actions = n_actions
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0

        self.Q_eval = DeepQNetwork(lr, n_actions=self.n_actions,
                              input_dims=input_dims, fc1_dims=32, fc2_dims=32)
        self.Q_eval.apply(self.weights_init)

        self.Q_next = DeepQNetwork(lr, n_actions=self.n_actions,
                                   input_dims=input_dims, fc1_dims=32, fc2_dims=32)
        self.Q_next.apply(self.weights_init)

        self.state_memory = np.zeros((self.mem_size, *input_dims))
        self.new_state_memory = np.zeros((self.mem_size, *input_dims))
        self.action_memory = np.zeros((self.mem_size, self.n_actions), dtype=np.uint8)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.uint8)

        self.replace_target_cnt = 2000
        self.learn_step_counter = 0


    # weight xavier initialize
    def weights_init(self, m):
        classname = m.__class__.__name__
        if classname.find('Linear') != -1:
            nn.init.xavier_uniform(m.weight)


    def storeTransition(self, state, action, reward, state_, terminal):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        actions = np.zeros(self.n_actions)
        actions[action] = 1.0
        self.action_memory[index] = actions
        self.reward_memory[index] = reward
        self.new_state_memory[index] = state_
        self.terminal_memory[index] = 1 - terminal

        self.mem_cntr += 1

    def replace_target_network(self):
        if self.replace_target_cnt is not None and \
                self.learn_step_counter % self.replace_target_cnt == 0:
            self.Q_next.load_state_dict(self.Q_eval.state_dict())

    def load_model(self, path):
        self.Q_eval.load_state_dict(T.load(path))
        self.Q_eval.eval()

    def chooseAction(self, observation, no_rnd=False):
        rand = np.random.random()
        rd = False
        actions = self.Q_eval.forward(observation)
        if rand > self.EPSILON or no_rnd:
            action = T.argmax(actions).item()
        else:
            rd = True
            action = np.random.choice(self.action_space)
        return action, rd

    def decay_epsilon(self):
        self.EPSILON = self.EPSILON - (self.EPSILON * self.EPS_DEC) if self.EPSILON > \
                                                      self.EPS_MIN else self.EPS_MIN

    def learn(self):
        if self.mem_cntr > self.batch_size:
            self.Q_eval.optimizer.zero_grad()

            max_mem = self.mem_cntr if self.mem_cntr < self.mem_size \
                                    else self.mem_size

            batch = np.random.choice(max_mem, self.batch_size, replace=False)
            state_batch = self.state_memory[batch]
            action_batch = self.action_memory[batch]
            action_values = np.array(self.action_space, dtype=np.int32)
            action_indices = np.dot(action_batch, action_values)
            reward_batch = self.reward_memory[batch]
            new_state_batch = self.new_state_memory[batch]
            terminal_batch = self.terminal_memory[batch]

            reward_batch = T.Tensor(reward_batch).to(self.Q_eval.device)
            terminal_batch = T.Tensor(terminal_batch).to(self.Q_eval.device)

            q_eval = self.Q_eval.forward(state_batch).to(self.Q_eval.device)
            #q_target = self.Q_eval.forward(state_batch).to(self.Q_eval.device)
            q_target = q_eval.clone()
            q_next = self.Q_next.forward(new_state_batch).to(self.Q_eval.device)

            batch_index = np.arange(self.batch_size, dtype=np.int32)
            q_target[batch_index, action_indices] = reward_batch + \
                                self.GAMMA*T.max(q_next, dim=1)[0]*terminal_batch

            loss = self.Q_eval.loss(q_target, q_eval).to(self.Q_eval.device)
            self.losses.append(loss.item())
            loss.backward()
            for param in self.Q_eval.parameters():
                param.grad.data.clamp_(-1, 1)
            self.Q_eval.optimizer.step()

            self.replace_target_network()

            self.learn_step_counter += 1
