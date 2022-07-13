import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

from dqn_model import *
from utils_replay_buffer import *
from baselines import *

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor

class DQNAgent:

    def __init__(self, env, learning_rate=3e-4, tau=0.01, gamma=0.99,
                 buffer_size=10000, target_update_freq=4, with_mask=False, soft_update=True):
        self.env = env
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.tau= tau
        self.replay_buffer = ReplayBuffer(buffer_size, frame_history_len=1)
        self.target_update_freq = target_update_freq
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if with_mask:
            self.num_actions = env.jobqueue_maxlen + 1
        else:
            self.num_actions = env.jobqueue_maxlen
        self.model = Dueling_DQN(env.observe().shape, self.num_actions).type(dtype)
        self.target_model = Dueling_DQN(env.observe().shape, self.num_actions).type(dtype)

        for target_param, param in zip(self.model.parameters(), self.target_model.parameters()):
            target_param.data.copy_(param)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.with_mask = with_mask
        self.soft_update = soft_update


    def get_action(self, states, eps=0.20):
        r = np.random.random()
        if (r < eps):
            if self.with_mask:
                random_tensor = torch.rand(self.num_actions).type(dtype)

                action = self.masked_action(random_tensor)
            else:
                action = torch.IntTensor([[np.random.randint(self.num_actions)]])[0][0]

        else:
            obs = torch.from_numpy(states).unsqueeze(0).type(dtype)
            with torch.no_grad():
                q_value_all_actions = self.model(Variable(obs))
            if self.with_mask:
                action = self.masked_action(q_value_all_actions[0])
            else:

            # running = self.env.get_waiting_jobs()
            # mask = np.zeros(self.num_actions)
            # mask[running] = 1
            # mask[-1] = 1
            # inv_mask = 1 - mask
            # mask = torch.tensor(mask, dtype=torch.float).type(dtype)
            # masked = mask.mul(q_value_all_actions[0])
            # masked[inv_mask == 1] = -np.inf
            # action = masked.data.max(0)[1]
            # print(action)
                action = ((q_value_all_actions).data.max(1)[1])[0]
        return action

    def compute_loss(self, batch):
        obs_t, act_t, rew_t, obs_tp1, done_mask = batch
        obs_t = Variable(torch.from_numpy(obs_t)).type(dtype)
        act_t = Variable(torch.from_numpy(act_t)).type(dlongtype)
        rew_t = Variable(torch.from_numpy(rew_t)).type(dtype)
        obs_tp1 = Variable(torch.from_numpy(obs_tp1)).type(dtype)
        done_mask = Variable(torch.from_numpy(done_mask)).type(dtype)

        q_values = self.model(obs_t)
        q_s_a = q_values.gather(1, act_t.unsqueeze(1))
        q_s_a = q_s_a.squeeze()

        q_tp1_values = self.target_model(obs_tp1).detach()
        _, a_prime = q_tp1_values.max(1)

        # # get Q values from frozen network for next state and chosen action
        # # Q(s',argmax(Q(s',a', theta_i), theta_i_frozen)) (argmax wrt a')
        # q_target_tp1_values = self.target_model(obs_tp1).detach()
        # q_target_s_a_prime = q_target_tp1_values.gather(1, a_prime.unsqueeze(1))
        # q_target_s_a_prime = q_target_s_a_prime.squeeze()
        #
        # # if current state is end of episode, then there is no next Q value
        # q_target_s_a_prime = (1 - done_mask) * q_target_s_a_prime

        exp_q = rew_t + (1-done_mask) * self.gamma * a_prime
        loss = F.mse_loss(q_s_a, exp_q)
        return loss

    def update(self, batch_size, t):
        batch = self.replay_buffer.sample(batch_size)
        loss= self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # target network update
        if self.soft_update:
            for target_param, param in zip(self.target_model.parameters(), self.model.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        else:
            if t % self.target_update_freq == 0:
                self.target_model.load_state_dict(self.model.state_dict())

    def masked_action(self, action_tensor):
        running = self.env.get_waiting_jobs()
        mask = np.zeros(self.num_actions)
        mask[running] = 1
        mask[-1] = 1
        inv_mask = 1 - mask
        mask = torch.tensor(mask, dtype=torch.float).type(dtype)
        masked = mask.mul(action_tensor)
        masked[inv_mask == 1] = -np.inf
        action = masked.data.max(0)[1]
        return action

def random_action(self, jobqueue):
    action = 0

    lst = np.arange(len(jobqueue))

    def getter(i):
        return jobqueue[i].status
    vfunc = np.vectorize(getter, otypes=[str])
    lst = np.where(vfunc(lst) == 'waiting')[0]
    if len(lst) > 0:
        action = np.random.choice(lst)
    return action
