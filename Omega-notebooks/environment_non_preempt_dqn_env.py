# %% [code] {"jupyter":{"outputs_hidden":false}}
import random

from environment_env_util import *
from environment_env import *
from environment_env_util import *
import numpy as np

__all__ = ['NonPreemptDQNEnv']



class NonPreemptDQNEnv(Env):
    def __init__(self, pa):
        super().__init__(pa)
        self.episode_reward = np.array([])

    def step(self, action):
        ''' Step function for nonpreemptive dqn case

        Parameters
        ----------
        action: ndarray
            Contains a job index, the gpu requested for this job, and a set of seleted gpus
            e.g., [0, 4, 1, 1, 1, 1, 0, 0]
            The first 0 is the job index, which is the first job in the queue, this job requests 4 gpus.
            [1, 1, 1, 1, 0, 0] means first 4 gpus are selected for this job among 6 gpus.
        '''
        done_jobs = self.get_done_jobs()
        self.remove_jobs(done_jobs)
        self.advance_runningjobs_onestep()

        j = int(action[0])
        selected_gpu = np.reshape(action[2:],
                                  (self.pa.num_racks_per_cluster, self.pa.num_machines_per_rack, self.pa.num_gpus_per_machine))
        selected_gpu = np.where(selected_gpu)
        num_selected_gpus = len(selected_gpu[0])
        if (not np.any(self.resources[selected_gpu] != -1)) and 0 <= j < len(self.jobqueue) and num_selected_gpus == action[1]:
            if self.jobqueue[j].status == 'waiting':
                self.assign_job_gpus(j, selected_gpu)
        if random.random() < self.pa.new_job_rate:
            self.insert_new_job()
        if (self.j_id >= len(self.len_seq) and len(self.jobqueue) == 0) or self.num_job_finished >= self.pa.target_num_job_arrive:
            self.done = True

        self.total_step += 1
        self.curr_time += 1
        self.episode_reward = np.append(self.episode_reward, self.reward())
        return self.done, self.reward(), self.observe()


    def observe(self):
        return observe_rltaps(self)

    def reward(self):
        return self.reward_throughput()


    def reward_slowdown(self):
        reward = 0

        for j in self.jobqueue[self.get_running_jobs()]:
            reward += self.pa.delay_penalty / float(j.job_len)

        for j in self.jobqueue[self.get_waiting_jobs()]:
            if j is not None:
                reward += self.pa.hold_penalty / float(j.job_len)

        for j in self.backlog:
            if j is not None:
                reward += self.pa.dismiss_penalty / float(j.job_len)
        return reward