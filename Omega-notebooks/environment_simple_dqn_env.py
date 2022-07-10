import random

from environment_env import *
from environment_env_util import *
import numpy as np


# __all__ = ['DQNTesting1']

# class DQNTesting1(Env):
#     def __init__(self, pa):
#         super().__init__(pa)

#     def step(self, action):
#         ''' Step function for nonpreemptive dqn case

#         Parameters
#         ----------
#         action: int
#             The index of the job in the jobqueue
#         '''
#         allocate = False
#         j = action
#         if j < len(self.jobqueue) and len(self.jobqueue) > 0:
#             selected_gpu = self.random_select_k_gpus_for_job(j)
#             if all(x == -1 for x in self.resources[selected_gpu]) and self.jobqueue[j].status == 'waiting' and len(selected_gpu[0]) > 0:
#                 self.assign_job_gpus(j, selected_gpu)
#                 job = self.jobqueue[j]
#                 allocate = True
#         if not allocate:
#             if np.random.random() < self.pa.new_job_rate:
#                 self.insert_new_job()

#         if self.num_job_finished >= self.pa.target_num_job_arrive:
#             self.done = True
#         # print(f'{self.num_job_finished} {len(self.jobqueue)} {len(self.backlog)}')
#         self.total_step += 1
#         self.episode_reward = np.append(self.episode_reward, self.reward())
#         done_jobs = self.get_done_jobs()

#         self.remove_jobs(done_jobs)

#         self.advance_runningjobs_onestep()
#         return self.observe(), self.reward(), self.done


#     def observe(self):
#         height = self.pa.max_job_len
#         width = self.jobqueue_maxlen * (self.pa.max_gpu_request)
#         #width = self.resources.size + self.jobqueue_maxlen * self.pa.max_gpu_request
#         image = np.zeros((2, height, width))
#         #image[:, 0: self.resources.size] = self.get_cluster_canvas()[:, :]
#         #pt = self.resources.size
#         pt = 0
#         for j in self.jobqueue:
#             if j.status == 'waiting':
#                 image[0, : j.job_len, pt: pt + j.gpu_request] = j.m
#                 image[1, : j.job_len, pt: pt + j.gpu_request] = j.waiting_time
#             pt += self.pa.max_gpu_request

#         # return np.expand_dims(image, axis=2)
#         return image

#     def get_cluster_canvas(self):
#         image = np.zeros((self.pa.max_job_len, self.resources.size))
#         gpus = self.resources.reshape(1, self.resources.size)
#         used = np.where(gpus != -1)
#         for i, j in zip(used[0], used[1]):
#             # print(f'{get_j_idx_by_id(gpus[i,j], jobqueue)} {gpus[i,j]}')
#             j_idx = self.get_j_idx_by_id(gpus[i, j])[0][0]
#             image[0:self.jobqueue[j_idx].job_len - self.jobqueue[j_idx].progress, j] = 1
#         # plt.imshow(image)
#         # plt.show()
#         return image

#     def get_j_idx_by_id(self, id):
#         def getter(j):
#             return j.job_id

#         vfunc = np.vectorize(getter, otypes=[int])
#         idx = np.where(vfunc(self.jobqueue) == id)
#         # for idx in range(len(jobqueue)):
#         #     if jobqueue[idx].job_id == id:
#         #         return idx
#         return idx


#     def reward(self):
#         # return self.reward_throughput()
#         return self.reward_throughput()

#     def testing_reward(self):
#         reward = 0
#         for j in self.jobqueue[self.get_running_jobs()]:
#             reward += 1
#         for j in self.jobqueue[self.get_waiting_jobs()]:
#             reward -= 1

#         return reward

#     def get_done_jobs_test(self):
#         """Get the index of finished jobs

#         Returns
#         -------
#         done_jobs :
#             Thie index of finished jobs
#         """

#         def getter(j):

#             return j.progress >= j.job_len

#         vfunc = np.vectorize(getter, otypes=[bool])
#         done_jobs = np.where(vfunc(self.jobqueue))
#         return done_jobs

__all__ = ['DQNTesting1']

class DQNTesting1(Env):
    def __init__(self, pa):
        super().__init__(pa)

    def step(self, action):
        ''' Step function for nonpreemptive dqn case

        Parameters
        ----------
        action: int
            The index of the job in the jobqueue
        '''
        done_jobs = self.get_done_jobs()
        self.remove_jobs(done_jobs)
        self.advance_runningjobs_onestep()

        j = action
        if j < len(self.jobqueue) and len(self.jobqueue) > 0:
            selected_gpu = self.random_select_k_gpus_for_job(j)
            if all(x == -1 for x in self.resources[selected_gpu]) and self.jobqueue[j].status == 'waiting':
                self.assign_job_gpus(j, selected_gpu)
        if random.random() < self.pa.new_job_rate:
                # (self.num_job_finished + len(self.jobqueue) + len(self.backlog)) < self.pa.target_num_job_arrive:
            self.insert_new_job()
        if self.num_job_finished >= self.pa.target_num_job_arrive:
            self.done = True
        # print(f'{self.num_job_finished} {len(self.jobqueue)} {len(self.backlog)}')
        self.total_step += 1
        self.episode_reward = np.append(self.episode_reward, self.reward())
        return self.observe(), self.reward(), self.done


    def observe(self):
        # print('self.resources: ',self.resources)
        height = self.pa.max_job_len
        # width = self.jobqueue_maxlen * (self.pa.max_gpu_request)
        width = self.resources.size + self.jobqueue_maxlen * self.pa.max_gpu_request
        image = np.zeros((height, width))
        image[:, 0: self.resources.size] = self.get_cluster_canvas()[:, :]
        pt = self.resources.size
        for j in self.jobqueue:
            if j.status == 'waiting':
                image[: j.job_len, pt: pt + j.gpu_request] = 1
            pt += self.pa.max_gpu_request

        return np.expand_dims(image, axis=2)

    def get_cluster_canvas(self):
        image = np.zeros((self.pa.max_job_len, self.resources.size))
        gpus = self.resources.reshape(1, self.resources.size)
        used = np.where(gpus != -1)
        # print(' Used GPUs index ',        used )
        for i, j in zip(used[0], used[1]):
            # print(f'{get_j_idx_by_id(gpus[i,j], jobqueue)} {gpus[i,j]}')
            j_idx = self.get_j_idx_by_id(gpus[i, j])[0][0]
            image[0:self.jobqueue[j_idx].job_len - self.jobqueue[j_idx].progress, j] = 1
        # plt.imshow(image)
        # plt.show()
        return image

    def get_j_idx_by_id(self, id):
        def getter(j):
            return j.job_id

        vfunc = np.vectorize(getter, otypes=[int])
        idx = np.where(vfunc(self.jobqueue) == id)
        # for idx in range(len(jobqueue)):
        #     if jobqueue[idx].job_id == id:
        #         return idx
        return idx


    def reward(self):
        return self.reward_throughput()