import copy
import numpy as np
import sys


class BaseLines:
    def __init__(self, baselines, clip_range=100):
        self.baselines = baselines
        self.clip_range = clip_range

    def get_multi_rewards(self, env):
        rewards = {}
        for scheduler in self.baselines:
            tmpenv = copy.deepcopy(env)
            i = 0
            while not tmpenv.done:
                if scheduler == 'SJF':
                    action = self.sjf_action(tmpenv.jobqueue)
                elif scheduler == 'RANDOM':
                    action = self.random_action(tmpenv.jobqueue)
                elif scheduler == 'LJF':
                    action = self.ljf_action(tmpenv.jobqueue)
                #print(f'{scheduler} : {len(tmpenv.get_avl_gpus()[0])}')
                tmpenv.step(action)
                i += 1
            print(f'{scheduler} : {np.mean(tmpenv.episode_reward[:self.clip_range])} {len(tmpenv.episode_reward)}')

            rewards[scheduler] = np.mean(tmpenv.episode_reward[:self.clip_range])
        return rewards


    def sjf_action(self, jobqueue):
        action = 0

        if len(jobqueue) > 0:
            lst = np.array([j.job_len for j in jobqueue])

            def getter(j):
                return j.status
            vfunc = np.vectorize(getter, otypes=[str])
            runnings = np.where(vfunc(jobqueue) == 'running')
            waitings = np.where(vfunc(jobqueue) == 'waiting')
            lst[runnings] = 1000
            action = np.argmin(lst)

        return action

    def ljf_action(self, jobqueue):
        action = 0

        if len(jobqueue) > 0:
            lst = np.array([j.job_len for j in jobqueue])

            def getter(j):
                return j.status

            vfunc = np.vectorize(getter, otypes=[str])
            runnings = np.where(vfunc(jobqueue) == 'running')
            waitings = np.where(vfunc(jobqueue) == 'waiting')
            lst[runnings] = -1
            action = np.argmax(lst)

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


if __name__ == '__main__':

    from environment.simple_dqn_env import *
    from env_components.parameters import *

    pa = Parameters(num_racks_per_cluster=2, num_machines_per_rack=2, num_gpus_per_machine=4,
                    max_gpu_request=4,
                    max_job_len=20
                    )
    env = DQNTesting1(pa)
    env.generate_job_sequence(20)
    baselines = BaseLines(['SJF'])
    for i in range(5):
        print('-' * 50)
        a = baselines.random_action(env.jobqueue)
        print(f'jobqueue {[j.job_len for j in env.jobqueue]}')
        print(f'action: {a} {[j.job_len for j in env.jobqueue[env.get_waiting_jobs()]]}')

        env.step(a)
