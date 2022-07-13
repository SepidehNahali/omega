import copy
import numpy as np

class BaseLines:
    def __init__(self, baselines):
        self.baselines = baselines

    def get_multi_rewards(self, env):
        rewards = {}
        for scheduler in self.baselines:
            tmpenv = copy.deepcopy(env)


            while not tmpenv.done:
                if scheduler == 'SJF':
                    action = self.sjf_action(tmpenv.jobqueue)
                elif scheduler == 'RANDOM':
                    action = self.random_action(tmpenv.jobqueue)
                tmpenv.step(action)
                # print(f'{tmpenv.num_job_finished}{tmpenv.pa.target_num_job_arrive}')
            rewards[scheduler] = np.sum(tmpenv.episode_reward)
        return rewards


    def sjf_action(self, jobqueue):
        action = 0

        lst = np.arange(len(jobqueue))

        def getter(i):
            return jobqueue[i].status

        vfunc = np.vectorize(getter, otypes=[str])
        lst = np.where(vfunc(lst) == 'waiting')[0]
        if len(lst) > 0:
            w = np.array([jobqueue[j].job_len for j in lst])
            action = lst[np.argmin(w)]
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
