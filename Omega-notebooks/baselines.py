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
                elif scheduler == 'LJF':
                    action = self.ljf_action(tmpenv.jobqueue)
                elif scheduler == 'RANDOM':
                    action = self.random_action(tmpenv.jobqueue)
                elif scheduler == 'TETRIS':
                    action = self.tetris_action(tmpenv.resources, tmpenv.jobqueue)
                # if scheduler == 'SJF':
                #     action = self.sjf_action(tmpenv.jobqueue)
                # elif scheduler == 'RANDOM':
                #     action = self.random_action(tmpenv.jobqueue)
                o,r,d=tmpenv.step(action)

            rewards[scheduler] = np.sum(tmpenv.episode_reward)
        return rewards
    def tetris_action(self, resources, jobqueue):
        num_available_gpus = np.count_nonzero(resources == -1)

        jobs_status = [job.status for job in jobqueue]
        jobs_gpu_request = [job.gpu_request for job in jobqueue]

        indices_possible = [i for i in range(len(jobs_status)) if (jobs_status[i] != 'running') and (jobs_gpu_request[i] <= num_available_gpus)]

        if len(indices_possible) == 0:
            action = len(jobqueue)  # if no action available, hold
        else:
            gpu_request_max = max([jobs_gpu_request[i] for i in indices_possible])
            action = jobs_gpu_request.index(gpu_request_max)

        # if len(jobqueue) > 0:
        #     lst = np.array([j.job_len for j in jobqueue])
        #
        #     def getter(j):
        #         return j.status
        #
        #     vfunc = np.vectorize(getter, otypes=[str])
        #     runnings = np.where(vfunc(jobqueue) == 'running')
        #     lst[runnings] = 1000
        #     action = np.argmin(lst)

        return action
    def sjf_action(self, jobqueue):
        action = 0

        if len(jobqueue) > 0:
            lst = np.array([j.job_len for j in jobqueue])

            def getter(j):
                return j.status

            vfunc = np.vectorize(getter, otypes=[str])
            runnings = np.where(vfunc(jobqueue) == 'running')
            lst[runnings] = 1000
            action = np.argmin(lst)

        return action

    def ljf_action(self, jobqueue):
        action = 0  # 0 ~ 15

        if len(jobqueue) > 0:
            lst = np.array([j.job_len for j in jobqueue])

            def getter(j):
                return j.status

            vfunc = np.vectorize(getter, otypes=[str])
            runnings = np.where(vfunc(jobqueue) == 'running')
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
