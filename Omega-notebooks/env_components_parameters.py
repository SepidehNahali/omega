import numpy as np

__all__ = ['Parameters']


class Parameters():
    """The class to hold the customized parameters of the environment.

    This class must be modified before use, it sets all configurations of
    the environment.

    Attributes
    ----------
    gpu_per_cluster : int
        The number of gpus per cluster
    num_cluster : int
        The number of resources in the environment
    max_gpu_request : int
        The maximum gpu request for the job
    max_job_len : int
        The maximum time step that a job can run
    jobqueue_maxlen : int
        The maximum length of the jobqueue
    max_backlog_len : int
        The maximum length of the backlog
    Methods
    -------

    """
 
    def __init__(self, num_gpus_per_machine: int = 4, num_machines_per_rack: int = 3, num_racks_per_cluster: int = 2,
                 max_gpu_request: int = 4, max_job_len: int = 10, jobqueue_maxlen: int = 10, max_backlog_len: int = 10,
                 new_job_rate: float = 1.0, delay_penalty: int = -1, hold_penalty: int = -1, dismiss_penalty: int = -1,
                 max_num_timesteps: int = 10000, target_num_job_done: int = 20, ret_reducer: float = 1.0,
                 gpu_request_skew: float = 0.0, job_len_skew: float = 0.0):
        """Constructor of the parameters

        Parameters
        ----------
        gpu_per_cluster : int
            The number of gpus per cluster
        num_cluster : int
            The number of resources in the environment
        max_gpu_request : int, optional, default: 4
            The maximum gpu request for the job
        max_job_len : int, optional, default: 10
            The maximum time step that a job can run
        jobqueue_maxlen : int, optional, default: 10
            The maximum length of the jobqueue
        max_backlog_len : int, optional, default: 10
            The maximum length of the backlog

        max_backlog_len : int, optional, default: 10
            The maximum length of the backlog

        delay_penalty : int,  optional, default: -1
            penalty for delaying jobs in their runs
        hold_penalty : int,  optional, default: -1
            penalty for holding jobs in the jobqueue without running
        dismiss_penalty : int,  optional, default: -1
            penalty for missing a job because the jobqueue is full

        new_job_rate : float
            A float number between 0 and 1, indicates the probability of a new job arrives.

        """
        self.num_gpus_per_machine = num_gpus_per_machine
        self.num_machines_per_rack = num_machines_per_rack
        self.num_racks_per_cluster = num_racks_per_cluster


        self.max_gpu_request = max_gpu_request
        self.max_job_len = max_job_len
        self.jobqueue_maxlen = jobqueue_maxlen
        self.max_backlog_len = max_backlog_len

        # self.time_horizon = 20  # Hossein
        self.new_job_rate = new_job_rate
        self.delay_penalty = delay_penalty
        self.hold_penalty = hold_penalty
        self.dismiss_penalty = dismiss_penalty
        self.max_num_timesteps = max_num_timesteps
        self.target_num_job_arrive = target_num_job_done
        self.ret_reducer = ret_reducer
        self.gpu_requst_skew = gpu_request_skew
        self.job_len_skew = job_len_skew
