import json
from abc import ABC, abstractmethod
from pickletools import int4
import numpy as np
from scipy.interpolate import interp1d

from env_components_topolgy import *

from env_components_gpu import Gpu
from env_components_parameters import Parameters
from env_components_job import *

from env_components_data_generator import *
from environment_env_util import *
from environment_env_util import choose_gpu_fn, gpu_assign_time_status_fn
from typing import List, Deque
from collections import deque
from environment_simple_dqn_env import *
########## Hossein ###########
import pdb
import random
from matplotlib import pyplot as plt

##############################

################################################## I changed the get-avl-gpu usage in all methods with cluster number(*C-num)=0 to have one cluster. but it is not 

########## Hossein ###########
import pdb
import random
from matplotlib import pyplot as plt

##############################


__all__ = ['Env']


class Env(ABC):
    """Metaclass for defining customized environments.

    Use this metaclass to create an environment. All customized environments
    are subclasses from this one. In order to create it, step() and observe()
    must be implemented.

    Attributes
    ----------
    pa : Parameters
        Parameter of the environment
    done : bool
        A varible that indicates whether the environment finished its work or not ()
        (used during the training)
    resources : ndarray
        Stores all resources, each cluster contains a set of gpus
    jobqueue : ndarray
        Stores waiting jobs, a maximum length should constraint the jobqueue
    runningjobqueue : ndarray
        Stores the running jobs
    jobqueue_maxlen : int
        The maximum length of the jobqueue
    datagenerator : DataGenerating
        The generator to generate the job attributes
    backlog : Deque
        Stores the waiting jobs when the jobqueue is full.
    len_seq : ndarray
        The sequence of job length
    gpu_request_seq : ndarray
        The sequence of gpu request of jobs

    Methods
    -------
    reset()
        Reset the environment
    step(action)
        Execute an action on the environment
    observe()
        Returns information of the environment in the designed format.
    reward()
        Returns the reward from the step function
    generate_job_sequence(size, skew1, skew2)
        Generate the job_len sequence and gpu request sequence.
    insert_new_job()
        Insert a new job into the jobqueue

    progress_all_job()
        Progress all running jobs
    get_avl_gpus(*c_idx)
        Find all available gpus in the environment or selected resources
    get_running_jobs()
        Get the index of running jobs
    get_waiting_jobs()
        Get the index of waiting jobs
    get_paused_jobs()
        Get the index of paused jobs
    get_done_jobs()
        Get the index of finished jobs
    get_first_k_gpus(k)
        Get k random avalible gpus

    """
    def __init__(self, pa):
        """Constructor of the environment

        Parameters
        ----------
        pa : Parameters
            The object that stores all customized attributes of the environment
        """
        self.done = False

        self.pa = pa
        # Host class, contains gpu
        self.resources = -np.ones((self.pa.num_racks_per_cluster, self.pa.num_machines_per_rack,
                                   self.pa.num_gpus_per_machine), dtype=np.int32)  # Hossein
        self.gpus_array = np.array([], dtype=Gpu)
        # self.resources[:] = -1 # Hossein
        
        # self.jobqueue = [None] * self.pa.jobqueue_maxlen # Hossein
        # self.jobqueue = np.full(self.pa.jobqueue_maxlen, None, dtype=np.object) # Hossein

        self.jobqueue = np.array([])
        self.backlog = deque(maxlen=pa.max_backlog_len)

        self.jobqueue_maxlen = pa.jobqueue_maxlen
        self.j_id = 0
        self.last_job_id = 0
        self.total_step = 0
        self.num_job_finished = 0

        self.datagenerator = DataGenerating(max_gpu_request=pa.max_gpu_request, max_len=pa.max_job_len,resources=self.resources,ret_reducer=pa.ret_reducer)
        self.len_seq = np.array([])
        self.gpu_request_seq = np.array([])
        self.gpus_per_rack = self.resources.shape[1] * self.resources.shape[2]
        self.curr_time = 0
        self.episode_reward = np.array([])
        self.topology_parameters = NetworkTopology()
        # self.topology_parameters.node_index_build()
        # self.topology_parameters.create_graph()
        # self.topology_parameters.create_adjacency_matrix()
        # self.topology_parameters.get_gpu_distance_from_all(1)
        # self.topology_parameters.get_gpu_sort_nearest(2)
        # print('here is env total gpu nymbers: ', topology_parameters.gpu_number)
        # for cluster in topology_parameters.cluster_gpus:
        #     print('cluster_gpus: ',cluster.gpu_id)
        # print('im in environment class here is the parameters: ',topology_parameters.get_gpu_sort_nearest(6))
        self.assigned_gpu_to_jobs = {}
        
    def reset(self):
        """Reset the environment

        """
        self.done = False

        # self.resources = np.empty((self.pa.num_cluster, self.pa.gpu_per_cluster)) # Hossein
        # self.resources[:] = np.nan # Hossein
        self.resources = -np.ones((self.pa.num_racks_per_cluster, self.pa.num_machines_per_rack,
                                   self.pa.num_gpus_per_machine), dtype=np.int32)  # Hossein

        # self.jobqueue = np.full(self.pa.jobqueue_maxlen, None, dtype=np.object) # Hossein

        self.jobqueue = np.array([])
        self.backlog = deque(maxlen=self.pa.max_backlog_len)

        self.jobqueue_maxlen = self.pa.jobqueue_maxlen
        self.j_id = 0
        self.total_step = 0
        self.num_job_finished = 0

        self.datagenerator = DataGenerating(max_gpu_request=self.pa.max_gpu_request, max_len=self.pa.max_job_len,
                                            resources=self.resources,
                                            ret_reducer=self.pa.ret_reducer)
        self.len_seq = np.array([])
        self.gpu_request_seq = np.array([])
        self.gpus_per_rack = self.resources.shape[1] * self.resources.shape[2]
        self.curr_time = 0
        self.episode_reward = np.array([])
        return self.observe()

    @abstractmethod
    def step(self, action):
        """Execute an action on the environment

        This function should be implemented by users for different purposes.

        Raises
        ------
        NotImplementedError
            If the action is none
        """
        if action is None:
            raise NotImplementedError("The step function has not defined yet")

    def observe_raw_data(self):
        """Returns information of the environment in the designed format

            This function should be implemented by users for different purposes.


        """
        # Locally Generate Jobs & Array of GPUs #############################
        random.seed(1)
        # initialize job queue
        for i in range(self.pa.jobqueue_maxlen):
            self.jobqueue = np.append(self.jobqueue,
                                      Job(id=i, gpu_request=random.randint(1, 3), len=random.randint(1, 10), ts_0j=0,
                                          d_exj_tlop=0, tt_m=0, m=0, d_fj_tflop=0, gradsize=0))

        # initialize an array of GPUs
        for rack_index in range(self.pa.num_racks_per_cluster):
            for machine_index in range(self.pa.num_machines_per_rack):
                for gpu_index in range(self.pa.num_gpus_per_machine):
                    self.gpus_array = np.append(self.gpus_array,
                                                Gpu(rack_id=rack_index, host_id=machine_index, gpu_id=gpu_index,
                                                    remaining_time=0, status=0))

        # define functions to update GPU objects in the Array of GPUs
        gpu_choose = np.frompyfunc(choose_gpu_fn, 4, 1)
        gpu_assign_time_status = np.frompyfunc(gpu_assign_time_status_fn, 2, 0)

        # manually update jobs in jobqueue and GPUs in the array of GPU
        self.jobqueue[2].status = 'running'
        self.resources[0][0][0] = 2
        selected_gpus = self.gpus_array[np.array(gpu_choose(self.gpus_array, 0, 0, 0), dtype=bool)]
        gpu_assign_time_status(selected_gpus, self.jobqueue[2])

        self.jobqueue[3].status = 'running'
        self.resources[0][0][2] = 3
        selected_gpus = self.gpus_array[np.array(gpu_choose(self.gpus_array, 0, 0, 2), dtype=bool)]
        gpu_assign_time_status(selected_gpus, self.jobqueue[3])
        for rack_no in range(self.resources.shape[0]):
            for machine_no in range(self.resources.shape[1]):
                self.resources[rack_no][machine_no][3] = 3
                selected_gpus = self.gpus_array[
                    np.array(gpu_choose(self.gpus_array, rack_no, machine_no, 3), dtype=bool)]
                gpu_assign_time_status(selected_gpus, self.jobqueue[3])
        #####################################################################

        # Create an Array of Waiting Jobs ###################################
        choose_indices_obj = np.frompyfunc(lambda jobqueue, status: status == jobqueue.status, 2, 1)
        indices_obj = choose_indices_obj(self.jobqueue, 'waiting')
        indices_bool = np.array(indices_obj, dtype=bool)
        jobs_waiting_array = self.jobqueue[indices_bool]
        #####################################################################

        # Create Adjacency Matrix ###########################################
        gpus_num_total = self.pa.num_racks_per_cluster * self.pa.num_machines_per_rack * self.pa.num_gpus_per_machine
        adj_matrix = np.zeros((gpus_num_total, gpus_num_total))
        with open('../environment/speed_config.json') as config_file:
            speed_config = json.load(config_file)
        within_cluster_between_racks_speed = speed_config['within_cluster_between_racks_speed']
        within_rack_between_hosts_speed = speed_config['within_rack_between_hosts_speed']
        within_host_speed = speed_config['within_host_speed']
        for i in range(gpus_num_total):
            for j in range(i + 1, gpus_num_total):
                if self.gpus_array[i].rack_id != self.gpus_array[j].rack_id:
                    adj_matrix[i, j] = 1 / within_cluster_between_racks_speed
                elif self.gpus_array[i].host_id != self.gpus_array[j].host_id:
                    adj_matrix[i, j] = 1 / within_rack_between_hosts_speed
                else:
                    adj_matrix[i, j] = 1 / within_host_speed
        adj_matrix = adj_matrix + adj_matrix.transpose()
        #####################################################################

        return self.gpus_array, jobs_waiting_array, adj_matrix

    def generate_job_sequence(self, size: int):

        """Function calls the datagenerating function in data_generator class

        Parameters
        ----------
        size : int
            The size of the generating data
        skew1 : float, optional, default: 0.0
            The skewness of the distribution of job len, negative is left skewed,
            positive is right skewed, default 0 is normal distribution
        skew2 : float, optional, default: 0.0
            The skewness of the distribution of gpu request, negative is left skewed,
            positive is right skewed, default 0 is normal distribution

        Returns
        -------
        A list of job len and gpu request
        """
        skew1 = self.pa.job_len_skew
        skew2 = self.pa.gpu_requst_skew
        self.len_seq, self.gpu_request_seq = self.datagenerator.data_gen(size, skew1, skew2)

    def insert_new_job(self):
        """Insert a new job into the jobqueue

        Returns
        -------
        None
        """
        if len(self.jobqueue) < self.jobqueue_maxlen:
            if len(self.backlog) > 0:
                self.jobqueue = np.append(self.jobqueue, self.backlog.popleft())
            elif (self.num_job_finished + len(self.jobqueue) + len(self.backlog)) < self.pa.target_num_job_arrive:
                d_ex, m, tt_m, d_fj_tflop, gradsize = self.datagenerator.dnn_data_gen(self.gpu_request_seq[self.j_id],
                                                                                      self.len_seq[self.j_id])
                j = Job(id=self.j_id, gpu_request=self.gpu_request_seq[self.j_id],
                        len=self.len_seq[self.j_id], ts_0j=self.curr_time, d_exj_tlop=d_ex, tt_m=tt_m,
                        m=m, d_fj_tflop=d_fj_tflop, gradsize=gradsize)
                self.jobqueue = np.append(self.jobqueue, j)
                self.j_id += 1
            else:
                return False
        elif len(self.backlog) < self.pa.max_backlog_len and \
                (self.num_job_finished + len(self.jobqueue) + len(self.backlog)) < self.pa.target_num_job_arrive:
            d_ex, m, tt_m, d_fj_tflop, gradsize = self.datagenerator.dnn_data_gen(self.gpu_request_seq[self.j_id],
                                                                                  self.len_seq[self.j_id])
            j = Job(id=self.j_id, gpu_request=self.gpu_request_seq[self.j_id],
                    len=self.len_seq[self.j_id], ts_0j=self.curr_time, d_exj_tlop=d_ex, tt_m=tt_m,
                    m=m, d_fj_tflop=d_fj_tflop, gradsize=gradsize)
            self.backlog.append(j)
            self.j_id += 1
        else:
            return False

        return True

    def progress_all_jobs(self):
        """Progress all running jobs in the runningjobqueue

        Add 1 on each job's progress

        Returns
        -------
        None
        """

        # Using internal clock, instead of calling from step
        run_jobs = self.get_running_jobs()
        for j in self.jobqueue[run_jobs]:
            j.forward_one_step()
        for job in self.jobqueue[self.get_waiting_jobs()]:
            job.waiting_time += 1

    def advance_runningjobs_onestep(self):
        gpus_per_rack = self.resources.shape[1] * self.resources.shape[2]
        for job in self.jobqueue[self.get_running_jobs()]:
            job.v_m, job.rt_m = calc_job_minbatch_speed(job, gpus_per_rack, self.pa.ret_reducer, singleormulti='multi',outdetails=True)
            d_delta = self.datagenerator.jobadvancecompdist(job)
            job.d_done += d_delta
            job.stepcounter += 1
            job.ts_togo, job.ts_done = self.datagenerator.jobnumrowstime(job)
        for job in self.jobqueue[self.get_waiting_jobs()]:
            job.waiting_time += 1

    def remove_jobs(self, jobs):
        for j in jobs[0]:
            self.resources[self.jobqueue[j].gpus] = -1
            # print(f'removed {self.jobqueue[j].job_id} {self.jobqueue[j].gpu_request}')
            self.num_job_finished += 1
        self.jobqueue = np.delete(self.jobqueue, jobs)

    def assign_job_gpus(self, j_idx, gpus: List[tuple]) -> bool:
        """Assign gpus to a job

        Parameters
        ---------
        j_indx : int
            The index of the job
        gpus : List[tuple]
            The coordinates of gpus, first item in the tuple contains the cluster index,
            the second item contains the gpu index in the cluster. e.g. ((0,0), (0,1))
            represents the first and the second gpu in the first cluster.

        Returns
        -------
        bool
            Indicate the job has been allocated or not
        """
        if all(x == -1 for x in self.resources[gpus]):
            self.resources[gpus] = self.jobqueue[j_idx].job_id
            # print('job ',self.jobqueue[j_idx].job_id,' Just assigend  cordination: ',gpus )

            # print('Job id: ',j_idx)
            self.jobqueue[j_idx].status = 'running'
            self.jobqueue[j_idx].gpus = gpus
            # print('self.resources : ',self.resources)
            # for ii in range(len(self.jobqueue)):
            #     print('Job Queue: ',self.jobqueue[ii].job_id)

            return True
        return False

    def random_select_k_gpus_for_job(self, j):
        # print('This is random_select_k_gpus_for_job: ')

        """Randomly select k gpus for the target job j

        Parameters
        ---------
        j : int
            The index of the job in the jobqueue

        Returns
        -------
        selected gpus : (ndarray, ndarray)
            A set of selected gpus
        """
   
        if j < len(self.jobqueue):
            gpu_request = self.jobqueue[j].gpu_request
            avl_gpus = self.get_avl_gpus()
            # print('avl_gpus[0]),len(avl_gpus[0]),gpu_request',avl_gpus[0],len(avl_gpus[0]),gpu_request)
            if len(avl_gpus[0]) >= 1 and len(avl_gpus[0]) >= gpu_request:
               
                # print('self.get_random_gpus(gpu_request): ',gpu_request,self.get_first_k_gpus(gpu_request))
                # print('1- random_select_k_gpus_for_job: ',j ,len(self.jobqueue),avl_gpus[0],len(avl_gpus[0]))
                return self.get_first_k_gpus(gpu_request)

            else:
                # print('2- random_select_k_gpus_for_job: ',j ,len(self.jobqueue),avl_gpus[0],len(avl_gpus[0]))
                # print('self.resources ',self.resources)

                # return ([], [], [], [], [])
                return ([], [], [])

        else:
            # return ([], [], [], [], [])
            print('3- random_select_k_gpus_for_job: ',j ,len(self.jobqueue),avl_gpus[0],len(avl_gpus[0]))

            return ([], [], [])

   
    def get_avl_gpus(self, *c_idx):
        """Find all available gpus in the environment

        Parameters
        ---------
        c_idx : (idx) optional
            A list of index of resources, only find the avalible gpus inside
            the selected clutsers if this parameter is used.

            e.g. get_avl_gpus(0,1) will find all available gpus in the first and the second cluster.
        """
        if not c_idx:
            result = np.where(self.resources == -1)
        else:
            result = np.where(self.resources[np.array(c_idx)] == -1)
        return result

    def get_running_jobs(self):
        """Get the index of running jobs

        Returns
        -------
        running_jobs :
            The index of running jobs
        """

        def getter(j):
            return j.status

        vfunc = np.vectorize(getter, otypes=[str])
        running_jobs = np.where(vfunc(self.jobqueue) == 'running')
        return running_jobs

    def get_waiting_jobs(self):
        """Get the index of waiting jobs

        Returns
        -------
        running_jobs :
            The index of waiting jobs
        """

        def getter(j):
            return j.status

        vfunc = np.vectorize(getter, otypes=[str])
        waiting_jobs = np.where(vfunc(self.jobqueue) == 'waiting')
        return waiting_jobs

    def get_paused_jobs(self):
        """Get the index of paused jobs

        Returns
        -------
        running_jobs :
            The index of waiting jobs
        """

        def getter(j):
            return j.status

        vfunc = np.vectorize(getter, otypes=[str])
        paused_jobs = np.where(vfunc(self.jobqueue) == 'paused')
        return paused_jobs

    def get_done_jobs(self):
        """Get the index of finished jobs

        Returns
        -------
        done_jobs :
            Thie index of finished jobs
        """

        def getter(j):
            # print('total computational distance of the job: ',j.job_id,' is equal to: ', j.d_f)
            # print('and','j.d_done is: ',j.d_done )
            return j.d_done >= j.d_f

        vfunc = np.vectorize(getter, otypes=[bool])
        # print('done jobs vfunc: ',vfunc )

        done_jobs = np.where(vfunc(self.jobqueue))

        # print('done jobs: ',done_jobs)
        # print('done jobs len: ',len(done_jobs))


        # print('done_jobs[0],done_jobs[1],done_jobs[2] : ',done_jobs[0],done_jobs[1],done_jobs[2])


        return done_jobs


    def get_first_k_gpus(self, k: int, random: bool=False) -> tuple:
        """Get random k available gpus

        Parametersgit
        ---------
        k : int
            The number of gpus that would like to pick up.

        Returns
        -------
        gpus : tuple
            A tuple represents the coordinates of selected gpus.git status
        """

        # print('This is get_random_gpus: ')

        """Get random k available gpus

        Parametersgit
        ---------
        k : int
            The number of gpus that would like to pick up.

        Returns
        -------
        gpus : tuple
            A tuple represents the coordinates of selected gpus.git status
        """
        def get_kth_dims(f):#returns the Kth element dimentions in the 3-darray of self.resources
            assert(f<32)
            if f==3:
               return 0,0,3
            if f==6:
               return 0,1,2
            if f==7:
               return 0,1,3
            if f==11:
               return 0,2,3
            if f<=15:
                x=0
                y = round(f/4)
                if(y>=4):
                  y=3
                z = f - 4*y
            if f>15:
                x=1
                f=f-16
                y=int(np.floor(f/4))
                if(y>=4):
                  y=3
                z= f- 4*y
            return x,y,z
        first_avl_gpus = self.get_avl_gpus()

        # x0 = self.get_avl_gpus(0)[0][:k] # first k index number of the first dim of resources(num of racks)
        # x1 = self.get_avl_gpus(0)[1][:k] # first k index number of the second dim of resources(machine per rack)
        # x2 = self.get_avl_gpus(0)[2][:k] # first k index number of the third dim of resources(gpu per machine)
        
        x_ = int(self.get_avl_gpus()[0][:1])####find the first idle gpu as before(first dim)
        y_ = int(self.get_avl_gpus()[1][:1])####find the first idle gpu as before(second dim)
        z_ = int(self.get_avl_gpus()[2][:1])####find the first idle gpu as before(third dim)
    
        WIDTH = 2 #(machine per rack)
        DEPTH = 4 #(gpu per machine)
        HEIGHT = 4 
        x0 = np.where(self.resources == -1)[0][:0]# create three empty array to put the nearest gpu dims in them
        x1 = np.where(self.resources == -1)[1][:0]
        x2 = np.where(self.resources == -1)[2][:0]#(by appending will be extended to (k,) or np.where(self.resources == -1)[2][:k])

        first_gpu_Id = x_ * HEIGHT * DEPTH + y_ * DEPTH + z_# get the first gpu number dims found randomly in(x,y,z)
        # print('first_gpu_Id:',first_gpu_Id)

        get_nearest = self.topology_parameters.get_gpu_sort_nearest(first_gpu_Id)#search nearest gpus to it
        # x0=np.append(x0,x_)# append the  to x0 the first gpu
        # x1=np.append(x1,y_)# append the  to x1 the first gpu
        # x2=np.append(x2,z_)# append the  to x2 the first gpu
        idleneighbour=0
        for i in range(len(get_nearest)-1):# for each neighbor gpu ID:[3,34,2,5],len=3,i=1,4
            x,y,z= get_kth_dims(get_nearest[i])# tell the dims in resources matrice
            if (self.resources[x][y][z]==-1): # to make sure that the neighbouring GPUs are idle
                idleneighbour=idleneighbour+1
                if (idleneighbour>k):#find the rest k-1 gpus from the sorted list
                    # print('get_kth_dims:    ',get_kth_dims)
                    break
                x0=np.append(x0,x)# append the  to x0 which contains the k nearest gpus first dims
                x1=np.append(x1,y)# append the  to x0 which contains the k nearest gpus second dims
                x2=np.append(x2,z)# append the  to x0 which contains the k nearest gpus third dims
        return (x0, x1, x2)# before x0=np.where(self.resources == -1)[0][:k],...  had the first k gpus in resources marice. now it has nearest k gpus in terms of topology



    def reward_throughput(self):
        def _log(x):
            if x <= 0:
               return 0.01
            else:
                return x
        total_waiting = 0
        reward = 0
        # print('here is env, reward funcion the job is :',self.joboo,' the gpus are: ',self.gpuu,' the gpus type: ',type(self.gpuu))
        # print('here is env, reward funcion availible gpus :',self.get_avl_gpus())
        # print('here is env, reward funcion all resources :',self.resources)
        # print('here is env, reward funcion resources of cluster:',clus,' : ',self.resources[np.array(clus)])
        # print('self.assigned_gpu_to_jobs: ',get_assigned_gpu_job_dict())
        distancefromothers=0
        topology_parameters = self.topology_parameters

        # for j in self.jobqueue[self.get_running_jobs()]:
        #     # print('im in reward loop, running job is: ',j.job_id)
        #     for gp in j.gpus:
        #         # print('list of gpus of the job: ',gp)
        #         if(len(gp)>1):
        #             # print('length of gpu array: ',len(gp))
        #             nearest = topology_parameters.get_gpu_sort_nearest(gp[0]) #containing the first node itself
        #             a=gp.tolist() # we could use (==).all in if statement compariation for np array too
        #             a.sort()
        #             b=nearest[:len(gp)].tolist()
        #             b.sort() # index is because selecting first n of needed gpus from nearest gpus list
        #             # topology_parameters.get_gpu_sort_nearest(gp[0])
        #             for i in gp:
        #                 distancefromothers += topology_parameters.get_gpu_distance_gpu(gp[0],i)
        #             if(a==b): # sort nearest gpus to the first gpu and selected gpus to compare if they are optimal
        #                 reward *= distancefromothers
        #             else:
        #                 reward /= distancefromothers
                   


        for j in self.jobqueue[self.get_running_jobs()]:
            reward += j.v_m
            if len(j.gpus[0]) != j.gpu_request:
                reward += penalty_assigned_gpus(job=j, gpus_per_rack=self.gpus_per_rack, ret_reducer=self.pa.ret_reducer) * self.pa.delay_penalty
            # if j.stepcounter == 0:
            #     reward += j.job_len * j.gpu_request
        for j in self.jobqueue[self.get_waiting_jobs()]:
            reward -= calc_job_minbatch_speed(job=j, gpus_per_rack=self.gpus_per_rack, ret_reducer=self.pa.ret_reducer, singleormulti='single') * self.pa.hold_penalty

        # for j in self.backlog:
        #     reward += calc_job_minbatch_speed(job=j, gpus_per_rack=self.gpus_per_rack, ret_reducer=self.pa.ret_reducer, singleormulti='single') * self.pa.dismiss_penalty
        reward = np.clip(reward, -300, 100)
        return reward




