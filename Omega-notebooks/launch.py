##### import os
import sys
sys.path.insert(0, "./.")
sys.path.insert(0, "../.")
import argparse
import torch.optim as optim

from env_components_parameters import Parameters
from environment_simple_dqn_env import *
from dqn_agent import *
from train import *
from utils_schedules import *
from environment_env_util import *
from env_components_topolgy import *


def exp1(name):
    num_gpus_per_machine = 4
    num_machines_per_rack = 4
    num_racks_per_cluster = 2
    max_gpu_request = 8
    max_job_len = 30
    jobqueue_maxlen = 20
    max_backlog_len = 5
    new_job_rate = 0.6
    target_num_job_done = 150
    delay_penalty = -1
    hold_penalty = -2
    dismiss_penalty = -1


    pa = Parameters(num_gpus_per_machine=num_gpus_per_machine,
                    num_racks_per_cluster=num_racks_per_cluster,
                    num_machines_per_rack=num_machines_per_rack,
                    max_gpu_request=max_gpu_request,
                    max_job_len=max_job_len,
                    jobqueue_maxlen=jobqueue_maxlen,
                    max_backlog_len=max_backlog_len,
                    new_job_rate=new_job_rate,
                    hold_penalty=hold_penalty,
                    delay_penalty=delay_penalty,
                    dismiss_penalty=dismiss_penalty,
                    target_num_job_done=target_num_job_done,
                    max_num_timesteps=90000)

    BATCH_SIZE = 64
    REPLAY_BUFFER_SIZE = 30000
    FRAME_HISTORY_LEN = 1
    TARGET_UPDATE_FREQ = 120
    GAMMA = 0.95
    LEARNING_FREQ = 40
    LEARNING_RATE = 0.01
    ALPHA = 0.95
    EPS = 0.01
    EXPLORATION_SCHEDULE = LinearSchedule(60000, 0.1)
    LEARNING_STARTS = 20000
    NUM_EPOCH= 100
    NUM_EPISODE = 100
    env = DQNTesting1(pa)
    baselines = ['SJF','LJF', 'RANDOM','TETRIS']
    


    # topology_parameters=NetworkTopology(pa)
    # topology_parameters.node_index_build()
    # topology_parameters.create_graph()
    # topology_parameters.create_adjacency_matrix()
    # distancefromothers = 0
    # jobGpusIDlist= [1,31,24]
    # print(    topology_parameters.get_gpu_distance_from_all(1))

    # for i in range(len(jobGpusIDlist)):
    #     for j in range(i+1):
    #       if(j<len(jobGpusIDlist)-1):
    #         if(i!=j):
    #           print(jobGpusIDlist[i],jobGpusIDlist[j],':',topology_parameters.get_gpu_distance_gpu(jobGpusIDlist[i],jobGpusIDlist[j]))
    #           distancefromothers += topology_parameters.get_gpu_distance_gpu(jobGpusIDlist[i],jobGpusIDlist[j])
    
   
    # print(distancefromothers)

    # distancefromothers = [topology_parameters.get_gpu_distance_gpu(jobGpusIDlist[i],jobGpusIDlist[j]) for i in range(len(jobGpusIDlist)) for j in range(i+1) if i!=j]
    
    # print(sum(distancefromothers))
    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return t > env.pa.max_num_timesteps

    optimizer = OptimizerSpec(
        constructor=optim.Adam,
        kwargs=dict(lr=LEARNING_RATE, eps=EPS)
    )
    train(
        env=env,
        agent=DQNAgent(env, learning_rate=LEARNING_RATE, gamma=GAMMA, buffer_size=REPLAY_BUFFER_SIZE,
                       target_update_freq=TARGET_UPDATE_FREQ),
        optimizer_spec=optimizer,
        exploration=EXPLORATION_SCHEDULE,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=REPLAY_BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        gamma=GAMMA,
        learning_starts=LEARNING_STARTS,
        learning_freq=LEARNING_FREQ,
        frame_history_len=FRAME_HISTORY_LEN,
        # num_epoch = NUM_EPOCH,
        # num_episode = NUM_EPISODE
        exp_name=name,
        baselines=baselines
    )

  

    return None


    return None

# def main():
parser = argparse.ArgumentParser(prog='launch',
    description='Choose and run the experiment')

parser.add_argument('-e',
                    type=int,
                    help='the id of the experiment',
                    required=True)
parser.add_argument('-g',
                    type=int,
                    help='the id of the gpu')
parser.add_argument('-n',
                    type=str,
                    help='name of the experiment')
# args = parser.parse_args()

# args=[1,2,'first']
# if (vars(args)['g'] != None):
#     if torch.cuda.is_available():
#         torch.cuda.set_device(vars(args)['g'])
#         print("CUDA Device: %d" % torch.cuda.current_device())
# if vars(args)['e'] == 1:
# exp1(vars(args)['n'])
exp1('eighth')
# train(env)

# if __name__ == '__main__':
#     main()
