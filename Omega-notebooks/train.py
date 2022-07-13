from torch.autograd import Variable
from collections import namedtuple
from utils_replay_buffer import *
from utils_logger import *
from baselines import *
import itertools
import random
import os
import sys
import numpy as np
import torch
import torch.nn as nn
__all__ = ['train', 'OptimizerSpec', 'train2']



OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs"])



USE_CUDA = torch.cuda.is_available()
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dlongtype = torch.cuda.LongTensor if torch.cuda.is_available() else torch.LongTensor


def to_np(x):
    return x.data.cpu().numpy()

def train(
        env,
        agent,
        optimizer_spec,
        exploration,
        stopping_criterion,
        batch_size,
        learning_starts,
        learning_freq,
        frame_history_len,
        exp_name,
        baselines,
        with_mask,
        clip_range
    ):
    baseline_schedulers = BaseLines(baselines,clip_range)
    logger = Logger(f'./logs/{exp_name}', baselines)
    ob_shape = env.observe().shape
    img_h = ob_shape[0]
    img_w = ob_shape[1]
    img_c = ob_shape[2]
    if with_mask:
        num_actions = env.jobqueue_maxlen + 1
    else:
        num_actions = env.jobqueue_maxlen

    # Training start
    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 3000
    SAVE_MODEL_EVERY_N_STEPS = 4000

    for t in itertools.count():
        if stopping_criterion is not None and stopping_criterion(env, t):
            break
        last_stored_frame_idx = agent.replay_buffer.store_frame(last_obs)
        observations = agent.replay_buffer.encode_recent_observation()

        if t < learning_starts:
            action = np.random.randint(num_actions)
        else:
            threshold = exploration.value(t)
            agent.get_action(observations, threshold)
        env.generate_job_sequence(env.pa.target_num_job_arrive)
        obs, reward, done = env.step(action)


        agent.replay_buffer.store_effect(last_stored_frame_idx, action, reward, done)

        if done:
            obs = env.reset()
            env.generate_job_sequence(env.pa.target_num_job_arrive)
        last_obs = obs

        if (t > learning_starts and t % learning_freq == 0 and agent.replay_buffer.can_sample(batch_size)):
            agent.update(batch_size, t)

            # (2) Log values and gradients of the parameters (histogram)
            if t % LOG_EVERY_N_STEPS == 0:
                for tag, value in agent.model.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, to_np(value), t + 1)
                    logger.histo_summary(tag + '/grad', to_np(value.grad), t + 1)

        if t % SAVE_MODEL_EVERY_N_STEPS == 0:
            if not os.path.exists("models"):
                os.makedirs("models")
            model_save_path = "models/%s_%d.model" % (
            exp_name, t)
            torch.save(agent.model.state_dict(), model_save_path)

        episode_rewards = env.episode_reward
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards)
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
        if t % LOG_EVERY_N_STEPS == 0:
            # ============ Testing and baselines ============#
            env.reset()
            np.random.seed()
            env.generate_job_sequence(env.pa.target_num_job_arrive)
            baseline_rewards = baseline_schedulers.get_multi_rewards(env)
            step_counter = 0
            avg_waiting_jobs = []
            while not env.done:
                # ob = env.observe().transpose(2, 0, 1)
                ob = env.observe()
                ob = torch.from_numpy(ob).unsqueeze(0).type(dtype)
                with torch.no_grad():
                    q_value_all_actions = agent.model(Variable(ob))

                ############  mask ###########
                if with_mask:
                    running = env.get_waiting_jobs()
                    mask = np.zeros(num_actions)
                    mask[running] = 1
                    mask[-1] = 1
                    inv_mask = 1 - mask
                    mask = torch.tensor(mask, dtype=torch.float).type(dtype)

                    masked = mask.mul(q_value_all_actions[0])
                    masked[inv_mask==1] = -np.inf
                    action = masked.data.max(0)[1]
                ############  mask ###########
                else:
                    action = ((q_value_all_actions).data.max(1)[1])[0]
                # print(f'waiting jobs{env.get_waiting_jobs()[0]} mased: {masked} action: {action}'
                #       f'number of waiting jobs: {len(env.get_waiting_jobs()[0])}')
                env.step(action)
                step_counter += 1
                if step_counter == clip_range:
                    break
                # print(f'action: {action} '
                #       f'{[j.status for j in env.jobqueue]}')

                avg_waiting_jobs.append(len(env.get_waiting_jobs()[0]))

            mean_test_rewards = np.mean(env.episode_reward[:clip_range])
            print(f'test: {mean_test_rewards} {len(env.episode_reward)}')
            logger.scalar_summary('average waiting jobs', np.mean(avg_waiting_jobs), t + 1)
            env.reset()
            np.random.seed()
            
            print("---------------------------------")
            print("Timestep %d" % (t,))
            print("learning started? %d" % (t > learning_starts))
            print("mean reward %f" % mean_episode_reward)
            print("mean testing reward %f" % mean_test_rewards)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t))
            print("learning_rate %f" % optimizer_spec.kwargs['lr'])
            print("exploration %f" % exploration.value(t))
            sys.stdout.flush()

            # ============ TensorBoard logging ============#
            # (1) Log the scalar values
            info = {
                'learning_started': (t > learning_starts),
                'exploration': exploration.value(t),
                'learning_rate': optimizer_spec.kwargs['lr']
            }

            for tag, value in info.items():
                logger.scalar_summary(tag, value, t + 1)

            if (best_mean_episode_reward != -float('inf')):
                info = {
                    'mean_training_episode_reward': mean_episode_reward,
                }

                for tag, value in info.items():
                    logger.scalar_summary(tag, value, t + 1)
            if (best_mean_episode_reward != -float('inf')):
                logger.test_summary(baseline_rewards, mean_test_rewards,t + 1)


def train2(
        env,
        agent,
        optimizer_spec,
        exploration,
        stopping_criterion,
        batch_size,
        learning_starts,
        learning_freq,
        frame_history_len,
        exp_name,
        baselines,
        with_mask,
        clip_range
):
    baseline_schedulers = BaseLines(baselines, clip_range)
    logger = Logger(f'./logs/{exp_name}', baselines)
    ob_shape = env.observe().shape
    img_h = ob_shape[0]
    img_w = ob_shape[1]
    img_c = ob_shape[2]
    if with_mask:
        max_num_jobs = env.jobqueue_maxlen + 1
    else:
        max_num_jobs = env.jobqueue_maxlen
    num_gpus = env.resources.size

    # Training start
    num_param_updates = 0
    mean_episode_reward = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 3000
    SAVE_MODEL_EVERY_N_STEPS = 4000

    for t in itertools.count():
        if stopping_criterion is not None and stopping_criterion(env, t):
            break
        last_stored_frame_idx = agent.replay_buffer.store_frame(last_obs)
        observations = agent.replay_buffer.encode_recent_observation()

        if t < learning_starts:
            j = np.random.randint(max_num_jobs)
            gpus = np.array()
            action = np.random.randint(max_num_jobs)

        else:
            threshold = exploration.value(t)
            agent.get_action(observations, threshold)
        env.generate_job_sequence(env.pa.target_num_job_arrive)
        obs, reward, done = env.step(action)

        agent.replay_buffer.store_effect(last_stored_frame_idx, action, reward, done)

        if done:
            obs = env.reset()
            env.generate_job_sequence(env.pa.target_num_job_arrive)
        last_obs = obs

        if (t > learning_starts and t % learning_freq == 0 and agent.replay_buffer.can_sample(batch_size)):
            agent.update(batch_size, t)

            # (2) Log values and gradients of the parameters (histogram)
            if t % LOG_EVERY_N_STEPS == 0:
                for tag, value in agent.model.named_parameters():
                    tag = tag.replace('.', '/')
                    logger.histo_summary(tag, to_np(value), t + 1)
                    logger.histo_summary(tag + '/grad', to_np(value.grad), t + 1)

        if t % SAVE_MODEL_EVERY_N_STEPS == 0:
            if not os.path.exists("models"):
                os.makedirs("models")
            model_save_path = "models/%s_%d.model" % (
                exp_name, t)
            torch.save(agent.model.state_dict(), model_save_path)

        episode_rewards = env.episode_reward
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards)
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
        if t % LOG_EVERY_N_STEPS == 0:
            # ============ Testing and baselines ============#
            env.reset()
            env.generate_job_sequence(env.pa.target_num_job_arrive)
            baseline_rewards = baseline_schedulers.get_multi_rewards(env)
            step_counter = 0
            avg_waiting_jobs = []
            while not env.done:
                # ob = env.observe().transpose(2, 0, 1)
                ob = env.observe()
                ob = torch.from_numpy(ob).unsqueeze(0).type(dtype)
                with torch.no_grad():
                    q_value_all_actions = agent.model(Variable(ob))

                ############  mask ###########
                if with_mask:
                    running = env.get_waiting_jobs()
                    mask = np.zeros(max_num_jobs)
                    mask[running] = 1
                    mask[-1] = 1
                    inv_mask = 1 - mask
                    mask = torch.tensor(mask, dtype=torch.float).type(dtype)

                    masked = mask.mul(q_value_all_actions[0])
                    masked[inv_mask == 1] = -np.inf
                    action = masked.data.max(0)[1]
                ############  mask ###########
                else:
                    action = ((q_value_all_actions).data.max(1)[1])[0]
                # print(f'waiting jobs{env.get_waiting_jobs()[0]} mased: {masked} action: {action}'
                #       f'number of waiting jobs: {len(env.get_waiting_jobs()[0])}')
                env.step(action)
                step_counter += 1
                if step_counter == clip_range:
                    break
                print(f'action: {action} '
                      f'{[j.status for j in env.jobqueue]}')

                avg_waiting_jobs.append(len(env.get_waiting_jobs()[0]))

            mean_test_rewards = np.mean(env.episode_reward[:clip_range])
            print(f'test: {np.mean(env.episode_reward[:clip_range])} {len(env.episode_reward)}')
            logger.scalar_summary('average waiting jobs', np.mean(avg_waiting_jobs), t + 1)
            env.reset()

            print("---------------------------------")
            print("Timestep %d" % (t,))
            print("learning started? %d" % (t > learning_starts))
            print("mean reward %f" % mean_episode_reward)
            print("mean testing reward %f" % mean_test_rewards)
            print("best mean reward %f" % best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            print("exploration %f" % exploration.value(t))
            print("learning_rate %f" % optimizer_spec.kwargs['lr'])
            print("exploration %f" % exploration.value(t))
            sys.stdout.flush()

            # ============ TensorBoard logging ============#
            # (1) Log the scalar values
            info = {
                'learning_started': (t > learning_starts),
                'exploration': exploration.value(t),
                'learning_rate': optimizer_spec.kwargs['lr']
            }

            for tag, value in info.items():
                logger.scalar_summary(tag, value, t + 1)

            if (best_mean_episode_reward != -float('inf')):
                info = {
                    'mean_training_episode_reward': mean_episode_reward,
                }

                for tag, value in info.items():
                    logger.scalar_summary(tag, value, t + 1)
            if (best_mean_episode_reward != -float('inf')):
                logger.test_summary(baseline_rewards, mean_test_rewards, t + 1)
