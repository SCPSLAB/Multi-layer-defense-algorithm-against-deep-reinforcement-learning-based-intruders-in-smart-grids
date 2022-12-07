
import random
import each_device_env
import numpy as np
import argparse
import torch
import torch.nn as nn
import numpy as np
import os
import pickle
from data_loader import get_loader 
from build_vocab import Vocabulary
from model import EncoderCNN, DecoderRNN
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms


class Environment:
    def __init__(self, Device_id, Device_usage, Energy_charge, udc, from_timeslotnumber, to_timeslotnumber
                 , consumption_period, penalty, incentive):
        self.to_timeslotnumber = to_timeslotnumber
        self.consumption_period = consumption_period
        self.penalty = penalty
        self.incentive = incentive
        self.preferences_satisfied = True
        self.Device_id = Device_id
        self.Device_usage = Device_usage
        self.Energy_charge = Energy_charge
        self.udc = udc
        self.from_timeslotnumber = from_timeslotnumber

        self.done = False
        self.time_stamp = 0
        self.state_accumulation = 0
        self.episode_rewards = []
        self.history_actions = []
        print(f'schedule: {self.from_timeslotnumber} - {self.to_timeslotnumber}, duration: {self.consumption_period}')

    def reset(self):
        self.done = False
        self.time_stamp = 0
        self.state_accumulation = 0
        self.history_actions = []
        return self.get_obs()

    
    def get_action_shape(self):
        return 1

    def get_obs_shape(self):
        return np.shape(self.get_obs())

    def action_space_sample(self):
        return [random.randint(0, 1)]
    
    def get_obs(self):
        return [self.time_stamp, self.state_accumulation, self.from_timeslotnumber, self.consumption_period, self.to_timeslotnumber]

    def get_obs_shape(self):
        return np.shape(self.get_obs())

    def reward(self, action):
        under_schedule= self.from_timeslotnumber <= self.time_stamp < self.to_timeslotnumber
        at_to_timeslotnumber = self.time_stamp == self.to_timeslotnumber

        reward_function = (1 - under_schedule) *                           (at_to_timeslotnumber * self.incentive * (self.consumption_period -                                                                 np.abs(self.consumption_period - self.state_accumulation)) +                            action * self.penalty +
                           (1 - action) * self.incentive) + \
                          under_schedule* (
                                  action * (self.Energy_charge[self.time_stamp] * self.Device_usage) + \
                                  (1 - action) * self.udc * self.Device_usage)
        reward_function *= -1
        self.episode_rewards.append(reward_function)
        return reward_function

    def old_reward(self, action):
        under_schedule= self.from_timeslotnumber <= self.time_stamp <= self.to_timeslotnumber
        at_to_timeslotnumber = self.time_stamp == self.to_timeslotnumber

        reward_function = (1 - under_schedule) *                           (action * self.penalty +
                           (1 - action) * self.incentive) + \
                          under_schedule* (
                                  at_to_timeslotnumber * ((not self.preferences_satisfied) *
                                                      self.penalty *
                                                      np.abs(self.consumption_period - self.state_accumulation) +
                                                      self.preferences_satisfied *
                                                      self.incentive * self.consumption_period) + \
                                  (1 - at_to_timeslotnumber) *
                                  (action * self.Energy_charge[self.time_stamp] * self.Device_usage + \
                                   (1 - action) * self.udc * self.Device_usage))
        reward_function *= -1
        self.episode_rewards.append(reward_function)
        return reward_function


    def step(self, action):

        self.history_actions.append(action)
        self.state_accumulation += action
        if self.state_accumulation != self.consumption_period and self.time_stamp == self.to_timeslotnumber:
            self.preferences_satisfied = False
        reward = self.reward(action)
        self.time_stamp += 1
        self.done = self.time_stamp == 24
        return self.get_obs(), reward, self.done, None

def get_random_env1():
    Device_id = 1
    udc = 0.
    Energy_charge = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 12, 12, 5, 5, 5, 5, 10, 10, 10, 5, 5, 5])
    Device_usage = 1
    from_timeslotnumber = 5
    to_timeslotnumber = 20
    consumption_period = 4
    penalty = 10.
    incentive = -10.

    return Environment(Device_id, Device_usage, Energy_charge, udc, from_timeslotnumber, to_timeslotnumber,
                           consumption_period, penalty, incentive)   




if __name__ == '__main__':
    env = get_random_env1()
    rewards = []
    while not env.done:
        a = np.random.randint(0, 2)
        ob, r, done, _ = env.step(a)
        rewards.append(r)
        print(f'action: {a}, reward: {r}, obs: {ob}')
    print("Mean reward of random actions: %s " % (sum(rewards) / len(rewards)))


# In[12]:


class MultipleDeviceEnvironment:
    def __init__(self, num_devices, devices=None):
        if devices is not None:
            self.devices = devices
        else:
            self.devices = [get_random_env1() for _ in range(num_devices)]
        self.history_actions = []

        self.reset()
        for d in self.devices:
            print(f'schedule: {d.from_timeslotnumber} - {d.to_timeslotnumber}, duration: {d.consumption_period}')

    def get_action_shape(self):
        return len(self.devices)

    def reset(self):
        self.done = False
        self.time_stamp = 0
        for d in self.devices:
            d.reset()
        return self.get_obs()

    def action_space_sample(self):
        return [random.randint(0, 1) for _ in self.devices]

    def get_obs_shape(self):
        return np.shape(self.get_obs())

    def get_obs(self):
        obs = []
        for d in self.devices:
            obs.extend(d.get_obs())
        return np.array(obs)

    def reward(self, action):
        r = 0
        for d, a in zip(self.devices, action):
            r += d.reward(a)
        return r

    def step(self, action):
        self.history_actions.append(action)
        reward = self.reward(action)
        self.time_stamp += 1
        self.done = self.time_stamp == 24
        return self.get_obs(), reward, self.done, None

def get_random_env():
    Device_id = 1
    udc = 0.
    Energy_charge = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 12, 12, 5, 5, 5, 5, 10, 10, 10, 5, 5, 5])
    Device_usage = 1
    from_timeslotnumber = 5
    to_timeslotnumber = 20
    consumption_period = 4
    penalty = 10.
    incentive = -10.

    return MultipleDeviceEnvironment(Device_id, Device_usage, Energy_charge, udc, from_timeslotnumber, to_timeslotnumber,
                       consumption_period, penalty, incentive)


# In[13]:



if __name__ == '__main__':
    env = MultipleDeviceEnvironment(1)
    while not env.done:
        a = env.action_space_sample()
        ob, r, done, _ = env.step(a)
        print(f'action: {a}, reward: {r}, obs: {ob}')


class GymDQNLearner:
    def __init__(self, multiple, num_devices):
        if multiple:
            self.saving_path = './saved_models/dqn/multiple/%s/' % (num_devices)
        else:
            self.saving_path = './saved_models/dqn/single/'

        self.num_devices = num_devices
        self.formatstr = "{0:0" + str(self.num_devices) + "b}"

        self.epochs = 10000
        self.gamma = .9
        self.epsilon = 1.
        self.train_per_epoch = 1
        self.n_generating_trajectories_per_epoch = 1
        self.max_memory_size = 2000
        self.max_trajectory_length = 1000
        self.batch_size = 256
        if not multiple:
            self.env = each_device_env.get_random_env()
        else:
            self.env = MultipleDeviceEnvironment(num_devices)
        self.state_embedding_size = self.env.get_obs_shape()[0]
        self.number_of_actions = self.env.get_action_shape()
        print(self.state_embedding_size, self.number_of_actions)
        self.layer_units = [32, 16, int(2**self.number_of_actions)]
        # self.layer_units = [64, 32, self.number_of_actions]
        self.layer_activations = ['tanh', 'relu', None]
        self.layer_keep_probs = [1., 1., 1.]
        self.layer_regularizers = [None,
                                   None,
                                   None]
        self.initialize_experience_replay_memory()

        self.create_model()
        self.load()

    def initialize_experience_replay_memory(self):
        self.experience_replay_memory = np.array([])

    def get_epsilon(self, i):
        return max(0.1, self.epsilon * (0.9989 ** i))

    def get_state_weights(self, trajectory):
        total_reward = np.sum([t[2] for t in trajectory])
        return [total_reward for i, t in enumerate(trajectory)]

    def add_to_memory(self, trajectory):
        weights = self.get_state_weights(trajectory)
        for (from_state, action, reward, to_state, done, q_value), weight in zip(trajectory, weights):
            if self.experience_replay_memory.shape[0] >= self.max_memory_size:
                min_element = np.argmin([exp['weight'] for exp in self.experience_replay_memory])
                self.experience_replay_memory =                     np.delete(self.experience_replay_memory, min_element)
            self.experience_replay_memory = np.append(self.experience_replay_memory, [
                {'from': from_state, 'action': action,
                 'reward': reward, 'done': done,
                 'to': to_state,
                 'q_value': q_value,
                 'weight': weight}])

    def softmax(self, logits):
        exps = np.exp(logits)
        return exps / np.sum(exps)

    def sample_from_memory(self):
        if self.experience_replay_memory.shape[0] > 1:
            weights = np.array([exp['weight'] for exp in self.experience_replay_memory])
            p = self.softmax(weights)
            return np.random.choice(self.experience_replay_memory,
                                    np.min([self.batch_size, self.experience_replay_memory.shape[0]]), p=p)
        else:
            return self.experience_replay_memory

    def create_multilayer_dense(self, scope, layer_input, layer_units, layer_activations, keep_probs=None,
                                regularizers=None, reuse_vars=None):
        with tf.variable_scope(scope, reuse=reuse_vars):
            last_layer = None
            if regularizers is None:
                regularizers = [None for _ in layer_units]
            if keep_probs is None:
                keep_probs = [1. for _ in layer_units]
            for i, (layer_size, activation, keep_prob, reg) in enumerate(zip(layer_units, layer_activations,
                                                                             keep_probs, regularizers)):
                if i == 0:
                    inp = layer_input
                else:
                    inp = last_layer
                last_layer = tf.layers.dense(inp, layer_size, activation, activity_regularizer=reg)
                if keep_prob != 1.0:
                    last_layer = tf.nn.dropout(last_layer, keep_prob)
        return last_layer

    def create_model(self):
        self.inputs = tf.placeholder(np.float32, [None, self.state_embedding_size], name='inputs')
        self.outputs = tf.placeholder(np.float32, [None, int(2**self.number_of_actions)], name='outputs')

        self.output_layer =             self.create_multilayer_dense('q_func', self.inputs, self.layer_units, self.layer_activations,
                                         self.layer_keep_probs, self.layer_regularizers)
        self.test_output_layer = self.create_multilayer_dense('q_func', self.inputs, self.layer_units,
                                                              self.layer_activations, reuse_vars=True)
        self.loss = tf.losses.mean_squared_error(self.outputs, self.output_layer, scope='q_func')

        trainable_variables = tf.trainable_variables('q_func')
        self.train_op = tf.train.AdamOptimizer(1e-3, name='optimizer').minimize(self.loss, var_list=trainable_variables)
        self.saver = tf.train.Saver()
        self.sess = tf.Session()

    def get_action(self, epoch, q_value):
        if random() < self.get_epsilon(epoch):
            action = self.env.action_space_sample()
        else:
            action = np.argmax(q_value)
            action = [int(a) for a in self.formatstr.format(action)]
        return action

    def generate_new_trajectories(self, epoch):
        for _ in range(self.n_generating_trajectories_per_epoch):
            observation = self.env.reset()
            done = False
            trajectory = []
            while not done:
                q_value = self.sess.run(self.test_output_layer, {self.inputs: [observation]})[0]
                action = self.get_action(epoch, q_value)
                new_observation, reward, done, info = self.env.step(action)
                trajectory.append((observation, action, reward, new_observation, done, q_value))
                observation = new_observation
                if len(trajectory) > self.max_trajectory_length:
                    break
            self.add_to_memory(trajectory)

    def create_batch(self):
        batch_q_values = []
        batch_observations = []
        for experience in self.sample_from_memory():
            action = experience['action']
            new_q_value = np.copy(experience['q_value'])
            new_q_value[action] = experience['reward']
            if not experience['done']:
                update_value = np.max(self.sess.run(self.output_layer, {self.inputs: [experience['to']]})[0])
                new_q_value[action] += self.gamma * update_value
            batch_q_values.append(new_q_value)
            batch_observations.append(experience['from'])
        return batch_observations, batch_q_values

    def train(self):
        epoch = 0
        while epoch < self.epochs:
            self.generate_new_trajectories(epoch)
            epoch_loss = None
            for sub_epoch_id in range(self.train_per_epoch):
                batch_observations, batch_q_values = self.create_batch()
                _, epoch_loss = self.sess.run((self.train_op, self.loss),
                                              {self.inputs: batch_observations, self.outputs: batch_q_values})
            self.save()
            epoch_total_reward = self.play()
            print(
                "*********** epoch %d ***********\n"
                "memory size: %d, mean-max state weights: %.3f\t%.3f\n"
                "total loss: %f\n"
                "total reward gained: %f\n"
                "epsilon: %.3f" % (epoch, self.experience_replay_memory.shape[0],
                                   np.mean([s['weight'] for s in self.experience_replay_memory]),
                                   np.max([s['weight'] for s in self.experience_replay_memory]),
                                   epoch_loss, epoch_total_reward, self.get_epsilon(epoch)))
            epoch += 1

    def play(self, render=False, monitor=False, max_timestep=None):
        total_reward = 0
        done = False
        observation = self.env.reset()
        reward = None
        timestep = 0
        if monitor:
            env = wrappers.Monitor(self.env, "./monitors/dqn/", force=True)
        else:
            env = self.env
        while not done:
            if render:
                env.render()
            q_value = self.sess.run(self.test_output_layer, {self.inputs: [observation]})[0]
            action = np.argmax(q_value)
            action = [int(a) for a in self.formatstr.format(action)]
            if timestep == self.max_trajectory_length:
                print(total_reward)
                break
            new_observation, reward, done, info = env.step(action)
            total_reward += reward
            timestep += 1
            observation = new_observation
            if done:
                break
            if max_timestep is not None:
                if timestep > max_timestep:
                    if monitor:
                        env.close()
                        env.reset()
                    break
        return total_reward

    def save(self):
        self.saver.save(self.sess, self.saving_path)

    def load(self):
        import os
        self.sess.run(tf.global_variables_initializer())
        self.sess.run(tf.local_variables_initializer())
        if not os.path.exists(self.saving_path):
            os.makedirs(self.saving_path)
        if not tf.train.checkpoint_exists(self.saving_path + 'checkpoint'):
            print('Saved temp_models not found! Randomly initialized.')
        else:
            self.saver.restore(self.sess, self.saving_path)
            print('Model loaded!')


def main(multiple, dnum):
    tr = True
    if not multiple:
        dnum = 1
    model = GymDQNLearner(multiple, dnum)
    if tr:
        model.train()
    episode_reward = model.play(False, False, 2000)
    print('total reward: %f' % episode_reward)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--multiple', action='store_true')
    parser.add_argument('--dnum', default=3)
    args = parser.parse_args()
    main(multiple=args.multiple, dnum=args.dnum)

