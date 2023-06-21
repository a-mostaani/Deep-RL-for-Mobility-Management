"""Script by Arsham: working process

Patent thread of socket client and 2 child threads:
1. listen: listen message in the background and put all it in the received message queue "receive_q"
2. run_agent: write message to config or kpi report, get the next config with predefined fixed values, and send back to
season

Experiment: Single cell DDPG to optimize CIO and TTT
- define a source cell and its neighboring cells
- for the source cell i, optimize its CIOs to the neighboring cells {CIO_{i,j}: i = 1,2, ...} and TTT values TTT_i
- using DDPG

Simulation time in sec. with realtime factor 200, 15 mins (60*15=900s) is 4.5s in real time, but if the
simulator real time performance is ~100, then we have 9s per sample, and configuration change needs 35-40s
in local computer, i.e. 50s real time per configuration, for (1344+192)*50/3600 = 21.3 hour
"""


import socket
import struct
import threading
import queue
import time
import numpy as np
import pandas as pd
import json
import pickle
from time import time
import matplotlib.pyplot as plt
import itertools
import copy
import collections
from collections import deque
import random
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
print(tf.__version__)
import os
from datetime import datetime
from tensorflow.keras.layers import Dense, Input, Lambda, concatenate, LeakyReLU


def dict_merge(dct, merge_dct):
    """ Recursive dict merge. Inspired by :meth:``dict.update()``, instead of
    updating only top-level keys, dict_merge recurses down into dicts nested
    to an arbitrary depth, updating keys. The ``merge_dct`` is merged into
    ``dct``.
    :param dct: dict onto which the merge is executed
    :param merge_dct: dct merged into dct
    :return: None
    """
    for k, v in merge_dct.items():
        if (k in dct and isinstance(dct[k], dict)
                and isinstance(merge_dct[k], collections.Mapping)):
            dict_merge(dct[k], merge_dct[k])
        else:
            dct[k] = merge_dct[k]


class Configuration:
    # test: C = Configuration(TC.agent.current_cfg, '*.RRH_cells.*.transceivers.*.HO_triggers'.split('.'))
    def __init__(self, dict_cfg, ofs_ttt_path, cio_path, dict_neighbor_cells):
        self.dict_cfg = dict_cfg
        self.ofs_ttt_search_path = ofs_ttt_path
        self.cio_search_path = cio_path
        self.dict_neighbor_cells = dict_neighbor_cells
        #
        self.cfg_paths = []
        self.cfg_values = []
        """self.ofs_ttt_values: a list of per-cell offset and TTT configuration, note that 'HO_triggers' is a list of
        dictionaries, e.g., [{'event_type': 'A3', 'offset': 3, 'TTT': 1}]"""
        self.ofs_ttt_paths = []
        self.ofs_ttt_values = []
        """self.cio_values: a list of neighbor cio dictionaries: list of dictionaries, each element indicates CIO to a 
        neighboring cell, e.g. [{"trigger_type": "A3", "other_cell_trx": "site03[0][A]", "offset": 0}, 
        {"trigger_type": "A3", "other_cell_trx": "site03[2][A]", "offset": 0}]"""
        self.cio_paths = []
        self.cio_values = []
        self.dict_cio_values = None
        #
        self.df_ofs_ttt = pd.DataFrame(columns=['src_cell', 'offset', 'TTT'])
        self.df_cio = pd.DataFrame(columns=['src_cell', 'nbr_cell', 'CIO'])
        self.get_ofs_ttt_path()
        self.get_cio_path()
        #
        self.Groups = None
        # self.cnt = 0

        # keep initial group definition for incremental changes
        if self.Groups is None:
            self.Groups = copy.deepcopy(dict_cfg['Groups'])
        '''define traffic mask of 24 hours'''
        self.group24 = [0.8, 0.65, 0.4, 0.15, 0.2, 0.3, 0.6, 0.8, 0.9, 0.8, 1.2, 1.3,
                        1.5, 1.8, 2.0, 1.7, 1.8,  1.5, 1.4, 1.5, 1.1, 1.0, 0.9, 0.8]

    def get_path_values(self, tree, path, indx=0):
        """walks down a json tree and stores all possible paths and values of a Season configuration

            :param tree: dictionary containing json tree
            :param path: path specifier
            :param indx: index level
        """
        if indx < len(path):
            # not the final path end reached continue walk down
            if isinstance(path[indx], str):
                # print(path[indx])
                if path[indx] == '*':
                    # wildcard found iterate through set of options
                    for k in tree.keys():
                        # print(k)
                        new_path = path.copy()
                        new_path[indx] = k

                        # recursive walk down on next level of tree
                        self.get_path_values(tree[k], new_path, indx + 1)
                else:
                    # recursive walk down on next level of tree
                    self.get_path_values(tree[path[indx]], path, indx + 1)
        else:
            # final end of path reached keep full path string and value in separate arrays
            self.cfg_paths.append(path)
            self.cfg_values.append(tree)

    def get_ofs_ttt_path(self):
        self.cfg_paths = []
        self.cfg_values = []
        self.get_path_values(self.dict_cfg['Sites'], self.ofs_ttt_search_path)
        self.ofs_ttt_paths = copy.deepcopy(self.cfg_paths)
        self.ofs_ttt_values = copy.deepcopy(self.cfg_values)
        self.df_ofs_ttt['src_cell'] = [n[0] + '[' + n[2] + '][' + n[4] + ']' for n in self.ofs_ttt_paths]
        self.df_ofs_ttt['offset'] = [x[0]['offset'] for x in self.ofs_ttt_values]
        self.df_ofs_ttt['TTT'] = [x[0]['TTT'] for x in self.ofs_ttt_values]

    def get_cio_path(self):
        self.cfg_paths = []
        self.cfg_values = []
        self.get_path_values(self.dict_cfg['Sites'], self.cio_search_path)
        self.cio_paths = copy.deepcopy(self.cfg_paths)
        self.cio_values = copy.deepcopy(self.cfg_values)
        ls_src_cells = [n[0] + '[' + n[2] + '][' + n[4] + ']' for n in self.cio_paths]
        self.dict_cio_values = {k: v for k, v in zip(ls_src_cells, self.cio_values)}
        #
        ls_df_cio = []
        # It returns the first element of each list, then 2nd element of each list, etc.
        # This is a trick to consider the two lists as key and data to create a dictionary.
        for n, ls_nbr in zip(self.cio_paths, self.cio_values):
            sub_df = pd.DataFrame(columns=['src_cell', 'nbr_cell', 'CIO'])
            ls_nbr_cell = [d['other_cell_trx'] for d in ls_nbr]
            ls_cio = [d['offset'] for d in ls_nbr]
            sub_df['nbr_cell'] = ls_nbr_cell
            sub_df['CIO'] = ls_cio
            sub_df['src_cell'] = n[0] + '[' + n[2] + '][' + n[4] + ']'
            ls_df_cio.append(sub_df)
        self.df_cio = pd.concat(ls_df_cio)

        """E.g., cio_values = [{"trigger_type": "A3", "other_cell_trx": "site03[0][A]", "offset": 0}, 
        {"trigger_type": "A3", "other_cell_trx": "site03[2][A]", "offset": 0}]"""

    def get_cfg(self, ofs_ttt_values=None, cio_values=None, timestamp=None):
        """ get new Season configuration for given set of values

        :param values:   values: overwrites internal configuration values
        :param timestamp:  24 hour seasonality to be used for group size modification
        :return: new Season configuration
        """
        # get path values to per cell offset and ttt
        my_ofs_ttt_values = self.ofs_ttt_values if ofs_ttt_values is None else ofs_ttt_values
        cfg_ofs_ttt = {}
        for i in range(len(self.ofs_ttt_paths)):
            self.get_path_config(cfg_ofs_ttt, self.ofs_ttt_paths[i], my_ofs_ttt_values[i])
        # print('cfg ofs ttt:',  cfg_ofs_ttt)
        # get path values to per cell cio (to all neighbor list of the src cell)
        my_cio_values = self.cio_values if cio_values is None else cio_values
        cfg_cio = {}
        for i in range(len(self.cio_paths)):
            self.get_path_config(cfg_cio, self.cio_paths[i], my_cio_values[i])
        # print('cfg cio:',  cfg_cio)

        dict_merge(cfg_ofs_ttt, cfg_cio)

        # print(cfg_ofs_ttt)
        cfg = {'Sites': cfg_ofs_ttt}

        # change group sizes acc. to group24 parameter - adds 24h seasonality
        if len(self.group24) == 24 and timestamp is not None:
            new_groups = copy.deepcopy(self.Groups)
            # take time tick 0 as 0:00 a.m.
            indx = int((timestamp // 3600) % 24)
            print('[INFO]: change traffic mask config, hour of the day', indx)
            for group in new_groups.values():
                # modify all groups acc. group24 parameter
                group['parameters']['group_size'] = int(group['parameters']['group_size'] * self.group24[indx])

            cfg['Groups'] = new_groups

        return cfg

    def get_path_config(self, config, path, value):
        """set value of given configuration for path path and value

        :param config: Season configuration
        :param path: actual path
        :param value: actual value
        :return: None
        """
        if len(path) == 1:
            # final end of path reached, set value
            config[path[0]] = value
        else:
            # intermediate stage reached
            if path[0] not in config.keys():
                # generate empty dicts for all possible keys
                config[path[0]] = {}

            # walk down next level on path
            self.get_path_config(config[path[0]], path[1:], value)



class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
        self.min_replay_size = 200  # minimum replay size for start training
        self.batch_size = 8

    def store(self, state, action, reward, next_state):
        self.buffer.append([state, action, reward, next_state])

    def sample(self):
        sample = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(self.batch_size, -1)
        next_states = np.array(next_states).reshape(self.batch_size, -1)
        return states, actions, rewards, next_states

    def size(self):
        return len(self.buffer)

    def get_mean_std(self):  # change: this whole function is added
        """
        get the mean and std for state and reward
        :return:
        """
        state_dim = len(self.buffer[0][0])
        ls_state = [x[0] for x in self.buffer]
        arr_state = np.concatenate(ls_state).reshape((-1, state_dim))
        mean_state = arr_state.mean(axis=0)
        std_state = arr_state.std(axis=0)
        # if the std = 0, then we need to assign a nonzeros offset
        std_state[np.where(std_state == 0)] = 1

        arr_reward = np.array([x[2] for x in self.buffer])
        mean_reward = arr_reward.mean()
        std_reward = arr_reward.std()
        return mean_state, std_state, mean_reward, std_reward

    def rescale_sample_in_buffer(self, mean_state, std_state, mean_reward, std_reward):   # change: This whole function is added
        for buf in self.buffer:
            for i in [0, 3]:
                buf[i] -= mean_state
                buf[i] /= std_state
            buf[2] -= mean_reward
            buf[2] /= std_reward

class Actor:
    def __init__(self, state_dim, action_dim, action_bound):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_bound = action_bound  #Action bound is normalized to 1. We then multiply each output layer to proper coefficients.
        #todo: multiply proper coefficients as mentioned above.
        self.actor_lr = 0.001  # input
        self.model = self.nn_model()
        self.opt = tf.keras.optimizers.Adam(self.actor_lr)

    def nn_model(self):
        return tf.keras.Sequential(
            [
                Input((self.state_dim,)),
                Dense(32, activation="relu"),
                Dense(32, activation="relu"),
                Dense(self.action_dim, activation="tanh"),
                # Lambda(lambda x: x * self.action_bound),
            ]
        )

    def train(self, states, q_grads):
        with tf.GradientTape() as tape:
            grads = tape.gradient(
                self.model(states), self.model.trainable_variables, -q_grads
            )
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))

    def predict(self, state):
        return self.model.predict(state)

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        return self.model.predict(state)[0]


class Critic:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.model = self.nn_model()
        self.critic_lr = 0.001  # input
        self.opt = tf.keras.optimizers.Adam(self.critic_lr)

    def nn_model(self): #todo: optimize your actor and critc networks
        state_input = Input((self.state_dim,))
        s1 = Dense(64, activation="relu")(state_input)
        s2 = Dense(32, activation="relu")(s1)
        action_input = Input((self.action_dim,))
        a1 = Dense(32, activation="relu")(action_input)
        c1 = concatenate([s2, a1], axis=-1)
        c2 = Dense(16, activation="relu")(c1)
        output = Dense(1, activation="linear")(c2)
        return tf.keras.Model([state_input, action_input], output)

    def predict(self, inputs):
        return self.model.predict(inputs)

    def q_gradients(self, states, actions):
        actions = tf.convert_to_tensor(actions)
        with tf.GradientTape() as tape:
            tape.watch(actions)
            q_values = self.model([states, actions])
            q_values = tf.squeeze(q_values)
        return tape.gradient(q_values, actions)

    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def train(self, states, actions, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model([states, actions], training=True)
            assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def get_action(self, state):
        state = np.reshape(state, [1, self.state_dim])
        return self.model.predict(state)[0]


class RlAgent:
    def __init__(self):
        self.configs = []  # list of ho parameters for all trx's along time
        self.reports = []  # list of reports
        self.rep_new_state = []   # collection of reports to compute the new state (after a new config)
        self.rep_state = []       # collection of reports to compute the state (before a new config)
        self.num_train = 96*14      # 96*14 two weeks of training
        self.num_test = 96*2        # 96 days of the testing
        # if 1 sample per 15 min (900s simulation time), 24 samples = 6h, train model every 6h, and target every day
        self.num_update_model = 12     # 24: no. of samples collected to train model with replay experience
        self.num_update_target = 4*12   # 4*24: no. of samples collected to update the target model
        self.num_train_per_update = 30  # input
        self.num_explore = 96*2
        self.ls_ep_reward = []
        self.ep_idx = 0
        self.gamma = 0.99  # input
        self.tau = 0.05  # input
        self.min_TTT = 0.004
        self.max_TTT = 5.12
        self.min_CIO = -6
        self.max_CIO = 6

        # each model parameters
        self.gamma = 0.7    # discount factor
        self.batch_size = 8
        #
        self.current_cfg = None  # current cfg as a class
        self.prev_raw_report = None  # previous raw report.
        self.timestamp = 0
        self.ls_realtime = [time()]

        '''get dictionary of cells and their corresponding neighbors '''
        self.ls_cells = []
        self.dict_neighbor_cells = None
        self.path_to_site_json = 'C:/Users/Administrator/OneDrive - Nokia/Documents\Software/Season-2020d/SEASON II' \
                                 ' 2020d/SON_v2/config/scenario_config/' \
                                 'Helsinki_test-Sites.json'
        self.path_to_cfg_json = 'C:/Users/Administrator/OneDrive - Nokia/Documents/Software/Season-2020d/SEASON II 2020d/SON_v2/config/scenario_config' \
                                '/Helsinki_MRO_test_QL.json'
        self.path_mean_std = 'Results/state_kpi_mean_and_std.pickle'    # new line
        self.get_neighbor_cells()

        '''DDPG to change TTT, and cell pair CIO'''
        self.src_cell = 'site04[0][A]'  # only get kpis of src cell and its neighboring cell
        self.nbr_cells = self.dict_neighbor_cells[self.src_cell]
        self.src_nbr_cells = [self.src_cell] + self.nbr_cells
        #self.norm_num_ues = self.get_group_size()   # number of ues for the normalization of num_connected ues in state

        '''reward'''
        self.ls_reward_kpi = ['early_handovers_count', 'wrong_cell_handovers_count', 'pingpong_handovers_count',
                              'late_handovers_count', 'total_number_of_ues']
        self.ls_state_kpi = ['number_of_connected_ues', 'actual_load']
        # self.weight_kpi = [0.25, 0.15, 0.05, 0.5]
        # self.weight_cell = [0.4] + [0.6/len(self.nbr_cells)]*len(self.nbr_cells)
        ''' State: load and num ues for src and nbr cells'''
        self.state_dim = len(self.ls_state_kpi) * len(self.src_nbr_cells)  # load and number of ues for src and neighboring cells
        self.state_mean = np.zeros(self.state_dim)    # change add this line
        self.state_std = np.ones(self.state_dim)   # change add this line
        self.reward_mean = 0    # change add this line
        self.reward_std = 1    # change add this line

        ''' Action space: HO parameters '''
        # self.TTT_pool = [0.128, 0.512, 1.024, 2.56]  # in sec
        # self.CIO_pool = [int(x) for x in [-5, -2, 0, 2, 5]]  # in dB
        # ls_config = [self.TTT_pool] + [self.CIO_pool] * len(self.nbr_cells)
        # self.permu_pool = list(itertools.product(*ls_config))

        self.action_dim = 2*(len(self.nbr_cells)+1) + (len(self.nbr_cells)+1)
        self.action_bound = 1

        ''' DDDPG model initialization'''
        self.actor = Actor(self.state_dim, self.action_dim, self.action_bound)
        self.critic = Critic(self.state_dim, self.action_dim)

        self.target_actor = Actor(self.state_dim, self.action_dim, self.action_bound)
        self.target_critic = Critic(self.state_dim, self.action_dim)
        self.update_target()
        self.current_action = None

        self.buffer = ReplayBuffer()
        self.last_train_bufsize = self.buffer.size()

        self.save_path = 'Helsinki_SC-DDPG_040A.pickle'   # single cell DDPG

        '''default HO parameters for other cells, HO parameter path'''
        self.default_offset = int(0)
        self.default_TTT = 0.512
        self.default_CIO = int(0)
        self.Offset_TTT_path = '*.cells.*.transceivers.*.HO_triggers'
        # note that 'HO_triggers' is a list of HO parameters, e.g., [{'event_type': 'A3', 'offset': 3, 'TTT': 1}]
        self.CIO_path = '*.cells.*.transceivers.*.cell_individual_HO_offset'

        self.last_msg = None

        # 'cell_individual_HO_offset' is a list of dictionaries, each element indicates CIO to a neighboring cell, e.g.
        # [{"trigger_type": "A3",
        #   "other_cell_trx": "site03[0][A]",
        #   "offset": 0},
        # {"trigger_type": "A3",
        #   "other_cell_trx": "site03[2][A]",
        #   "offset": 0
        #   }]

    def update_target(self):
        # actor_weights = self.actor.model.get_weights()
        # self.target_actor.model.set_weights(actor_weights)
        # critic_weights = self.critic.model.get_weights()
        # self.target_critic.model.set_weights(critic_weights)
        actor_weights = self.actor.model.get_weights()
        t_actor_weights = self.target_actor.model.get_weights()
        critic_weights = self.critic.model.get_weights()
        t_critic_weights = self.target_critic.model.get_weights()

        for i in range(len(actor_weights)):
            t_actor_weights[i] = (
                self.tau * actor_weights[i] + (1 - self.tau) * t_actor_weights[i]
            )

        for i in range(len(critic_weights)):
            t_critic_weights[i] = (
                self.tau * critic_weights[i] + (1 - self.tau) * t_critic_weights[i]
            )

        self.target_actor.model.set_weights(t_actor_weights)
        self.target_critic.model.set_weights(t_critic_weights)

    def get_td_target(self, rewards, q_values):
        targets = np.asarray(q_values)
        for i in range(q_values.shape[0]):
            # if dones[i]:
            #     targets[i] = rewards[i]
            # else:
            targets[i] = self.gamma * q_values[i]
        return targets

    def add_ou_noise(self, x, rho=0.15, mu=0, dt=1e-1, sigma=0.2, dim=1):
        return (
            x + rho * (mu - x) * dt + sigma * np.sqrt(dt) * np.random.normal(size=dim)
        )

    def replay_experience(self):
        for _ in range(self.num_train_per_update):
            states, actions, rewards, next_states = self.buffer.sample()
            target_q_values = self.target_critic.predict(
                [next_states, self.target_actor.predict(next_states)]
            )
            td_targets = self.get_td_target(rewards, target_q_values)
            self.critic.train(states, actions, td_targets)

            s_actions = np.ones([self.batch_size, self.action_dim])
            for i in range(self.batch_size):
                s_actions[i] = self.actor.predict(states[i, :].reshape(1, 12))

            s_grads = self.critic.q_gradients(states, s_actions)
            grads = np.array(s_grads).reshape((-1, self.action_dim))
            states = states.astype('float32')
            grads = grads.astype('float32')
            self.actor.train(states, grads)
            #self.update_target()  # todo: should I update the target network here or not?

    def get_neighbor_cells(self):
        dict_sites = json.load(open(self.path_to_site_json))
        ls_key = []
        ls_neighbors = []
        for k, v in dict_sites.items():
            for k_cell, v_cell in v['cells'].items():
                for k_sec, v_sec in v_cell['transceivers'].items():
                    ls_key.append(k + '[' + k_cell + '][' + k_sec + ']')
                    ls_neighbors.append(v_sec['neighbour_relations'])
        self.dict_neighbor_cells = {k: v for k, v in zip(ls_key, ls_neighbors)}
        self.ls_cells = list(self.dict_neighbor_cells.keys())

    # def get_group_size(self):
    #     """
    #     get the average number of ues per site for the state normalization
    #     :return:
    #     """
    #     dict_cfg = json.load(open(self.path_to_cfg_json))
    #     ave_num_ue = 0
    #     for k, v in dict_cfg['Groups'].items():
    #         ave_num_ue += v['parameters']['group_size']
    #     # average number of ues in the whole playground/number of sites
    #     ave_num_ue = ave_num_ue/len(self.dict_neighbor_cells)
    #     return ave_num_ue

    def add_config(self, dict_cfg):
        """ add new configurtaion to the list

        :param dict_cfg: dictionary of actual season configuration
        :return:
        """
        # change: from here till cfg is added
        # when receives a next config, the state before the config becomes state, and waiting for the states after to be
        # the new state
        # sometimes simulator sends two configs consecutively. then rep_state =[] and after receiving
        #  a KPI report len(rep_new_state) = 1, therefore we update rep_state only when rep_new_state is not empty
        if len(self.reports) > 0:
            if len(self.rep_new_state) > 0:
                self.rep_state = self.rep_new_state.copy()
            else:
                self.rep_state = [self.reports[-1].copy()]
        self.rep_new_state = []

        cfg = Configuration(dict_cfg, self.Offset_TTT_path.split('.'), self.CIO_path.split('.'), self.dict_neighbor_cells)
        self.configs.append([self.timestamp, cfg.df_ofs_ttt, cfg.df_cio])
        # self.cfg_timestamp.append(self.timestamp)   # this is the timestamp of the last KPI report
        print('[INFO] number of collected config: ', len(self.configs))
        self.current_cfg = cfg
        self.last_msg = 'config'

    def add_report(self, dict_rpt):
        # write report dict in a sub df
        self.ls_realtime.append(time())
        print('[INFO] real time interval between the reports: ', self.ls_realtime[-1] - self.ls_realtime[-2])
        # simulator timestamp
        self.timestamp = dict_rpt['kpi_report']['all_cell_trx_kpis']['timestamp']
        #
        ls_kpi_names = dict_rpt['kpi_report']['all_cell_trx_kpis']['kpi_names']
        ls_kpi_values = dict_rpt['kpi_report']['all_cell_trx_kpis']['kpi_values']
        sub_df = pd.DataFrame(columns=['timestamp'] + ls_kpi_names)

        num_trx = len(dict_rpt['kpi_report']['all_cell_trx_kpis']['kpi_values'])
        for idx_trx in range(num_trx):
            ls_trx_values = [self.timestamp] + ls_kpi_values[idx_trx]
            sub_df.at[idx_trx, :] = ls_trx_values
        # self.reports.append(sub_df)

        # subtract the HO counts from the previous report, since counts accumulated along the time
        proc_sub_df = sub_df.copy(deep=True)
        if self.prev_raw_report is not None:
            for col in list(proc_sub_df.columns):
                if 'count' in col:
                    proc_sub_df[col] = proc_sub_df[col].values - self.prev_raw_report[col].values

        self.reports.append(proc_sub_df)
        self.rep_new_state.append(proc_sub_df)
        print('[INFO] number of state report: ', len(self.rep_state))
        print('[INFO] number of new state report: ', len(self.rep_new_state))

        self.last_msg = 'report'
        print('[INFO] number of collected report: ', len(self.reports))
        self.rep_new_state.append(sub_df)

    def add_replay_sample(self):
        """
        - state, action, reward, next_state
        in this case we do not have episode, continuous training with one environment, we do not reset the environment
        :return:
        """

        '''action'''
        action = self.current_action   # index of the action
        '''state'''
        # get state (before taking the action): number of ues and actual load for src and nbr stations
        state = self.get_ddpg_state(self.rep_state[-1])
        state = self.normalize_state(state)
        '''new_state'''
        next_state = self.get_ddpg_state(self.rep_new_state[0])
        next_state = self.normalize_state(next_state)


        '''reward'''
        arr_kpi = self.get_reward_kpis()
        # if arr_kpi.sum(axis=0)[4] > 0:
        #     reward = - arr_kpi.sum(axis=0)[:4] @ np.array([1, 0.5, 0.5, 1]) / arr_kpi.sum(axis=0)[4]
        #     reward = self.normalize_reward(reward)
        #     self.buffer.store(state, action, reward, next_state)
        #     self.ls_ep_reward.append(reward)
        #
        # self.cnt_sample += 1
        if arr_kpi.sum(axis=0)[4] > 0:
            reward = - arr_kpi.sum(axis=0)[:4] @ np.array([1, 0.5, 0.5, 1]) / arr_kpi.sum(axis=0)[4]
            reward = self.normalize_reward(reward)
            self.buffer.store(state, action, reward, next_state)
            self.ls_ep_reward.append(reward)

            ''' --- Right after exploration: get mean and std for normalization & normalize samples in buffer  --- '''
            if len(self.buffer.buffer) == self.num_explore:
                self.state_mean, self.state_std, self.reward_mean, self.reward_std = self.buffer.get_mean_std()
                print('[INFO]: Finished exploration, reward mean = {} and reward std = {}'.format(self.reward_mean,
                                                                                                  self.reward_std))
                # rescale/normalize state and reward samples in buffer
                self.buffer.rescale_sample_in_buffer(self.state_mean, self.state_std, self.reward_mean, self.reward_std)
                # rescale/normalize the rewards in the ls_ep_reward
                self.ls_ep_reward = [(x[2] - self.reward_mean)/self.reward_std for x in self.buffer.buffer]

            print('[Info]: add replay sample {}, action index is {} and reward is {}'.format(self.buffer.size(),
                                                                                             action, reward))

    def get_ddpg_state(self, df_rep):
        vec_state = np.concatenate([df_rep.loc[df_rep['unique_name'] == n,
                                               self.ls_state_kpi].values for n in self.src_nbr_cells], axis=1)
        # vec_state[:, 0] = vec_state[:, 0] / self.norm_num_ues
        return vec_state.flatten().astype('float32')

    def normalize_state(self, state):
        state = np.divide(state - self.state_mean, self.state_std)
        return state

    def normalize_reward(self, reward):
        reward -= self.reward_mean
        reward /= self.reward_std
        return reward

    def get_reward_kpis(self):
        df_rwd = self.rep_new_state[0]
        arr_kpi = np.concatenate([df_rwd.loc[df_rwd['unique_name'] == n, self.ls_reward_kpi].values
                                  for n in self.src_nbr_cells])
        arr_kpi = arr_kpi.clip(min=0)
        return arr_kpi

    def next_config(self, timestamp=None):
        """

        :param timestamp:
        :return:
        """
        # when defines a next config, the state before the config becomes state, and waiting for the states after to be
        # the new state
        self.rep_state = self.rep_new_state.copy()
        self.rep_new_state = []
        bg_noise = np.zeros(self.action_dim)

        '''
        train and update the model when:
         1) enough samples in buffer is collected
         2) collected sample (configurations) smaller than the number of trainings 
        '''
        if (self.buffer.size() >= self.buffer.min_replay_size) & (self.buffer.size() < self.num_train):
            if (self.buffer.size() % self.num_update_model == 0) & (self.buffer.size() > self.last_train_bufsize):
                self.replay_experience()
                ep_reward = np.array(self.ls_ep_reward).mean()
                # from the previous episode
                print(f"Episode#{self.ep_idx} reward:{ep_reward}")
                tf.summary.scalar("episode_reward", ep_reward, step=self.ep_idx)
                self.ep_idx += 1
                self.ls_ep_reward = []

            if self.buffer.size() % self.num_update_target == 0:
                self.update_target()

        # creating and adding exploration noise to the actions
        noise = self.add_ou_noise(bg_noise, dim=self.action_dim)

        '''decide the next config'''
        # note that 'HO_triggers' is a list of dictionaries, e.g., [{'event_type': 'A3', 'offset': 3, 'TTT': 1}]
        '''get TTT and CIO_s,t and CIO_t,s'''
        if self.buffer.size() > self.num_explore:
            state = self.get_ddpg_state(self.rep_state[-1])
            state = self.normalize_state(state)

            # no exploration in the testing phase
            if self.buffer.size() > self.num_train:
                noise = np.zeros(self.action_dim)
            action = self.actor.get_action(state)
        else:
            action = (np.random.random(self.action_dim)-0.5) * (self.max_CIO - self.min_CIO)
            action[0] = np.random.random(1) * self.max_TTT
            print('[INFO]: explore random action...')

        #todo: actions should be resclaed after the normalization

        # creating and adding exploration noise to the actions

        action = np.clip(
            action + noise, -self.action_bound, self.action_bound
                        )
        self.current_action = action
        # action_value = self.permu_pool[action]

        '''assign default offset and TTT values'''
        df_ofs_ttt = self.configs[-1][1].copy(deep=True)
        df_ofs_ttt['offset'] = self.default_offset
        df_ofs_ttt['TTT'] = self.default_TTT
        '''assign TTT to the source cell'''
        df_ofs_ttt.loc[df_ofs_ttt['src_cell'] == self.src_cell, 'TTT'] = float(tf.clip_by_value(action[0], self.min_TTT, self.max_TTT))
        # print('[INFO] next config ttt = ', action_value[0])
        ls_ofs_ttt = [[{'event_type': 'A3', 'offset': int(o), 'TTT': float(t)}]
                      for o, t in zip(df_ofs_ttt['offset'].values, df_ofs_ttt['TTT'].values)]

        '''initialize all cells CIOs with default CIO value: after receiving the first configuration'''
        if len(self.configs) == 1:
            # set all cio to default values
            # [{"trigger_type": "A3",
            #   "other_cell_trx": "site03[0][A]",
            #   "offset": 0},
            # {"trigger_type": "A3",
            #   "other_cell_trx": "site03[2][A]",
            #   "offset": 0
            #   }]
            for k, v in self.current_cfg.dict_cio_values.items():
                for i in range(len(self.current_cfg.dict_cio_values[k])):
                    self.current_cfg.dict_cio_values[k][i]['offset'] == self.default_CIO

        '''assign CIOs for every pair of (source cell, target cell)'''
        # print('[INFO] next config src cell cio = ', action_value[1:])
        for i in range(len(self.current_cfg.dict_cio_values[self.src_cell])):
            self.current_cfg.dict_cio_values[self.src_cell][i]['offset'] = float(tf.clip_by_value(action[i+1], self.min_CIO, self.max_CIO))
            # action_value[0] is TTT, CIO values starts from action_value[1:]

        ls_cio = list(self.current_cfg.dict_cio_values.values())

        return self.current_cfg.get_cfg(ofs_ttt_values=ls_ofs_ttt, cio_values=ls_cio, timestamp=timestamp)

    def save_results(self):
        pickle.dump((self.configs, self.reports, self.buffer.buffer), open(self.save_path, 'wb'))


class ThreadedClient(threading.Thread):
    def __init__(self, host, port):
        threading.Thread.__init__(self)
        # set up queues
        self.receive_q = queue.Queue()
        self.send_q = queue.Queue()
        self.msgs = ''
        self.flag_run = True
        # self.last_msg = 'report'
        # declare instance variables
        self.host = host
        self.port = port
        # connect to socket
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.host, self.port))
        self.s.settimeout(.1)
        self.agent = RlAgent()
        # self.current_config = None
        # self.report_cnt = 0

    '''LISTEN to season, but if send_q has new config, send to season'''
    def listen(self):
        """ listen to season and add the config and report messages in the queue
        :return:
        """
        while self.flag_run:
            try:
                # if there is a updated configuration, then push config to season
                if not self.send_q.empty():
                    print('[INFO] send a new configuration...')
                    self.send_config()

                # print('listening...')
                m = self.s.recv(26240000).decode("utf-8").split('\0')
                if 'config' in m[0][:20]:
                    print('[INFO] RECEIVED: config head message...')
                elif 'kpi_report' in m[0][:20]:
                    print('[INFO] RECEIVED: KPI head message...')

                if ('kpi_report' in m[0][:20]) or ('config' in m[0][:20]):
                    self.msgs = m[0]
                else:
                    self.msgs += m[0]

                try:
                    msg_json = json.loads(self.msgs)
                    # put all kpi_reports in the queue
                    self.receive_q.put(msg_json)
                    # print('report queue size: ', self.receive_q.qsize())
                    self.msgs = ''
                except:
                    pass

            except socket.timeout:
                pass

    def start_listen(self):
        t_listen = threading.Thread(target=self.listen)
        t_listen.start()
        print('started listen')

    '''RUN AGENT: write received messages to report and config, decide next config, put in to the send_q queue'''
    def run_agent(self):
        # keep reading self.receive_q and collect the report into a data frame
        while self.flag_run:
            try:
                # get report dict from the queue
                dict_msg = self.receive_q.get()
                if list(dict_msg.keys())[0] == 'kpi_report':
                    """add report to the RL Agent"""
                    if self.agent.last_msg == 'config':
                        flag_add_sample = True
                    else:
                        flag_add_sample = False

                    self.agent.add_report(dict_msg)

                    if flag_add_sample & (len(self.agent.configs) >= 2):
                        self.agent.add_replay_sample()

                    if self.agent.buffer.size() <= self.agent.num_train + self.agent.num_test & \
                            (len(self.agent.configs) >= 1):
                        next_config = self.agent.next_config()
                        self.send_q.put(next_config)
                    else:
                        self.flag_run = False
                        self.agent.save_results()

                elif list(dict_msg.keys())[0] == 'config':
                    self.agent.add_config(dict_msg['config'])

            except queue.Empty:
                pass

    def start_run_agent(self):
        t_agent = threading.Thread(target=self.run_agent)
        t_agent.start()
        print('started run agent')

    def send_config(self):
        next_config = self.send_q.get()
        cmd = json.dumps(next_config) + '\0'
        self.s.send(cmd.encode())


if __name__ == '__main__':
    port = 8000
    address = 'localhost'

    TC = ThreadedClient(address, port)
    logdir = os.path.join('logs', 'DDPG', TC.agent.src_cell, datetime.now().strftime("%Y%m%d-%H%M%S"))
    print(f"Saving training logs to:{logdir}")
    writer = tf.summary.create_file_writer(logdir)

    with writer.as_default():
        TC.start()
        print('Server started, port: ', port)
        TC.start_listen()
        TC.start_run_agent()
