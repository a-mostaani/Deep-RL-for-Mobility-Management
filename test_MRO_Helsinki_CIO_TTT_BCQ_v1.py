"""Script by QL: completed and tested

Patent thread of socket client and 2 child threads:
1. listen: listen message in the background and put all it in the received message queue "receive_q"
2. run_agent: write message to config or kpi report, get the next config with predefined fixed values, and send back to
season

Experiment: Single cell DQN to optimize CIO and TTT
- define a source cell and its neighboring cells
- for the source cell i, optimize its CIOs to the neighboring cells {CIO_{i,j}: i = 1,2, ...} and TTT values TTT_i
- use DQN


!!! State and reward normalized based on the exploration phase

- State:
    1) per-cell number of ues
    2) per-cell actual load
    3) per-cell average throughput
    4) per-cell too-early HO
    5) per-cell too-late HO
    6) per-cell ping-pong HO
    7) per-cell wrong-cell HO

- action:
    1) TTT of source cell
    2) CIO of source cell to each of the neighboring cell

- reward:
    (sum_cell_too_early + sum_cell_too_late + 0.5* sum_cell_pingpong + 0.5*sum_cell_wrong + sum_cell_too_late)
    /sum_cell_number_of_ue


Note!!! both state and reward needs to be normalized!! we use the samples in exploration phase for normalization.

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
import copy
import pickle
from time import time
import matplotlib.pyplot as plt
import itertools
import copy
import collections
from collections import deque
import random
# import tensorflow as tf
# from tensorflow.keras.layers import Dense, Input, LeakyReLU, Dropout  # change
# from tensorflow.keras.optimizers import Adam
import os
from datetime import datetime
import torch


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
    """
    Replay Buffer:
    we randomly sample the batch of transitions from the replay buffer, which allows us to break
    the correlation between subsequent steps in the environment
    """
    def __init__(self, capacity=2100):
        self.buffer = deque(maxlen=capacity)
        self.min_replay_size = 200  # minimum replay size for start training   # change

    def store(self, state, action, reward, next_state):
        self.buffer.append([state, action, reward, next_state])

    def sample(self, batch_size=32):
        sample = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states = map(np.asarray, zip(*sample))
        states = np.array(states).reshape(batch_size, -1)
        next_states = np.array(next_states).reshape(batch_size, -1)
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

    def rescale_sample_in_buffer(self, mean_state, std_state, mean_reward, std_reward):  # change: this whole function is added
        for buf in self.buffer:
            for i in [0, 3]:
                buf[i] -= mean_state
                buf[i] /= std_state
            buf[2] -= mean_reward
            buf[2] /= std_reward


# class DQN:
#     def __init__(self, state_dim, action_dim):
#         """
#         :param num_CIOs: number of CIOs in the source cell
#         """
#         self.state_dim = state_dim
#         self.action_dim = action_dim
#         self.learning_rate = 0.001      # change
#         self.epsilon = 1   # epsilon-greedy algorithm, 1 means every step is random
#         self.epsilon_min = 0.01    # change
#         self.decay = 0.997  # change
#         self.model = self.nn_model()
#
#     def nn_model(self):
#         model = tf.keras.Sequential(
#             [Input((self.state_dim,)),
#              #Dense(32, activation="relu"),  # change commented out
#              #Dense(64, activation="relu"),
#              #Dense(256, activation="relu"),
#              #Dense(512, activation="relu"),
#              # Dense(512, activation="relu"),
#              Dense(32),  # change the layers
#              LeakyReLU(alpha=0.2),
#              Dropout(0.5),
#              Dense(64),
#              LeakyReLU(alpha=0.2),
#              Dropout(0.5),
#              Dense(256),
#              LeakyReLU(alpha=0.2),
#              Dropout(0.5),
#              Dense(512),
#              LeakyReLU(alpha=0.2),
#              Dropout(0.5),
#              Dense(512),
#              LeakyReLU(alpha=0.2),
#              Dropout(0.5),
#              Dense(self.action_dim),
#              ]
#         )
#         # 1) use leakyRelu as activation function: activation=partial(tf.nn.leaky_relu, alpha=0.01),
#         # 2) delete few layers, the network may easily overfitting
#         model.compile(loss="mse", optimizer=Adam(self.learning_rate))
#         return model
#
#     def predict(self, state):
#         return self.model.predict(state)
#
#     def get_action(self, state):
#         state = np.reshape(state, [1, self.state_dim])
#         self.epsilon *= self.decay
#         self.epsilon = max(self.epsilon, self.epsilon_min)
#         q_value = self.predict(state)[0]
#         if np.random.random() < self.epsilon:
#             print('[INFO]: explore random action...')   # change
#             return random.randint(0, self.action_dim-1)
#         print('[INFO]: model output action...')  # change
#         return np.argmax(q_value)
#
#     def train(self, states, targets):
#         self.model.fit(states, targets, epochs=1)


class RlAgent:
    def __init__(self):
        self.configs = []  # list of ho parameters for all trx's along time
        self.reports = []  # list of reports
        self.rep_new_state = []   # collection of reports to compute the new state (after a new config)
        self.rep_state = []       # collection of reports to compute the state (before a new config)
        self.num_train = 96*2   # 96*14 (3 days)two weeks of training    # change
        self.num_test = 5        # 2 days of the testing  # change
        # if 1 sample per 15 min (900s simulation time), 24 samples = 6h, train model every 6h, and target every day
        self.num_update_model = 12     # 24: no. of samples collected to train model with replay experience
        # self.num_update_target = 4*12   # 4*24: no. of samples collected to update the target model
        self.num_train_per_update = 30
        self.num_explore = 96*0.5  # change 96*2 is the original
        self.ls_ep_reward = []
        self.ep_idx = 0

        # each model parameters
        self.gamma = 0.7    # discount factor
        self.batch_size = 8   # change
        #
        self.current_cfg = None  # current cfg as a class
        self.prev_raw_report = None  # previous raw report.   # change: this line is added
        self.timestamp = 0
        self.ls_realtime = [time()]

        '''get dictionary of cells and their corresponding neighbors '''
        self.ls_cells = []
        self.dict_neighbor_cells = None

        self.path_to_site_json = 'C:/Users/Administrator/OneDrive - Nokia/Documents\Software/Season-2020d/SEASON II' \
                                 ' 2020d/SON_v2/config/scenario_config/' \
                                 'Helsinki_test-Sites-new.json'
        self.path_to_cfg_json = 'C:/Users/Administrator/OneDrive - Nokia/Documents/Software/Season-2020d/SEASON II 2020d/SON_v2/config/scenario_config' \
                                '/Helsinki_MRO_test_QL_25_05.json'
        '''
        self.path_to_site_json = 'C:/Users/liaoq/Documents/Software/SEASON II 2020d/SEASON II 2020d/SON_v2/config/' \
                                 'scenario_config/Helsinki_test-Sites.json'
        self.path_to_cfg_json = 'C:/Users/liaoq/Documents/Software/SEASON II 2020d/SEASON II 2020d/SON_v2/config/' \
                                'scenario_config/Helsinki_MRO_test_QL.json'
        '''

        self.path_mean_std = 'Results/state_kpi_mean_and_std.pickle'    # change add this line
        self.get_neighbor_cells()

        '''DQN to change TTT, and cell pair CIO'''
        self.src_cell = 'site04[0][A]'  # only get kpis of src cell and its neighboring cell
        self.nbr_cells = self.dict_neighbor_cells[self.src_cell]
        self.src_nbr_cells = [self.src_cell] + self.nbr_cells
        #self.norm_num_ues = self.get_group_size()    # change commented out

        '''reward'''
        self.ls_reward_kpi = ['early_handovers_count', 'wrong_cell_handovers_count', 'pingpong_handovers_count',
                              'late_handovers_count', 'total_number_of_ues']
        # change: make sure you have correct states here
        self.ls_state_kpi = ['number_of_connected_ues', 'actual_load', 'early_handovers_count',
                             'wrong_cell_handovers_count', 'pingpong_handovers_count', 'late_handovers_count']    # change
        # self.weight_kpi = [1, 1, 1, 0.5]   # [0.25, 0.15, 0.05, 0.5]   # change comment out these two lines
        # self.weight_cell = [0.4] + [0.6/len(self.nbr_cells)]*len(self.nbr_cells)
        self.state_dim = len(self.ls_state_kpi) * len(self.src_nbr_cells)
        self.state_mean = np.zeros(self.state_dim)    # change add this line
        self.state_std = np.ones(self.state_dim)   # change add this line
        self.reward_mean = 0    # change add this line
        self.reward_std = 1    # change add this line

        ''' Action space: HO parameters '''
        self.TTT_pool = [0.128, 0.512, 1.024, 2.56]  # in sec
        self.CIO_pool = [int(x) for x in [-5, -2, 0, 2, 5]]  # in dB
        ls_config = [self.TTT_pool] + [self.CIO_pool] * len(self.nbr_cells)
        self.permu_pool = list(itertools.product(*ls_config))
        self.action_dim = len(self.permu_pool)

        ''' DQN model initialization'''
        # self.dqn_model = DQN(self.state_dim, self.action_dim)
        # self.dqn_target_model = DQN(self.state_dim, self.action_dim)
        # self.update_target()
        self.current_action = None

        self.buffer = ReplayBuffer()
        self.last_train_bufsize = self.buffer.size()     # change: replace it with  cnt_sample

        self.save_path = 'Helsinki_BCQ_25_05_Test4.pickle'   # single cell DQN

        '''default HO parameters for other cells, HO parameter path'''
        self.default_offset = int(0)
        self.default_TTT = 0.512
        self.default_CIO = int(0)
        self.Offset_TTT_path = '*.cells.*.transceivers.*.HO_triggers'
        # note that 'HO_triggers' is a list of HO parameters, e.g., [{'event_type': 'A3', 'offset': 3, 'TTT': 1}]
        self.CIO_path = '*.cells.*.transceivers.*.cell_individual_HO_offset'

        self.last_msg = None
        self.actor = torch.load('trained_actor_1')
        self.vae = torch.load('trained_vae_1')
        self.critic = torch.load('trained_critic_1')

        # 'cell_individual_HO_offset' is a list of dictionaries, each element indicates CIO to a neighboring cell, e.g.
        # [{"trigger_type": "A3",
        #   "other_cell_trx": "site03[0][A]",
        #   "offset": 0},
        # {"trigger_type": "A3",
        #   "other_cell_trx": "site03[2][A]",
        #   "offset": 0
        #   }]

    # def update_target(self):
    #     weights = self.dqn_model.model.get_weights()
    #     self.dqn_target_model.model.set_weights(weights)

    # def replay_experience(self):
    #     for _ in range(self.num_train_per_update):
    #         states, actions, rewards, next_states = self.buffer.sample(self.batch_size)
    #         targets = self.dqn_target_model.predict(states)
    #         next_q_values = self.dqn_target_model.predict(next_states)[
    #             range(self.batch_size),
    #             np.argmax(self.dqn_model.predict(next_states), axis=1)
    #         ]
    #         targets[range(self.batch_size), actions] = (rewards + next_q_values * self.gamma)
    #         self.dqn_model.train(states, targets)

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

    # change: comment out: def get_group_size(self):

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
        # change: "self.reports.append(sub_df)" is commented out
        # change: from here till the end of "self.prev_raw_report = sub_df" is added
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

        self.prev_raw_report = sub_df
        self.last_msg = 'report'

        print('[INFO] number of collected report: ', len(self.reports))
        # chnage: commented out: self.rep_new_state.append(sub_df)

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
        state = self.get_dqn_state(self.rep_state[-1])
        state = self.normalize_state(state)  # change: this line is added
        # print("++++++++++++ state dimensions:", state.shape)
        '''new_state'''
        next_state = self.get_dqn_state(self.rep_new_state[0])  # change: the argument -1 is changed to 0
        next_state = self.normalize_state(next_state)   # change: this line is added

        '''reward'''
        arr_kpi = self.get_reward_kpis()  # change: large degree of changes from here till the end of method

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

    def get_dqn_state(self, df_rep):
        vec_state = np.concatenate([df_rep.loc[df_rep['unique_name'] == n,
                                               self.ls_state_kpi].values for n in self.src_nbr_cells], axis=1)  # change: , axis=1 is added
        # change: this line is removed:
        # vec_state[:, 0] = vec_state[:, 0] / self.norm_num_ues
        return vec_state.flatten().astype('float32')
    # change: the following three methods are added
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

        '''
        train and update the model when:
         1) enough samples in buffer is collected
         2) collected sample (configurations) smaller than the number of trainings 
        '''
        # if (self.buffer.size() >= self.buffer.min_replay_size) & (self.buffer.size() < self.num_train):  # change: self.buffer.size() is replaced with self.cnt_sample
        #     # training only when 1) periodically after collecting samples, 2) buffer size larger than the last training
        #     # This is to prevent delayed received report because of long communication time with SEASON
        #     # change: self.buffer.size() is replaced with self.cnt_sample and another if condition is added:
        #     # & (self.buffer.size() > self.last_train_bufsize)
        #     if (self.buffer.size() % self.num_update_model == 0) & (self.buffer.size() > self.last_train_bufsize):
                # self.replay_experience()
                # ep_reward = np.array(self.ls_ep_reward).mean()
                # # from the previous episode
                # print(f"Episode#{self.ep_idx} reward:{ep_reward}")
                # tf.summary.scalar("episode_reward", ep_reward, step=self.ep_idx)
                # self.ep_idx += 1
                # self.ls_ep_reward = []
                # # the following line is added
                # self.last_train_bufsize = self.buffer.size()  # buffer size of when last training

            # if self.buffer.size() % self.num_update_target == 0:  # change: cnt_sample is replaced with buffer.size() #done till here
            #     self.update_target()

        '''decide the next config'''
        # note that 'HO_triggers' is a list of dictionaries, e.g., [{'event_type': 'A3', 'offset': 3, 'TTT': 1}]
        '''get TTT and CIO_s,t and CIO_t,s'''
        if self.buffer.size() > self.num_explore:
            # change: the following 5 lines are added till the end of if clause
            state = self.get_dqn_state(self.rep_state[-1])
            state = self.normalize_state(state)
            "BCQ Decision Making"
            print("==================================")
            print("++++++ BCQ Decision Making ++++++")
            print("==================================")
            repeated_state = torch.FloatTensor(state.reshape(1, -1)).repeat(100, 1).to(torch.device("cpu"))
            print("repteated state dimensions:", repeated_state.size())
            candid_action = self.actor(repeated_state, self.vae.decode(repeated_state))
            q1 = self.critic.q1(repeated_state, candid_action)
            action_ind = q1.argmax(0)
            action = candid_action[action_ind].cpu().data.numpy().flatten()
            # Clipping action values
            action[0] = np.clip(action[0], 0.004, 5.12)
            action[1:4] = np.clip(action[1:4], -6, 6)
            action_value = action
            action_value = np.float64(action_value)



            # no exploration in the testing phase
            # if self.buffer.size() > self.num_train:
            #     self.dqn_model.epsilon = 0

            # action = self.dqn_model.get_action(state)  # change: the argument is changed to state
        else:
            action = random.randint(0, self.action_dim-1)
            print('[INFO]: explore random action...')  # change: this print is added
            action_value = self.permu_pool[action]

        self.current_action = action

        # action_value = self.permu_pool[action]
        # action_value = action
        # print("action values:", action_value)

        '''assign default offset and TTT values'''
        df_ofs_ttt = self.configs[-1][1].copy(deep=True)
        df_ofs_ttt['offset'] = self.default_offset
        df_ofs_ttt['TTT'] = self.default_TTT
        '''assign TTT to the source cell'''
        df_ofs_ttt.loc[df_ofs_ttt['src_cell'] == self.src_cell, 'TTT'] = action_value[0]
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
            self.current_cfg.dict_cio_values[self.src_cell][i]['offset'] = action_value[i+1]
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
                    # change: cnt_sample to buffer.size() and second if condition is added
                    if (self.agent.buffer.size() <= self.agent.num_train + self.agent.num_test) & \
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
        print("next_config type:", type(next_config))
        print("next_config:", next_config)
        cmd = json.dumps(next_config) + '\0'
        self.s.send(cmd.encode())


if __name__ == '__main__':
    port = 8000
    address = 'localhost'

    TC = ThreadedClient(address, port)
    logdir = os.path.join('logs', 'DQN', TC.agent.src_cell, datetime.now().strftime("%Y%m%d-%H%M%S"))
    print(f"Saving training logs to:{logdir}")
    # writer = tf.summary.create_file_writer(logdir)
    #
    # with writer.as_default():
    TC.start()
    print('Server started, port: ', port)
    TC.start_listen()
    TC.start_run_agent()

