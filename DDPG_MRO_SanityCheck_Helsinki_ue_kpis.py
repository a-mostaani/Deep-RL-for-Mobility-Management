"""Script by QL: working in process

Patent thread of socket client and 2 child threads:
1. listen: listen message in the background and put all it in the received message queue "receive_q"
2. run_agent: write message to config or kpi report, get the next config with predefined fixed values, and send back to
season

Experiment: randomly choose
"""

import socket
import struct
import threading
import queue
import time
import pandas as pd
import json
import copy
import pickle
from time import time
import matplotlib.pyplot as plt
import itertools
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class Configuration:

    # test: C = Configuration(TC.agent.current_cfg, '*.RRH_cells.*.transceivers.*.HO_triggers'.split('.'))
    def __init__(self, dict_cfg, cfg_path):
        self.cfg_paths = []
        self.cfg_values = []
        self.ho_offsets = []
        self.ho_ttt = []
        self.df_cfg_ho = pd.DataFrame(columns=['unique_name', 'offset', 'TTT'])
        self.Groups = None
        self.cnt = 0
        # self.get_path_values(dict_cfg['RRH'], cfg_path)
        self.get_path_values(dict_cfg['Sites'], cfg_path)
        self.get_cfg_df()
        # keep initial group definition for incremental changes
        if self.Groups is None:
            self.Groups = copy.deepcopy(dict_cfg['Groups'])
        '''define traffic mask of 24 hours'''
        #Todo: import traffic data here
        traffic_data = pd.read_csv('Traffic_Mask_Plot.csv')
        self.group24 = list(traffic_data.values[i][2] for i in range(96))
        #self.group24 = [0.8, 0.65, 0.4, 0.15, 0.2, 0.3, 0.6, 0.8, 0.9, 0.8, 1.2, 1.3,
        #               1.5, 1.8, 2.0, 1.7, 1.8, 1.5, 1.4, 1.5, 1.1, 1.0, 0.9, 0.8]

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
            # self.values.append(int(tree))
            '''note that 'HO_triggers' is a list of dictionaries, e.g., [{'event_type': 'A3', 'offset': 3, 'TTT': 1}]'''
            self.cfg_values.append(tree)
            self.ho_offsets.append(tree[0]['offset'])
            self.ho_ttt.append(tree[0]['TTT'])

    def get_cfg_df(self):
        ls_site_uniquename = [n[0] + '[' + n[2] + '][' + n[4] + ']' for n in self.cfg_paths]
        self.df_cfg_ho['unique_name'] = ls_site_uniquename
        self.df_cfg_ho['offset'] = self.ho_offsets
        self.df_cfg_ho['TTT'] = self.ho_ttt

    def get_cfg(self, values=None, timestamp=None):
        """ get new Season configuration for given set of values

        :param values:   values: overwrites internal configuration values
        :param timestamp:  24 hour seasonality to be used for group size modification
        :return: new Season configuration
        """
        myvalues = self.cfg_values if values is None else values
        cfg = {}

        for i in range(len(self.cfg_paths)):
            self.get_path_config(cfg, self.cfg_paths[i], myvalues[i])

        # cfg = {'RRH': cfg}
        cfg = {'Sites': cfg}
        # change group sizes acc. to group24 parameter - adds 24h seasonality
        if len(self.group24) == 24 and timestamp is not None:
            new_groups = copy.deepcopy(self.Groups)
            # take time tick 0 as 0:00 a.m.
            indx = int((timestamp // 3600) % 96)
            print('[INFO]: change traffic mask config, hour of the day', indx)
            for group in new_groups.values():
                # modify all groups acc. group24 parameter
                ##Todo: replace group24 with the traffic data : Resolved
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

class Buffer:
    def __init__(self, buffer_capacity=100000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # It tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.action_buffer = np.zeros((self.buffer_capacity, num_actions))
        self.reward_buffer = np.zeros((self.buffer_capacity, 1))
        self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,a,r,s') obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        index = self.buffer_counter % self.buffer_capacity

        self.state_buffer[index] = obs_tuple[0]
        self.action_buffer[index] = obs_tuple[1]
        self.reward_buffer[index] = obs_tuple[2]
        self.next_state_buffer[index] = obs_tuple[3]

        self.buffer_counter += 1

    # Eager execution is turned on by default in TensorFlow 2. Decorating with tf.function allows
    # TensorFlow to build a static graph out of the logic and computations in our function.
    # This provides a large speed up for blocks of code that contain many small TensorFlow operations such as this one.
    @tf.function
    def update(
        self, state_batch, action_batch, reward_batch, next_state_batch,
    ):
        # Training and updating Actor & Critic networks.
        # See Pseudo Code.
        with tf.GradientTape() as tape:
            target_actions = target_actor(next_state_batch, training=True)
            y = reward_batch + gamma * target_critic(
                [next_state_batch, target_actions], training=True
            )
            critic_value = critic_model([state_batch, action_batch], training=True)
            critic_loss = tf.math.reduce_mean(tf.math.square(y - critic_value))

        critic_grad = tape.gradient(critic_loss, critic_model.trainable_variables)
        critic_optimizer.apply_gradients(
            zip(critic_grad, critic_model.trainable_variables)
        )

        with tf.GradientTape() as tape:
            actions = actor_model(state_batch, training=True)
            critic_value = critic_model([state_batch, actions], training=True)
            # Used `-value` as we want to maximize the value given
            # by the critic for our actions
            actor_loss = -tf.math.reduce_mean(critic_value)

        actor_grad = tape.gradient(actor_loss, actor_model.trainable_variables)
        actor_optimizer.apply_gradients(
            zip(actor_grad, actor_model.trainable_variables)
        )

    # We compute the loss and update parameters
    def learn(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = tf.convert_to_tensor(self.state_buffer[batch_indices])
        action_batch = tf.convert_to_tensor(self.action_buffer[batch_indices])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[batch_indices])
        reward_batch = tf.cast(reward_batch, dtype=tf.float32)
        next_state_batch = tf.convert_to_tensor(self.next_state_buffer[batch_indices])

        self.update(state_batch, action_batch, reward_batch, next_state_batch)


# This update target parameters slowly
# Based on rate `tau`, which is much less than one.
@tf.function
def update_target(target_weights, weights, tau):
    for (a, b) in zip(target_weights, weights):
        a.assign(b * tau + a * (1 - tau))



class RlAgent:
    def __init__(self):
        self.configs = []  # list of ho parameters for all trx's along time
        self.reports = []
        self.phases = []
        self.current_cfg = None  # current cfg as a class
        self.timestamp = 0
        self.realtime = 0
        self.ls_realtime = []
        self.ls_simtime = []
        self.cfg_timestamp = []
        self.num_report_per_config = 1  # number of report to list per config
        # self.TTT_pool = [0.004, 0.064, 0.080, 0.1, 0.128, 0.16, 0.256, 0.320, 0.48, 0.512, 0.640, 1.024, 2.56, 5.12]
        self.TTT_pool = [0.004, 0.080, 0.128, 0.256, 0.48, 0.640, 1.024, 2.56, 5.12]  # in sec
        # self.offset_pool = np.arange(-24, 24, 1)
        # self.offset_pool = list(np.arange(-10, 11, 1))
        self.offset_pool = list(np.arange(-9, 10, 2))
        self.permu_pool = list(itertools.product(self.offset_pool, self.TTT_pool))
        self.config_max = len(self.permu_pool)  # maximum number of the configuration changes
        self.config_cnt = 0
        self.target_cell = {21: 'site8[0][A]'}
        self.save_path = 'Helsinki_cfg_kpi_test.pickle'
        self.default_offset = 0
        self.default_TTT = 0.512
        self.num_states = 0 #ToDo
        self.num_actions = 0 #ToDo
        '''
        # interested config parameters
        self.ls_cfg_path_in_json = [
            '*.RRH_cells.*.transceivers.*.' + x for x in
            ['slices',
             'full_bandwidth_tx_power_dBm',
             'antenna_parameters.electrical_downtilt_degrees',
             'HO_triggers']
        ]
        self.ls_action_path_in_json = ['eMBB_slice.serving_weight',
                                       'URLLC_slice.serving_weight',
                                       'IoT_slice.serving_weight']
        '''
        self.ls_config_path = '*.cells.*.transceivers.*.HO_triggers'
        # note that 'HO_triggers' is a list of HO parameters, e.g., [{'event_type': 'A3', 'offset': 3, 'TTT': 1}]

    def add_config(self, dict_cfg):
        """ add new configurtaion to the list

        :param dict_cfg: dictionary of actual season configuration
        :return:
        """
        """get interested config parameters per site under 
        dict_cfg['config']['RRH']['site*']['RRH_cells'][*]['transceivers'][*]:
        1. ['slices']
        2. ['full_bandwidth_tx_power_dBm']
        3. ['antenna_parameters']['electrical_downtilt_degrees']
        4. ['HO_triggers']: a list with each element as a dictionary with keys 'event_type', 'offset', 'TTT'
        
        ----
        Interested network slicing parameters: general for all sites:
        - dict_cfg['config']['NetworkSlices']['eMBB_slice']['serving_weight']
        - dict_cfg['config']['NetworkSlices']['URLLC_slice']['serving_weight']
        - dict_cfg['config']['NetworkSlices']['IoT_slice']['serving_weight']
        ----
        Interested network slicing parameters: for each site:
        Todo: can be defined under ['RRH']['site*']['RRH_cells'][*]['transceivers'] the Cell Transceiver object 
        ['transceivers'][*]['weight_per_slice']
        note that this parameter overrides "serving_weight" defined in "NetworkSlices" section
        """
        cfg = Configuration(dict_cfg, self.ls_config_path.split('.'))
        self.configs.append(cfg.df_cfg_ho)
        self.cfg_timestamp.append(self.timestamp)  # this is the timestamp of the last KPI report
        print('[INFO] number of collected config: ', len(self.configs))
        self.current_cfg = cfg
        self.config_cnt += 1

    def add_report(self, dict_rpt):
        # write report dict in a sub df
        realtime = time()
        diff_realtime = realtime - self.realtime
        self.ls_realtime.append(diff_realtime)
        print('[INFO] real time interval between the reports: ', diff_realtime)
        self.realtime = realtime
        #
        diff_simtime = dict_rpt['kpi_report']['all_ue_kpis']['timestamp'] - self.timestamp  # in sec
        self.ls_simtime.append(diff_simtime)
        # print('[INFO] simulation time interval between the reports: ', diff_simtime)
        self.timestamp = dict_rpt['kpi_report']['all_ue_kpis']['timestamp']
        print('[INFO] new timestamp: ', self.timestamp)
        #
        ls_kpi_names = dict_rpt['kpi_report']['all_ue_kpis']['kpi_names']
        ls_kpi_values = dict_rpt['kpi_report']['all_ue_kpis']['kpi_values']
        sub_df = pd.DataFrame(columns=['timestamp'] + ls_kpi_names)

        num_trx = len(dict_rpt['kpi_report']['all_ue_kpis']['kpi_values'])
        for idx_trx in range(num_trx):
            ls_trx_values = [self.timestamp] + ls_kpi_values[idx_trx]
            sub_df.at[idx_trx, :] = ls_trx_values
        self.reports.append(sub_df)
        print('[INFO] number of collected report: ', len(self.reports))
        #ToDo: append the newly available data to your exp replay
        #ToDo: episodic_reward += reward

    def get_actor():
        # Initialize weights between -3e-3 and 3-e3
        last_init = tf.random_uniform_initializer(minval=-0.003, maxval=0.003)

        inputs = layers.Input(shape=(num_states,))
        out = layers.Dense(256, activation="relu")(inputs)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1, activation="tanh", kernel_initializer=last_init)(out)

        # Our upper bound is 2.0 for Pendulum.
        outputs = outputs * upper_bound #ToDo: you need to convert actions to discrete actions
                                        #ToDo: You need to decode discrete actions to meaningful actions
        model = tf.keras.Model(inputs, outputs)
        return model

    def get_critic():
        # State as input
        state_input = layers.Input(shape=(num_states))
        state_out = layers.Dense(16, activation="relu")(state_input)
        state_out = layers.Dense(32, activation="relu")(state_out)

        # Action as input
        action_input = layers.Input(shape=(num_actions))
        action_out = layers.Dense(32, activation="relu")(action_input)

        # Both are passed through seperate layer before concatenating
        concat = layers.Concatenate()([state_out, action_out])

        out = layers.Dense(256, activation="relu")(concat)
        out = layers.Dense(256, activation="relu")(out)
        outputs = layers.Dense(1)(out)

        # Outputs single value for give state-action
        model = tf.keras.Model([state_input, action_input], outputs)

        return model

    def policy(state, noise_mean, noise_var):
        sampled_actions = tf.squeeze(actor_model(state))
        noise = noise_var + noise_mean #ToDo: create a discrete r.v that is normally distributed to encourage exploration
        # Adding noise to action
        sampled_actions = sampled_actions.numpy() + noise

        # We make sure action is within bounds
        legal_action = np.clip(sampled_actions, lower_bound, upper_bound) #Todo: You need to replace lower_bound ...

        return [np.squeeze(legal_action)]

    def next_config(self, timestamp=None):
        # if only updates the traffic mask, but not the HO parameters
        '''temporary placeholder for config optimization: now just randomly assign offset and TTT'''
        # note that 'HO_triggers' is a list of dictionaries, e.g., [{'event_type': 'A3', 'offset': 3, 'TTT': 1}]
        num_trx = len(self.configs[0])
        '''
        offset = np.around(np.random.uniform(low=self.offset_range[0], high=self.offset_range[1], size=num_trx),
                           decimals=1)
        ttt = np.around(np.random.uniform(low=self.TTT_range[0], high=self.TTT_range[1], size=num_trx), decimals=1)
        '''
        #ToDo: replace the following part of the code with action selection using RL agent
        # offset = self.default_offset * np.ones(num_trx)
        # ttt = self.default_TTT * np.ones(num_trx)
        # for k, v in self.target_cell.items():
        #     offset[k] = self.permu_pool[self.config_cnt - 1][0]
        #     ttt[k] = self.permu_pool[self.config_cnt - 1][1]
        print('[INFO] next config offset = ', self.permu_pool[self.config_cnt - 1][0])
        print('[INFO] next config ttt = ', self.permu_pool[self.config_cnt - 1][1])
        ls_values = [[{'event_type': 'A3', 'offset': o, 'TTT': t}] for o, t in zip(offset, ttt)]
        # return self.current_cfg.get_cfg(values=None, timestamp=timestamp)   # update group mask
        cfg = Configuration(dict_cfg, self.ls_config_path.split('.')) #ToDo: shouldn't be read everytime
        #traffic_data = pd.read_csv('Traffic_Mask_Plot.csv')
        # cfg.agent.current_cfg.Groups ["Group1"]["parameters"]["ue_prototype"]["transceivers"]["LTE"]["expected_bit_rate"]\
        #                                 = traffic_data.values[self.config_cnt][2] #ResToDo: should be changed every 900 steps
        #                                                                           #ResTodo: change the report period in json file
        return self.current_cfg.get_cfg(values=ls_values, timestamp=timestamp)  # update parameters

    def save_results(self):
        pickle.dump((self.configs, self.cfg_timestamp, self.reports), open(self.save_path, 'wb'))


class ThreadedClient(threading.Thread):
    def __init__(self, host, port):
        threading.Thread.__init__(self)
        # set up queues
        self.receive_q = queue.Queue()
        self.send_q = queue.Queue()
        self.msgs = ''
        self.flag_run = True
        self.last_msg = 'report'
        # declare instance variables
        self.host = host
        self.port = port
        # connect to socket
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.connect((self.host, self.port))
        self.s.settimeout(.1)
        self.agent = RlAgent()
        # self.df_kpi_report = pd.DataFrame()
        self.current_config = None
        self.report_cnt = 0

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
                    self.agent.add_report(dict_msg)
                    self.report_cnt += 1
                    self.last_msg = 'report'
                elif list(dict_msg.keys())[0] == 'config':
                    self.current_config = dict_msg['config']
                    self.agent.add_config(dict_msg['config'])
                    self.last_msg = 'config'
                # should not reconfig again immediately after the last config, reconfig only after receiving reports
                # of previous new config
                if (len(self.agent.reports) % self.agent.num_report_per_config == 0) & (self.agent.config_cnt > 0) \
                        & (self.last_msg == 'report') & (self.agent.config_cnt - 1 < self.agent.config_max):
                    # - giving timestamp changes both config and group size
                    # next_config = self.agent.next_config(timestamp=self.agent.timestamp)
                    # - if giving timestamp=None, change only configs, not the group size
                    next_config = self.agent.next_config()
                    # put it in send queue
                    self.send_q.put(next_config)

                if (len(self.agent.configs) > self.agent.config_max) & (self.last_msg == 'report'):
                    self.flag_run = False
                    self.agent.save_results()

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
    TC.start()
    print('Server started, port: ', port)
    TC.start_listen()
    TC.start_run_agent()
