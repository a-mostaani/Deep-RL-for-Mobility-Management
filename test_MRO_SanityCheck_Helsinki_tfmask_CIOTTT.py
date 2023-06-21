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
import numpy as np
import pandas as pd
import json
import copy
import pickle
from time import time
import matplotlib.pyplot as plt
import itertools


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

        # reading traffic mask of 96 quarters
        traffic_data = pd.read_csv('Traffic_Mask_Plot.csv')
        self.group24 = list(traffic_data.values[i][2] for i in range(96))
        '''define traffic mask of 24 hours'''
        #self.group24 = [0.8, 0.65, 0.4, 0.15, 0.2, 0.3, 0.6, 0.8, 0.9, 0.8, 1.2, 1.3,
        #                 1.5, 1.8, 2.0, 1.7, 1.8, 1.5, 1.4, 1.5, 1.1, 1.0, 0.9, 0.8]

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

        #cfg = {'RRH': cfg}
        cfg = {'Sites': cfg}
        # change group sizes acc. to group24 parameter - adds 24h seasonality
        if len(self.group24) == 96 and timestamp is not None:
            new_groups = copy.deepcopy(self.Groups)
            # take time tick 0 as 0:00 a.m.
            indx = int((timestamp // 3600) % 96)
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
        self.TTT_pool = [0.004, 0.080, 0.128, 0.256,  0.48,  0.640, 1.024, 2.56, 5.12]  # in sec
        # self.offset_pool = np.arange(-24, 24, 1)
        # self.offset_pool = list(np.arange(-10, 11, 1))
        self.offset_pool = list(np.arange(-9, 10, 2))
        self.permu_pool = list(itertools.product(self.offset_pool, self.TTT_pool))
        self.config_max = len(self.permu_pool)   # maximum number of the configuration changes
        self.config_cnt = 0
        self.target_cell = {21: 'site8[0][A]'}
        self.save_path = 'Helsinki_cfg_kpi_test.pickle'
        self.default_offset = 0
        self.default_TTT = 0.512
        self.state = []
        self.reward = 0
        self.alpha = 0.5
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
        self.cfg_timestamp.append(self.timestamp)   # this is the timestamp of the last KPI report
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
        diff_simtime = dict_rpt['kpi_report']['all_cell_trx_kpis']['timestamp'] - self.timestamp  # in sec
        self.ls_simtime.append(diff_simtime)
        # print('[INFO] simulation time interval between the reports: ', diff_simtime)
        self.timestamp = dict_rpt['kpi_report']['all_cell_trx_kpis']['timestamp']
        print('[INFO] new timestamp: ', self.timestamp)
        #
        ls_kpi_names = dict_rpt['kpi_report']['all_cell_trx_kpis']['kpi_names']
        ls_kpi_values = dict_rpt['kpi_report']['all_cell_trx_kpis']['kpi_values']
        sub_df = pd.DataFrame(columns=['timestamp'] + ls_kpi_names)

        num_trx = len(dict_rpt['kpi_report']['all_cell_trx_kpis']['kpi_values'])
        for idx_trx in range(num_trx):
            ls_trx_values = [self.timestamp] + ls_kpi_values[idx_trx]
            sub_df.at[idx_trx, :] = ls_trx_values
        self.reports.append(sub_df)
        print('[INFO] number of collected report: ', len(self.reports))

        # extract state:
        # index of the downlink and uplink throughput parameter in the kpi rep
        avg_dl_ind = ls_kpi_names.index('downlink_throughput_bits_s')
        avg_ul_ind = ls_kpi_names.index('uplink_throughput_bits_s')
        self.state = list(ls_kpi_values[i][avg_dl_ind] +
                              ls_kpi_values[i][avg_ul_ind] for i in range(len(ls_kpi_values)))

        # extract reward:
        # index of the HO kpis in the kpi rep
        pp_ind = ls_kpi_names.index('pingpong_handovers_count')
        wh_ind = ls_kpi_names.index('wrong_cell_handovers_count')
        eh_ind = ls_kpi_names.index('early_handovers_count')
        lh_ind = ls_kpi_names.index('late_handovers_count')
        sum_pp = 0
        sum_wh = 0
        sum_eh = 0
        sum_lh = 0
        self.alpha = 0.5
        for i in range(len(ls_kpi_values)): sum_pp = sum_pp + ls_kpi_values[i][pp_ind]
        for i in range(len(ls_kpi_values)): sum_wh = sum_wh + ls_kpi_values[i][wh_ind]
        for i in range(len(ls_kpi_values)): sum_eh = sum_eh + ls_kpi_values[i][eh_ind]
        for i in range(len(ls_kpi_values)): sum_lh = sum_lh + ls_kpi_values[i][lh_ind]
        self.reward = - (sum_lh + sum_eh + sum_wh + self.alpha * sum_pp)

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
        offset = self.default_offset*np.ones(num_trx)
        ttt = self.default_TTT*np.ones(num_trx)
        for k, v in self.target_cell.items():
            offset[k] = self.permu_pool[self.config_cnt-1][0]
            ttt[k] = self.permu_pool[self.config_cnt-1][1]
        print('[INFO] next config offset = ', self.permu_pool[self.config_cnt-1][0])
        print('[INFO] next config ttt = ', self.permu_pool[self.config_cnt-1][1])
        ls_values = [[{'event_type': 'A3', 'offset': o, 'TTT': t}] for o, t in zip(offset, ttt)]
        # return self.current_cfg.get_cfg(values=None, timestamp=timestamp)   # update group mask
        return self.current_cfg.get_cfg(values=ls_values, timestamp=timestamp)   # update parameters

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

