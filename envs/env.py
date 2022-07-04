"""
Traffic network simulator w/ defined sumo files
@author: Tianshu Chu
"""
from envs.functions import getState, getStockDataVec, formatPrice, formatPercent
import xml.etree.cElementTree as ET
import subprocess
import pandas as pd
import numpy as np
import logging
import sys
import os
os.environ['SUMO_HOME'] = "/usr/share/sumo"
sys.path.append(os.path.join(os.environ.get("SUMO_HOME"), 'tools'))


DEFAULT_PORT = 8000
SEC_IN_MS = 1000

# hard code real-net reward norm
REALNET_REWARD_NORM = 20


class PhaseSet:
    def __init__(self, phases):
        self.num_phase = len(phases)
        self.num_lane = len(phases[0])
        self.phases = phases
        # self._init_phase_set()

    @staticmethod
    def _get_phase_lanes(phase, signal='r'):
        phase_lanes = []
        for i, l in enumerate(phase):
            if l == signal:
                phase_lanes.append(i)
        return phase_lanes

    def _init_phase_set(self):
        self.red_lanes = []
        # self.green_lanes = []
        for phase in self.phases:
            self.red_lanes.append(self._get_phase_lanes(phase))
            # self.green_lanes.append(self._get_phase_lanes(phase, signal='G'))


class PhaseMap:
    def __init__(self):
        self.phases = {}

    def get_phase(self, phase_id, action):
        # phase_type is either green or yellow
        return self.phases[phase_id].phases[int(action)]

    def get_phase_num(self, phase_id):
        return self.phases[phase_id].num_phase

    def get_lane_num(self, phase_id):
        # the lane number is link number
        return self.phases[phase_id].num_lane

    def get_red_lanes(self, phase_id, action):
        # the lane number is link number
        return self.phases[phase_id].red_lanes[int(action)]


class Node:
    def __init__(self, name, neighbor=[], control=False):
        self.control = control  # disabled
        self.lanes_in = []
        self.ilds_in = []  # for state
        self.fingerprint = []  # local policy
        self.name = name
        self.neighbor = neighbor
        self.num_state = 0  # wave and wait should have the same dim
        self.num_fingerprint = 0
        self.wave_state = []  # local state
        self.wait_state = []  # local state
        self.phase_id = -1
        self.n_a = 0
        # self.prev_action = -1


class TrafficSimulator:
    def __init__(self, config, output_path, is_record, record_stats, port=0):
        self.name = config.get('scenario')
        self.control_interval_sec = config.getint('control_interval_sec')
        self.yellow_interval_sec = config.getint('yellow_interval_sec')
        self.port = DEFAULT_PORT + port
        self.sim_thread = port
        self.obj = config.get('objective')
        self.data_path = config.get('data_path')
        self.price_data = config.get('price_data')
        self.key = config.get('key')
        self.window_size = config.getint('window_size')
        self.balance = config.getint('balance')
        self.agent = config.get('agent')
        self.coop_gamma = config.getfloat('coop_gamma')
        self.cur_episode = 0
        self.norms = {'wave': config.getfloat('norm_wave'),
                      'wait': config.getfloat('norm_wait')}
        self.clips = {'wave': config.getfloat('clip_wave'),
                      'wait': config.getfloat('clip_wait')}
        self.coef_wait = config.getfloat('coef_wait')
        self.train_mode = True
        test_seeds = config.get('test_seeds').split(',')
        test_seeds = [int(s) for s in test_seeds]

        self.data = getStockDataVec(self.price_data, self.key)
        self.l = len(self.data) - 1
        self.t = 0
        self.n = self.window_size + 1
        self.inventory = {}
        self.agent_type = {'long': 1, 'short': -1}
        self.total_profit = 0
        # self.drawdown = []
        self.balance_list = []
        self.cur_balance = self.balance

        self._init_map()
        self.init_data(is_record, record_stats, output_path)
        self.init_test_seeds(test_seeds)
        self._init_nodes()

    def _get_node_phase_id(self, node_name):
        # needs to be overwriteen
        raise NotImplementedError()

    def _get_state(self):
        # hard code the state ordering as wave, wait, fp
        state = []
        # measure the most recent state
        self._measure_state_step()

        # get the appropriate state vectors
        for node_name in self.node_names:
            # node_name 10026
            node = self.nodes[node_name]
            # wave is required in state
            cur_state = [node.wave_state]
            # include wave states of neighbors
            for nnode_name in node.neighbor:
                # discount the neigboring states
                cur_state.append(
                    self.nodes[nnode_name].wave_state * self.coop_gamma)
            # include wait state
            if 'wait' in self.state_names:
                cur_state.append(node.wait_state)
            # include fingerprints of neighbors
            for nnode_name in node.neighbor:
                cur_state.append(self.nodes[nnode_name].fingerprint)
            state.append(np.concatenate(cur_state))

        return state

    def _init_nodes(self):
        nodes = {}
        trafficlight_id = ['long', 'short']
        for node_name in trafficlight_id:
            if node_name in self.neighbor_map:
                neighbor = self.neighbor_map[node_name]
            else:
                logging.info('node %s can not be found!' % node_name)
                neighbor = []
            nodes[node_name] = Node(node_name,
                                    neighbor=neighbor,
                                    control=True)
            # Init inventory
            self.inventory[node_name] = []

        self.nodes = nodes
        self.node_names = sorted(list(nodes.keys()))
        s = 'Env: init %d node information:\n' % len(self.node_names)
        for node in self.nodes.values():
            s += node.name + ':\n'
            s += '\tneigbor: %r\n' % node.neighbor
            # s += '\tlanes_in: %r\n' % node.lanes_in
            s += '\tilds_in: %r\n' % node.ilds_in
            # s += '\tedges_in: %r\n' % node.edges_in
        logging.info(s)
        self._init_action_space()
        self._init_state_space()

    def _init_action_space(self):
        # for local and neighbor coop level
        self.n_a_ls = []
        for node_name in self.node_names:
            node = self.nodes[node_name]
            # phase_id 3.0
            phase_id = self._get_node_phase_id(node_name)
            node.phase_id = phase_id
            node.n_a = self.phase_map.get_phase_num(phase_id)
            self.n_a_ls.append(node.n_a)
        # for global coop level
        self.n_a = np.prod(np.array(self.n_a_ls))

    def _init_map(self):
        # needs to be overwriteen
        self.neighbor_map = None
        self.phase_map = None
        self.state_names = None
        raise NotImplementedError()

    def _init_policy(self):
        policy = []
        for node_name in self.node_names:
            phase_num = self.nodes[node_name].n_a
            p = 1. / phase_num
            policy.append(np.array([p] * phase_num))
        return policy

    def _init_state_space(self):
        self._reset_state()
        self.n_s_ls = []
        self.n_w_ls = []
        self.n_f_ls = []
        for node_name in self.node_names:
            node = self.nodes[node_name]
            # num_wave = node.num_state
            num_wave = self.window_size
            num_fingerprint = 0
            for nnode_name in node.neighbor:
                if self.agent not in ['a2c', 'greedy']:
                    # all marl agents have neighborhood communication
                    num_wave += self.nodes[nnode_name].num_state
                if self.agent == 'ma2c':
                    # only ma2c uses neighbor's policy
                    num_fingerprint += self.nodes[nnode_name].num_fingerprint
            num_wait = 0 if 'wait' not in self.state_names else node.num_state
            self.n_s_ls.append(num_wave + num_wait + num_fingerprint)
            self.n_f_ls.append(num_fingerprint)
            self.n_w_ls.append(num_wait)
        self.n_s = np.sum(np.array(self.n_s_ls))

    def _measure_reward_step(self, action):
        rewards = []
        for node_name, a in zip(self.node_names, list(action)):
            reward = 0
            if a == 1:  # buy
                self.inventory[node_name].append(self.data[self.t])
                # print("Buy: " + formatPrice(self.data[self.t]))

            elif a == 2 and len(self.inventory[node_name]) > 0:  # sell
                order_price = self.inventory[node_name].pop(0)
                profit = (self.data[self.t] - order_price) * \
                    self.agent_type[node_name]
                reward = max(profit, 0)
                # if node_name == 'short':
                self.total_profit += profit

                # if profit < 0:
                #     self.drawdown.append(-profit/self.cur_balance)
                self.cur_balance += profit
                self.balance_list.append(round(self.cur_balance, 2))
                # print("Sell: " + formatPrice(self.data[self.t]) + " | Profit: " + formatPrice(
                #     self.data[self.t] - bought_price))
            rewards.append(reward)

        return np.array(rewards)

    def _measure_state_step(self):
        for node_name in self.node_names:
            node = self.nodes[node_name]
            for state_name in self.state_names:
                cur_state = getState(self.data, self.t, self.n)
                if self.record_stats:
                    self.state_stat[state_name] += list(cur_state)
                # normalization
                norm_cur_state = self._norm_clip_state(cur_state,
                                                       self.norms[state_name],
                                                       self.clips[state_name])
                node.wave_state = norm_cur_state

    def _measure_traffic_step(self):
        cur_traffic = {'episode': self.cur_episode,
                       'time_sec': self.t
                       }

        self.traffic_data.append(cur_traffic)

    @staticmethod
    def _norm_clip_state(x, norm, clip=-1):
        x = x / norm
        return x if clip < 0 else np.clip(x, 0, clip)

    def _reset_state(self):
        for node_name in self.node_names:
            node = self.nodes[node_name]
            # prev action for yellow phase before each switch
            node.prev_action = 0
            # fingerprint is previous policy[:-1]
            node.num_fingerprint = node.n_a - 1
            # node.num_state = self._get_node_state_num(node)

    def _simulate(self):
        self.t += 1
        if self.is_record:
            self._measure_traffic_step()

    def _transfer_action(self, action):
        '''Transfer global action to a list of local actions'''
        phase_nums = []
        for node in self.control_node_names:
            phase_nums.append(self.nodes[node].phase_num)
        action_ls = []
        for i in range(len(phase_nums) - 1):
            action, cur_action = divmod(action, phase_nums[i])
            action_ls.append(cur_action)
        action_ls.append(action)
        return action_ls

    def _update_waits(self, action):
        for node_name, a in zip(self.node_names, action):
            red_lanes = set()
            node = self.nodes[node_name]
            for i in self.phase_map.get_red_lanes(node.phase_id, a):
                red_lanes.add(node.lanes_in[i])
            for i in range(len(node.waits)):
                lane = node.ilds_in[i]
                if lane in red_lanes:
                    node.waits[i] += self.control_interval_sec
                else:
                    node.waits[i] = 0

    def init_data(self, is_record, record_stats, output_path):
        self.is_record = is_record
        self.record_stats = record_stats
        self.output_path = output_path
        if self.is_record:
            self.traffic_data = []
            self.control_data = []
            self.trip_data = []
        if self.record_stats:
            self.state_stat = {}
            for state_name in self.state_names:
                self.state_stat[state_name] = []

    def init_test_seeds(self, test_seeds):
        self.test_num = len(test_seeds)
        self.test_seeds = test_seeds

    def output_data(self):
        if not self.is_record:
            logging.error('Env: no record to output!')
        control_data = pd.DataFrame(self.control_data)
        control_data.to_csv(self.output_path +
                            ('%s_%s_control.csv' % (self.name, self.agent)))
        traffic_data = pd.DataFrame(self.traffic_data)
        traffic_data.to_csv(self.output_path +
                            ('%s_%s_traffic.csv' % (self.name, self.agent)))
        trip_data = pd.DataFrame(self.trip_data)
        trip_data.to_csv(self.output_path + ('%s_%s_trip.csv' %
                         (self.name, self.agent)))

    def reset(self):
        self._reset_state()
        self.t = 0
        self.cur_episode += 1
        # initialize fingerprint
        self.update_fingerprint(self._init_policy())
        # next environment random condition should be different
        self.inventory = dict.fromkeys(self.inventory, [])
        self.total_profit = 0
        # self.drawdown = []
        self.balance_list = []
        self.cur_balance = self.balance
        return self._get_state()

    def step(self, action):
        state = self._get_state()
        reward = self._measure_reward_step(action)
        global_reward = np.sum(reward)  # for fair comparison
        # New t for new price
        self.t += 1
        # self._simulate()
        done = True if self.t == self.l - 1 else False

        if done:
            print("--------------------------------")
            print("Total Profit: " + formatPrice(self.total_profit))
            # print('Max Drawdown: ' + formatPercent(max(self.drawdown)))
            print('Balance List', self.balance_list)
            print('Final Balance: ' + formatPrice(self.cur_balance))
            print("--------------------------------")

        if self.is_record:
            action_r = ','.join(['%d' % a for a in action])
            cur_control = {'episode': self.cur_episode,
                           'time_sec': self.t,
                           'total_profit': formatPrice(self.total_profit),
                           'action': action_r,
                           'reward': global_reward}
            self.control_data.append(cur_control)

        # use local rewards in test
        if not self.train_mode:
            return state, reward, done, global_reward

        # discounted global reward for ma2c
        new_reward = []
        for node_name, r in zip(self.node_names, reward):
            cur_reward = r
            for nnode_name in self.nodes[node_name].neighbor:
                i = self.node_names.index(nnode_name)
                cur_reward += self.coop_gamma * reward[i]
            if self.name != 'real_net':
                new_reward.append(cur_reward)
            else:
                n_node = 1 + len(self.nodes[node_name].neighbor)
                new_reward.append(
                    cur_reward / (n_node * REALNET_REWARD_NORM))
        reward = np.array(new_reward)
        return state, reward, done, global_reward

    def update_fingerprint(self, policy):
        for node_name, pi in zip(self.node_names, policy):
            self.nodes[node_name].fingerprint = np.array(pi)[:-1]
