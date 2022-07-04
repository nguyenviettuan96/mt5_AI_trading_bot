"""
Particular class of real traffic network
@author: Tianshu Chun 
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns
from envs.env import PhaseMap, PhaseSet, TrafficSimulator
# from real_net.data.build_file import gen_rou_file

import sys
import os
sys.path.append(os.path.abspath(
    "/home/smartcube/tuannguyen/deeprl_signal_control/envs/"))
sys.path.append(os.path.abspath(
    "/home/smartcube/tuannguyen/deeprl_signal_control/real_net/data"))

sns.set_color_codes()

STATE_NAMES = ['wave']
# node: (phase key, neighbor list)
NODES = {'long': ('3.0', []),
         'short': ('3.1', [])}

PHASES = {
    '3.0': ['hold', 'buy', 'sell'],
    '3.1': ['hold', 'sell', 'buy']
}


class RealNetPhase(PhaseMap):
    def __init__(self):
        self.phases = {}
        for key, val in PHASES.items():
            self.phases[key] = PhaseSet(val)


class RealNetController:
    def __init__(self, node_names, nodes):
        self.name = 'greedy'
        self.node_names = node_names
        self.nodes = nodes

    def forward(self, obs):
        actions = []
        for ob, node_name in zip(obs, self.node_names):
            actions.append(self.greedy(ob, node_name))
        return actions

    def greedy(self, ob, node_name):
        # get the action space
        phases = PHASES[NODES[node_name][0]]
        flows = []
        node = self.nodes[node_name]
        # get the green waves
        for phase in phases:
            wave = 0
            visited_ilds = set()
            for i, signal in enumerate(phase):
                if signal == 'G':
                    # find controlled lane
                    lane = node.lanes_in[i]
                    # ild = 'ild:' + lane
                    ild = lane
                    # if it has not been counted, add the wave
                    if ild not in visited_ilds:
                        j = node.ilds_in.index(ild)
                        wave += ob[j]
                        visited_ilds.add(ild)
            flows.append(wave)
        return np.argmax(np.array(flows))


class RealNetEnv(TrafficSimulator):
    def __init__(self, config, port=0, output_path='', is_record=False, record_stat=False):
        super().__init__(config, output_path, is_record, record_stat, port=port)

    def _get_node_phase_id(self, node_name):
        return self.phase_node_map[node_name]

    def _init_neighbor_map(self):
        return dict([(key, val[1]) for key, val in NODES.items()])

    def _init_map(self):
        self.neighbor_map = self._init_neighbor_map()
        self.phase_map = RealNetPhase()
        self.phase_node_map = dict([(key, val[0])
                                   for key, val in NODES.items()])
        self.state_names = STATE_NAMES

    # def _init_sim_config(self, seed):
    #     # comment out to call build_file.py
    #     return gen_rou_file(self.data_path,
    #                         self.flow_rate,
    #                         seed=seed,
    #                         thread=self.sim_thread)

    def plot_stat(self, rewards):
        self.state_stat['reward'] = rewards
        for name, data in self.state_stat.items():
            fig = plt.figure(figsize=(8, 6))
            plot_cdf(data)
            plt.ylabel(name)
            fig.savefig(self.output_path + self.name + '_' + name + '.png')


def plot_cdf(X, c='b', label=None):
    sorted_data = np.sort(X)
    yvals = np.arange(len(sorted_data))/float(len(sorted_data)-1)
    # print('sorted_data', sorted_data)
    plt.plot(sorted_data, yvals, color=c, label=label)
