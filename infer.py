import MetaTrader5 as mt5
import pandas as pd
import numpy as np
import math
import argparse
import configparser
from envs.real_net_env import RealNetEnv, RealNetController
from envs.functions import getState, formatPrice
from agents.models import MA2C
from utils import Predictor


SYMBOL = "EURUSD"
DEVIATION = 20
TIMEFRAME = mt5.TIMEFRAME_H4
VOLUME = 0.03
PERIOD = 11


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config-dir', type=str, required=False,
                        default='config_ma2c_real.ini', help="inference config dir")
    parser.add_argument('--port', type=int, required=False,
                        default=0, help="running port")
    parser.add_argument('--policy-type', type=str, required=False, default='default',
                        help="inference policy type in evaluation: default, stochastic, or deterministic")
    parser.add_argument('--position-type', type=dict, required=False,
                        default={'long': 1, 'short': -1}, help="types of position")
    args = parser.parse_args()
    return args


def market_order(symbol, volume, order_type):
    tick = mt5.symbol_info_tick(symbol)

    order_dict = {'long': 0, 'short': 1}
    price_dict = {'long': tick.ask, 'short': tick.bid}

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": volume,
        "type": order_dict[order_type],
        "price": price_dict[order_type],
        "deviation": DEVIATION,
        "magic": 100,
        "comment": "python market order",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    order_result = mt5.order_send(request)
    print(order_result)

    return order_result


# function to close an order base don ticket id
def close_order(ticket):
    positions = mt5.positions_get()

    for pos in positions:
        tick = mt5.symbol_info_tick(pos.symbol)
        # 0 represents buy, 1 represents sell - inverting order_type to close the position
        type_dict = {0: 1, 1: 0}
        price_dict = {0: tick.ask, 1: tick.bid}

        if pos.ticket == ticket:
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "position": pos.ticket,
                "symbol": pos.symbol,
                "volume": pos.volume,
                "type": type_dict[pos.type],
                "price": price_dict[pos.type],
                "deviation": DEVIATION,
                "magic": 100,
                "comment": "python close order",
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }
            order_result = mt5.order_send(request)
            print(order_result)

            return order_result

    return 'Ticket does not exist'


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def _norm_clip_state(x, norm, clip=-1):
    x = x / norm
    return x if clip < 0 else np.clip(x, 0, clip)


def getState(symbol, timeframe, period, index=1000):
    bars = mt5.copy_rates_from_pos(symbol, timeframe, 1, period)
    bars_df = pd.DataFrame(bars)
    vec = bars_df.close.tolist()
    vec = [x*index for x in vec]
    res = []
    for i in range(period - 1):
        res.append(sigmoid(vec[i + 1] - vec[i]))

    return np.array(res)


def init_env(config, port=1, naive_policy=False):
    if not naive_policy:
        return RealNetEnv(config, port=port)
    else:
        env = RealNetEnv(config, port=port)
        policy = RealNetController(env.node_names, env.nodes)
        return env, policy


def data_preprocessing(cur_state, norm, clip, agents):
    # hard code the state ordering as wave, wait, fp
    state = []
    # measure the most recent state
    norm_cur_state = _norm_clip_state(cur_state, norm, clip)
    # get the appropriate state vectors
    for _ in agents:
        # wave is required in state
        cur_state = [norm_cur_state]
        state.append(np.concatenate(cur_state))

    return state


def main(args):
    config_dir = args.config_dir
    port = args.port
    policy_type = args.policy_type
    agent_type = args.position_type

    # initialize start value
    inventory = {}
    open_ticket = {}
    for agent in [*agent_type]:
        inventory[agent] = []
        open_ticket[agent] = []
    total_profit = 0
    balance_list = []
    pre_state = np.array([])

    # load config file for env
    config = configparser.ConfigParser()
    config.read(config_dir)
    cur_balance = config['ENV_CONFIG'].getint('balance')
    norm = config['ENV_CONFIG'].getfloat('norm_wave')
    clip = config['ENV_CONFIG'].getfloat('clip_wave')

    # init env
    env = init_env(config['ENV_CONFIG'], port)
    # load model for agent
    # init centralized or multi agent
    model = MA2C(env.n_s_ls, env.n_a_ls, env.n_w_ls,
                 env.n_f_ls, 0, config['MODEL_CONFIG'])
    model.load('weights/')
    model.reset()
    # collect evaluation data
    predictor = Predictor(env, model, policy_type=policy_type)
    # init mt5
    mt5.initialize()

    while True:
        cur_state = getState(symbol=SYMBOL, timeframe=TIMEFRAME,
                             period=PERIOD)
        if not np.array_equal(cur_state, pre_state):
            state = data_preprocessing(cur_state, norm, clip, [*agent_type])
            action = predictor.run(state)
            print('---ACTION--- :', action)
            tick = mt5.symbol_info_tick(SYMBOL)
            price_dict = {'long': tick.ask, 'short': tick.bid}

            for agent, a in zip([*agent_type], list(action)):
                if a == 1:
                    market_order(SYMBOL, VOLUME, agent)
                    inventory[agent].append(price_dict[agent])
                    open_ticket[agent].append(mt5.positions_get()[-1].ticket)

                elif a == 2 and len(inventory[agent]) > 0:  
                    close_order(open_ticket[agent].pop(0))
                    order_price = inventory[agent].pop(0)
                    profit = (price_dict[agent] - order_price) * \
                        agent_type[agent] * VOLUME * 100000
                    total_profit += profit
                    cur_balance += profit
                    balance_list.append(round(cur_balance, 2))

            print("--------------------------------")
            print("Total Profit: " + formatPrice(total_profit))
            print('Curent Balance: ' + formatPrice(cur_balance))
            print('Balance List', balance_list)
            print("--------------------------------")

        pre_state = cur_state


if __name__ == '__main__':
    args = parse_args()
    main(args)
