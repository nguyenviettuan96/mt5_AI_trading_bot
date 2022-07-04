from inspect import indentsize
import numpy as np
import math

# prints formatted price


def formatPrice(n):
    return ("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n))


def formatPercent(n):
    return "{0:.2f}".format(n*100) + "%"

# returns the vector containing stock data from a fixed file


def getStockDataVec(data_dir, key, index=1000):
    vec = []
    lines = open(data_dir + key + ".csv", "r").read().splitlines()

    for line in lines[1:]:
        l1 = line.split(",")[4]  # [1:7] --> not use by no have " "
        # vec.append(float(line.split(",")[1])*volume)
        vec.append(float(l1)*index)
        # list1 = [float(x) for x in l1]
        # print('num', float(l1)*volume)

    return vec

# returns the sigmoid


def sigmoid(x):
    return 1 / (1 + math.exp(-x))

# returns an an n-day state representation ending at time t


def getState(data, t, n):
    d = t - n + 1
    block = data[d:t + 1] if d >= 0 else -d * \
        [data[0]] + data[0:t + 1]  # pad with t0
    res = []
    for i in range(n - 1):
        res.append(sigmoid(block[i + 1] - block[i]))

    return np.array(res)  # [res]
