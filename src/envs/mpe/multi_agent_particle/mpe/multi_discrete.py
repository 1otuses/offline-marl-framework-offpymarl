# OpenAI Gym 中 multi_discrete.py 的旧版本（避免受 Gym 更新影响）
# (https://github.com/openai/gym/blob/1fb81d4e3fb780ccf77fec731287ba07da35eb84/gym/spaces/multi_discrete.py)

import numpy as np

import gym

class MultiDiscrete(gym.Space):
    """
    - 多离散动作空间由一系列参数不同的离散动作空间组成。
    - 它既适用于 Discrete 动作空间，也适用于连续的 Box 动作空间。
    - 它可用于表示游戏控制器或键盘，其中每个按键都可表示为离散动作空间。
    - 初始化参数为数组的数组，每个离散动作空间由 [min, max] 描述，
      可取从 `min` 到 `max`（包含两端）的任意整数。
    注意：值 0 必须始终表示 NOOP 动作。
    以任天堂游戏控制器为例，可将其视为 3 个离散动作空间：
        1) 方向键：Discrete 5 - NOOP[0], UP[1], RIGHT[2], DOWN[3], LEFT[4] - 参数：min: 0, max: 4
        2) A 键：Discrete 2 - NOOP[0], Pressed[1] - 参数：min: 0, max: 1
        3) B 键：Discrete 2 - NOOP[0], Pressed[1] - 参数：min: 0, max: 1
    初始化示例：
        MultiDiscrete([ [0,4], [0,1], [0,1] ])
    """
    def __init__(self, array_of_param_array):
        self.low = np.array([x[0] for x in array_of_param_array])
        self.high = np.array([x[1] for x in array_of_param_array])
        self.num_discrete_space = self.low.shape[0]

    def sample(self):
        """ 返回一个数组，其中包含来自每个离散动作空间的一个样本 """
        # 对每一行执行：round(random .* (max - min) + min, 0)
        random_array = np.random.RandomState().rand(self.num_discrete_space)
        return [int(x) for x in np.floor(np.multiply((self.high - self.low + 1.), random_array) + self.low)]
    def contains(self, x):
        return len(x) == self.num_discrete_space and (np.array(x) >= self.low).all() and (np.array(x) <= self.high).all()

    @property
    def shape(self):
        return self.num_discrete_space
    def __repr__(self):
        return "MultiDiscrete" + str(self.num_discrete_space)
    def __eq__(self, other):
        return np.array_equal(self.low, other.low) and np.array_equal(self.high, other.high)