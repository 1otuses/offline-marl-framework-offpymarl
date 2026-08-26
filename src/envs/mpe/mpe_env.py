import numpy as np
import random
import gym
from gym.envs.registration import register
from gym.spaces.discrete import Discrete
from sys import stderr
import matplotlib.pyplot as plt
import os
import datetime

from envs.multiagentenv import MultiAgentEnv as PyMARLMultiAgentEnv
from pretrained import REGISTRY as wrapper_REGISTRY
from mpe.environment import MultiAgentEnv as MPEMultiAgentEnv
import mpe.scenarios as scenarios


class MPEWrapperEnv(PyMARLMultiAgentEnv):
    """
    仅支持部分 MPE 场景。
    """
    def __init__(self,
                seed: int,
                scenario_name: str='configurable_tag',
                time_limit: int=50,
                num_good_agents: int=1,
                num_adversaries: int=3,
                num_landmarks: int=2,
                wrapper: str=None,
                key: str=None,
                pretrained_wrapper: str=None):
        # 构建 MPE 环境
        scenario = scenarios.load(scenario_name + ".py").Scenario()
        world = scenario.make_world(num_good_agents=num_good_agents, num_adversaries=num_adversaries, num_landmarks=num_landmarks)
        self._env = MPEMultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
        # 包裹 MPE 环境
        if wrapper is not None:
            self._env = wrapper_REGISTRY[wrapper](self._env, num_good_agents=num_good_agents, num_adversaries=num_adversaries, num_landmarks=num_landmarks)

        # 获取 nof_agents 和 nof_actions
        self.n_agents = self._env.n_agents
        self.n_actions = self.get_total_actions()
        # 设置随机种子
        np.random.seed(seed)
        random.seed(seed)
        # 设置一些属性
        self._episode_steps = 0
        self.episode_limit = time_limit

    def step(self, actions):
        if actions.__class__ == np.ndarray:
            self.obs, reward_n, done_n, info_n = self._env.step(actions)
        else:
            self.obs, reward_n, done_n, info_n = self._env.step(actions.cpu().numpy())
        # 获取封装后的奖励和终止标记
        self.obs, reward, terminated = tuple(self.obs), reward_n[0], np.all(done_n)
        # 设置 episode_steps
        self._episode_steps += 1
        if self._episode_steps >= self.episode_limit:
            # 保证回合步数在限制范围内
            terminated = True
        return reward, terminated, {}

    def get_obs(self):
        """ 返回元组中的所有智能体观测 """
        return self.obs
    
    def get_obs_agent(self, agent_id):
        """ 返回 agent_id 的观测 """
        return np.array(self.obs[agent_id])

    def get_obs_size(self):
        """ 返回观测的形状 """
        obs_shapes = [np.prod(item.shape).item() for item in self._env.observation_space]
        all_equal = len(set(obs_shapes)) == 1
        if not all_equal:
            raise Exception("Agents' observation shapes are not all equal in chosen scenario")
        return obs_shapes[0]

    def get_state(self):
        """ 将所有智能体观测拼接为状态 """
        state = self.obs[0]
        for i in range(self.n_agents - 1):
            state = np.concatenate([state, self.obs[i + 1]])
        return state

    def get_state_size(self):
        """ 返回状态形状 """
        return self.get_obs_size() * self.n_agents
    
    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]
    
    def get_avail_agent_actions(self, agent_id):
        return np.ones(self.n_actions)
    
    def get_total_actions(self):
        # assert valid action_space
        all_discrete = np.all([isinstance(item, Discrete) for item in self._env.action_space])
        if not all_discrete:
            raise Exception("Not supported scenario for pymarl")
        action_shapes = [item.n for item in self._env.action_space]
        all_equal = len(set(action_shapes)) == 1
        if not all_equal:
            raise Exception("Agents' action shapes are not all equal in chosen scenario")
        # 返回动作形状
        return action_shapes[0]

    def reset(self):
        self._episode_steps = 0
        self.obs = tuple(self._env.reset())
        return self.obs, self.get_state()
    
    def render(self):
        self._env.render()
    
    def close(self):
        pass
    
    def seed(self):
        pass
    
    def save_replay(self):
        pass

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info

    def get_stats(self):
        stats = {
        }
        return stats
        
    def get_dynamic_env_info(self):
        """返回可帮助基于规则的智能体进行决策的信息。"""
        dynamic_env_info = {
            "state": self.get_state(),
        }
        return dynamic_env_info


if __name__ == "__main__":
    mpe_env = MPEWrapperEnv(scenario_name="configurable_tag", seed=7, time_limit=10, wrapper="frozen_tag")
    env_info = mpe_env.get_env_info()
    print(env_info)
