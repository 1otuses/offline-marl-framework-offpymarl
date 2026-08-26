class MultiAgentEnv(object):

    def step(self, actions):
        """ 返回奖励、终止、信息 """
        raise NotImplementedError

    def get_obs(self):
        """ 返回列表中的所有智能体观测 """
        raise NotImplementedError

    def get_obs_agent(self, agent_id):
        """ 返回 agent_id 的观测 """
        raise NotImplementedError

    def get_obs_size(self):
        """ 返回观测的形状 """
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def get_state_size(self):
        """ 返回状态的形状 """
        raise NotImplementedError

    def get_avail_actions(self):
        raise NotImplementedError

    def get_avail_agent_actions(self, agent_id):
        """ 返回agent_id的可用动作 """
        raise NotImplementedError

    def get_total_actions(self):
        """ 返回智能体可以执行的动作总数 """
        # TODO：这仅适用于每个智能体的离散一维动作空间
        raise NotImplementedError

    def reset(self):
        """ 返回初始观测和状态 """
        raise NotImplementedError

    def render(self):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def seed(self):
        raise NotImplementedError

    def save_replay(self):
        raise NotImplementedError

    def get_env_info(self):
        env_info = {"state_shape": self.get_state_size(),
                    "obs_shape": self.get_obs_size(),
                    "n_actions": self.get_total_actions(),
                    "n_agents": self.n_agents,
                    "episode_limit": self.episode_limit}
        return env_info
