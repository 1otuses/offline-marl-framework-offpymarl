import gym
from gym import spaces
from gym.envs.registration import EnvSpec
import numpy as np
from mpe.multi_discrete import MultiDiscrete
import copy

# 多智能体世界中所有智能体的环境
# 当前代码假设在运行时不会创建/销毁任何智能体！
class MultiAgentEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, world, reset_callback=None, reward_callback=None,
                 observation_callback=None, info_callback=None,
                 done_callback=None, shared_viewer=True):
        world = copy.deepcopy(world)
        self.world = world
        self.agents = self.world.policy_agents
        # 设置所需的向量化 Gym 环境属性
        self.n = len(world.policy_agents)
        # 场景回调
        self.reset_callback = reset_callback
        self.reward_callback = reward_callback
        self.observation_callback = observation_callback
        self.info_callback = info_callback
        self.done_callback = done_callback
        # 环境参数
        self.discrete_action_space = True
        # 若为 true，动作为 0...N 的整数；否则为 N 维 one-hot 向量
        self.discrete_action_input = False
        # 若为 true，即使动作是连续值，也会以离散方式执行
        self.force_discrete_action = world.discrete_action if hasattr(world, 'discrete_action') else False
        # 若为 true，所有智能体共享相同奖励
        self.shared_reward = world.collaborative if hasattr(world, 'collaborative') else False
        self.time = 0

        # 配置各类空间
        self.action_space = []
        self.observation_space = []
        for agent in self.agents:
            total_action_space = []
            # 物理动作空间
            if self.discrete_action_space:
                u_action_space = spaces.Discrete(world.dim_p * 2 + 1)
            else:
                u_action_space = spaces.Box(low=-agent.u_range, high=+agent.u_range, shape=(world.dim_p,), dtype=np.float32)
            if agent.movable:
                total_action_space.append(u_action_space)
            # 通信动作空间
            if self.discrete_action_space:
                c_action_space = spaces.Discrete(world.dim_c)
            else:
                c_action_space = spaces.Box(low=0.0, high=1.0, shape=(world.dim_c,), dtype=np.float32)
            if not agent.silent:
                total_action_space.append(c_action_space)
            # 总动作空间
            if len(total_action_space) > 1:
                # 所有动作空间都是离散的，因此简化为多离散动作空间
                if all([isinstance(act_space, spaces.Discrete) for act_space in total_action_space]):
                    act_space = MultiDiscrete([[0, act_space.n - 1] for act_space in total_action_space])
                else:
                    act_space = spaces.Tuple(total_action_space)
                self.action_space.append(act_space)
            else:
                self.action_space.append(total_action_space[0])
            # 观测空间
            obs_dim = len(observation_callback(agent, self.world))
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(obs_dim,), dtype=np.float32))
            agent.action.c = np.zeros(self.world.dim_c)

        self.action_space = spaces.Tuple(tuple(self.action_space))
        self.observation_space = spaces.Tuple(tuple(self.observation_space))
        self.n_agents = self.n

        # 渲染
        self.shared_viewer = shared_viewer
        if self.shared_viewer:
            self.viewers = [None]
        else:
            self.viewers = [None] * self.n
        self._reset_render()

    def seed(self, seed):
        self.world.seed(seed)

    def step(self, action_n):

        one_hot_actions = []
        for act, acsp in zip(action_n, self.action_space):
            one_hot = np.zeros(acsp.n)
            one_hot[act] = 1.0
            one_hot_actions.append(one_hot)
        action_n = one_hot_actions

        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        self.agents = self.world.policy_agents
        # 为每个智能体设置动作
        for i, agent in enumerate(self.agents):
            self._set_action(action_n[i], agent, self.action_space[i])
        # 推进世界状态
        self.world.step()
        # 记录每个智能体的观测
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
            reward_n.append(self._get_reward(agent))
            done_n.append(self._get_done(agent))

            info_n['n'].append(self._get_info(agent))

        return tuple(obs_n), reward_n, done_n, info_n

    def reset(self):

        # 重置世界
        self.reset_callback(self.world)
        # 重置渲染器
        self._reset_render()
        # 记录每个智能体的观测
        obs_n = []
        self.agents = self.world.policy_agents
        for agent in self.agents:
            obs_n.append(self._get_obs(agent))
        return tuple(obs_n)

    # 获取用于基准测试的信息
    def _get_info(self, agent):
        if self.info_callback is None:
            return {}
        return self.info_callback(agent, self.world)

    # 获取特定智能体的观测
    def _get_obs(self, agent):
        if self.observation_callback is None:
            return np.zeros(0)
        return self.observation_callback(agent, self.world).astype(np.float32)

    # 获取特定智能体的终止标记
    # 目前未使用——允许智能体超出可视区域
    def _get_done(self, agent):
        if self.done_callback is None:
            return False
        return self.done_callback(agent, self.world)
    
    def get_state(self):
        # 用于 pp
        # 获取该智能体参考系中所有实体的位置
        entity_pos = []
        for entity in self.world.landmarks:
            if not entity.boundary:
                entity_pos.append(entity.state.p_pos)
        prey_pos_vel = []
        predator_pos_vel = []
        for agent in self.world.agents:
            if agent.adversary:
                predator_pos_vel.append(agent.state.p_pos)
                predator_pos_vel.append(agent.state.p_vel)
            else:
                prey_pos_vel.append(agent.state.p_pos)
                prey_pos_vel.append(agent.state.p_vel)
        return np.concatenate(entity_pos + predator_pos_vel + prey_pos_vel)
        
    def get_state_size(self):
        return 20 # 用于 pp

    # 获取特定智能体的奖励
    def _get_reward(self, agent):
        if self.reward_callback is None:
            return 0.0
        return self.reward_callback(agent, self.world)

    # 为特定智能体设置 env 动作
    def _set_action(self, action, agent, action_space, time=None):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)
        # 处理动作
        if isinstance(action_space, MultiDiscrete):
            act = []
            size = action_space.high - action_space.low + 1
            index = 0
            for s in size:
                act.append(action[index:(index+s)])
                index += s
            action = act
        else:
            action = [action]

        if agent.movable:
            # 物理动作
            if self.discrete_action_input:
                agent.action.u = np.zeros(self.world.dim_p)
                # 处理离散动作
                if action[0] == 1: agent.action.u[0] = -1.0
                if action[0] == 2: agent.action.u[0] = +1.0
                if action[0] == 3: agent.action.u[1] = -1.0
                if action[0] == 4: agent.action.u[1] = +1.0
            else:
                if self.force_discrete_action:
                    d = np.argmax(action[0])
                    action[0][:] = 0.0
                    action[0][d] = 1.0
                if self.discrete_action_space:
                    agent.action.u[0] += action[0][1] - action[0][2]
                    agent.action.u[1] += action[0][3] - action[0][4]
                else:
                    agent.action.u = action[0]
            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # 通信动作
            if self.discrete_action_input:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            else:
                agent.action.c = action[0]
            action = action[1:]
        # 确保我们使用了所有动作元素
        assert len(action) == 0

    # 重置渲染资源
    def _reset_render(self):
        self.render_geoms = None
        self.render_geoms_xform = None

    # 渲染环境
    def render(self, mode='human'):
        if mode == 'human':
            alphabet = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
            message = ''
            for agent in self.world.agents:
                comm = []
                for other in self.world.agents:
                    if other is agent: continue
                    if np.all(other.state.c == 0):
                        word = '_'
                    else:
                        word = alphabet[np.argmax(other.state.c)]
                    message += (other.name + ' to ' + agent.name + ': ' + word + '   ')
            # print(message)

        for i in range(len(self.viewers)):
            # 创建查看器（如有必要）
            if self.viewers[i] is None:
                # 仅在需要时导入 rendering（无显示设备的机器不导入）
                #from gym.envs.classic_control import rendering
                from mpe import rendering
                self.viewers[i] = rendering.Viewer(700,700)

        # 创建渲染几何体
        if self.render_geoms is None:
            # 仅在需要时导入 rendering（无显示设备的机器不导入）
            #from gym.envs.classic_control import rendering
            from mpe import rendering
            self.render_geoms = []
            self.render_geoms_xform = []
            for entity in self.world.entities:
                geom = rendering.make_circle(entity.size)
                xform = rendering.Transform()
                if 'agent' in entity.name:
                    geom.set_color(*entity.color, alpha=0.5)
                else:
                    geom.set_color(*entity.color)
                geom.add_attr(xform)
                self.render_geoms.append(geom)
                self.render_geoms_xform.append(xform)

            # 将几何对象添加到查看器
            for viewer in self.viewers:
                viewer.geoms = []
                for geom in self.render_geoms:
                    viewer.add_geom(geom)

        results = []
        for i in range(len(self.viewers)):
            from mpe import rendering
            # 更新边界以智能体为中心
            cam_range = 1
            if self.shared_viewer:
                pos = np.zeros(self.world.dim_p)
            else:
                pos = self.agents[i].state.p_pos
            self.viewers[i].set_bounds(pos[0]-cam_range,pos[0]+cam_range,pos[1]-cam_range,pos[1]+cam_range)
            # 更新几何位置
            for e, entity in enumerate(self.world.entities):
                self.render_geoms_xform[e].set_translation(*entity.state.p_pos)
            # 渲染到显示器或数组
            results.append(self.viewers[i].render(return_rgb_array = mode=='rgb_array'))

        if self.shared_viewer:
            assert len(results) == 1
            return results[0]

        return results

    # 在局部坐标系中创建受体场位置
    def _make_receptor_locations(self, agent):
        receptor_type = 'polar'
        range_min = 0.05 * 2.0
        range_max = 1.00
        dx = []
        # 圆形感受野
        if receptor_type == 'polar':
            for angle in np.linspace(-np.pi, +np.pi, 8, endpoint=False):
                for distance in np.linspace(range_min, range_max, 3):
                    dx.append(distance * np.array([np.cos(angle), np.sin(angle)]))
            # 添加原点
            dx.append(np.array([0.0, 0.0]))
        # 网格感受野
        if receptor_type == 'grid':
            for x in np.linspace(-range_max, +range_max, 5):
                for y in np.linspace(-range_max, +range_max, 5):
                    dx.append(np.array([x,y]))
        return dx

    def close(self):
        for viewer in self.viewers:
            if viewer:
                viewer.close()


# 用于一批多智能体环境的向量化封装器
# 假设所有环境都有相同的观测和动作空间
class BatchMultiAgentEnv(gym.Env):
    metadata = {
        'runtime.vectorized': True,
        'render.modes' : ['human', 'rgb_array']
    }

    def __init__(self, env_batch):
        self.env_batch = env_batch

    @property
    def n(self):
        return np.sum([env.n for env in self.env_batch])

    @property
    def action_space(self):
        return self.env_batch[0].action_space

    @property
    def observation_space(self):
        return self.env_batch[0].observation_space

    def step(self, action_n, time):
        obs_n = []
        reward_n = []
        done_n = []
        info_n = {'n': []}
        i = 0
        for env in self.env_batch:
            obs, reward, done, _ = env.step(action_n[i:(i+env.n)], time)
            i += env.n
            obs_n += obs
            # reward = [r / len(self.env_batch) for r in reward]
            reward_n += reward
            done_n += done
        return obs_n, reward_n, done_n, info_n

    def reset(self):
        obs_n = []
        for env in self.env_batch:
            obs_n += env.reset()
        return obs_n

    # 渲染环境
    def render(self, mode='human', close=True):
        results_n = []
        for env in self.env_batch:
            results_n += env.render(mode, close)
        return results_n
