import numpy as np

# 所有实体的物理/外部基础状态
class EntityState(object):
    def __init__(self):
        # 物理位置
        self.p_pos = None
        # 物理速度
        self.p_vel = None

# 智能体状态（包括通信和内部/心理状态）
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # 通信量
        self.c = None

# 智能体的动作
class Action(object):
    def __init__(self):
        # 物理动作
        self.u = None
        # 通信动作
        self.c = None

# 物理世界实体的属性和状态
class Entity(object):
    def __init__(self):
        # 名称
        self.name = ''
        # 属性：
        self.size = 0.050
        # 实体可以移动/被推动
        self.movable = False
        # 实体与其他实体发生碰撞
        self.collide = True
        # 材料密度（影响质量）
        self.density = 25.0
        # 颜色
        self.color = None
        # 最大速度和加速度
        self.max_speed = None
        self.accel = None
        # 状态
        self.state = EntityState()
        # 质量
        self.initial_mass = 1.0

    @property
    def mass(self):
        return self.initial_mass

# 地标实体的属性
class Landmark(Entity):
     def __init__(self):
        super(Landmark, self).__init__()

# 智能体实体的属性
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # 默认情况下智能体是可移动的
        self.movable = True
        # 无法发送通信信号
        self.silent = False
        # 无法观测世界
        self.blind = False
        # 物理执行器噪声幅度
        self.u_noise = None
        # 通信噪声幅度
        self.c_noise = None
        # 控制范围
        self.u_range = 1.0
        # 状态
        self.state = AgentState()
        # 动作
        self.action = Action()
        # 要执行的脚本行为
        self.action_callback = None

# 多智能体世界
class World(object):
    def __init__(self):
        # 智能体和实体列表（可以在执行时更改！）
        self.agents = []
        self.landmarks = []
        # 通信通道维度
        self.dim_c = 0
        # 位置维度
        self.dim_p = 2
        # 颜色维度
        self.dim_color = 3
        # 模拟时间步长
        self.dt = 0.1
        # 物理阻尼
        self.damping = 0.25
        # 接触响应参数
        self.contact_force = 1e+2
        self.contact_margin = 1e-3

        self.np_random = np.random.RandomState()

    def seed(self, seed):
        self.np_random = np.random.RandomState(seed)

    # 返回世界中的所有实体
    @property
    def entities(self):
        return self.agents + self.landmarks

    # 返回所有可由外部策略控制的智能体
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # 返回所有由世界脚本控制的智能体
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # 更新世界状态
    def step(self):
        # 为脚本化智能体设置动作
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # 汇总施加到实体上的力
        p_force = [None] * len(self.entities)
        # 应用智能体物理控制
        p_force = self.apply_action_force(p_force)
        # 施加环境力
        p_force = self.apply_environment_force(p_force)
        # 整合物理状态
        self.integrate_state(p_force)
        # 更新智能体状态
        for agent in self.agents:
            self.update_agent_state(agent)

    # 汇总智能体的动作力
    def apply_action_force(self, p_force):
        # 设置施加的力
        for i,agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = agent.action.u + noise                
        return p_force

    # 汇总作用于实体的物理力
    def apply_environment_force(self, p_force):
        # 简单（但低效）的碰撞响应
        for a,entity_a in enumerate(self.entities):
            for b,entity_b in enumerate(self.entities):
                if(b <= a): continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if(f_a is not None):
                    if(p_force[a] is None): p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a] 
                if(f_b is not None):
                    if(p_force[b] is None): p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]        
        return p_force

    # 整合物理状态
    def integrate_state(self, p_force):
        for i,entity in enumerate(self.entities):
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if (p_force[i] is not None):
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                                  np.square(entity.state.p_vel[1])) * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt

    def update_agent_state(self, agent):
        # 设置通信状态（目前直接设置）
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise      

    # 获取两个实体之间任何接触的碰撞力
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None] # 不参与碰撞
        if (entity_a is entity_b):
            return [None, None] # 不与自身发生碰撞
        # 计算实体之间的实际距离
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # 最小允许距离
        dist_min = entity_a.size + entity_b.size
        # Softmax 平滑穿透量
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        if dist == 0:
            print(entity_a, entity_b)
            print(entity_a.state.p_pos, entity_b.state.p_pos)
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]
