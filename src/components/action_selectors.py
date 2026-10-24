import torch as th
from torch.distributions import Categorical
from .epsilon_schedules import DecayThenFlatSchedule

REGISTRY = {}

# 几个常用的动作选择器，供多智能体控制器使用
# multinomial: 根据策略概率分布选择动作 logits
# epsilon_greedy: 在贪婪和随机之间进行权衡
# soft_policies: 根据软策略选择动作

class MultinomialActionSelector():

    def __init__(self, args):
        self.args = args
        # 采用线性衰减
        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish,
                                              args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)
        self.test_greedy = getattr(args, "test_greedy", True) # 在测试阶段是否使用贪婪策略

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        # 屏蔽非法动作
        masked_policies = agent_inputs.clone()
        masked_policies[avail_actions == 0.0] = 0.0

        self.epsilon = self.schedule.eval(t_env)

        if test_mode and self.test_greedy: # 是否采用贪婪策略
            picked_actions = masked_policies.max(dim=2)[1] # 选择概率最大合法动作,dim=2为动作维度;确定性策略选择
        else:
            picked_actions = Categorical(masked_policies).sample().long() # 进行概率分布采样,保留策略选择的随机性
            if "icq" in self.args.name: # 若为ICQ算法,用专门的epsilon-random探索
                random_numbers = th.rand_like(agent_inputs[:, :, 0]) # [B,N]
                pick_random = (random_numbers < self.epsilon).long()
                random_actions = Categorical(avail_actions.float()).sample().long()
                picked_actions = pick_random * random_actions + (1 - pick_random) * picked_actions # 混合策略与平均随机策略
        if "icq" in self.args.name and \
        not (th.gather(avail_actions, dim=2, index=picked_actions.unsqueeze(2)) > 0.99).all():
            return self.select_action(agent_inputs, avail_actions, t_env, test_mode)
        return picked_actions


REGISTRY["multinomial"] = MultinomialActionSelector


class EpsilonGreedyActionSelector():

    def __init__(self, args):
        self.args = args

        self.schedule = DecayThenFlatSchedule(args.epsilon_start, args.epsilon_finish, 
                                              args.epsilon_anneal_time,
                                              decay="linear")
        self.epsilon = self.schedule.eval(0)

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):

        # 假设 agent_inputs 是每个智能体 bav 的一批 Q 值
        self.epsilon = self.schedule.eval(t_env)

        if test_mode:
            # 仅贪婪动作选择
            self.epsilon = 0.0

        # 屏蔽从选择中排除的动作
        masked_q_values = agent_inputs.clone()
        masked_q_values[avail_actions == 0.0] = -float("inf")  # 永远不应该被选择！

        random_numbers = th.rand_like(agent_inputs[:, :, 0])
        pick_random = (random_numbers < self.epsilon).long()
        random_actions = Categorical(avail_actions.float()).sample().long()

        picked_actions = pick_random * random_actions + (1 - pick_random) * masked_q_values.max(dim=2)[1]
        return picked_actions


REGISTRY["epsilon_greedy"] = EpsilonGreedyActionSelector

class SoftPoliciesSelector():

    def __init__(self, args):
        self.args = args

    def select_action(self, agent_inputs, avail_actions, t_env, test_mode=False):
        m = Categorical(agent_inputs)
        picked_actions = m.sample().long()
        return picked_actions


REGISTRY["soft_policies"] = SoftPoliciesSelector