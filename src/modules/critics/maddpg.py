import torch as th
import torch.nn as nn
import torch.nn.functional as F


# 确定性策略,Crtic:学一个具体动作的标量Q(s,a)
class MADDPGCritic(nn.Module):
    def __init__(self, scheme, args):
        super(MADDPGCritic, self).__init__()
        self.args = args
        self.n_actions = args.n_actions
        self.n_agents = args.n_agents
        self.input_shape = self._get_input_shape(scheme)
        self.output_type = "q"

        self.fc1 = nn.Linear(self.input_shape, args.hidden_dim)
        self.fc2 = nn.Linear(args.hidden_dim, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, 1)
    
    def forward(self, inputs, actions):
        inputs = th.cat((inputs, actions), dim=-1) # [B, T, N, input_shape + n_actions]
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        q = self.fc3(x)
        return q
        
    def _get_input_shape(self, scheme):
        # 状态
        input_shape = scheme["state"]["vshape"]
        # print(scheme["state"]["vshape"], scheme["obs"]["vshape"], self.n_agents, scheme["actions_one"])
        # 是否添加个体观测
        if self.args.critic_individual_obs:
            input_shape += scheme["obs"]["vshape"]
        
        if self.args.critic_last_action:
            input_shape += self.n_actions
        # 智能体 ID
        if self.args.critic_agent_id:
            input_shape += self.n_agents
        
        return input_shape + self.n_actions * self.n_agents