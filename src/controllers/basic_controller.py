from modules.agents import REGISTRY as agent_REGISTRY
from components.action_selectors import REGISTRY as action_REGISTRY
import torch as th


# 该多智能体控制器在智能体之间共享参数
# 设计的功能是如何组织multi-agent结构,
class BasicMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape) # 创建共享参数的Agent网络——默认为rnn_agent
        self.agent_output_type = args.agent_output_type

        self.action_selector = action_REGISTRY[args.action_selector](args)

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # 仅对 bs 中选定的批次元素选择动作
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(
            agent_outputs[bs], 
            avail_actions[bs], 
            t_env, 
            test_mode=test_mode
            )
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        agent_inputs = self._build_inputs(ep_batch, t)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        # agent_outs: [B*N, A]
        # Softmax 智能体输出(如果它们是策略 logits)
        if self.agent_output_type == "pi_logits":

            if getattr(self.args, "mask_before_softmax", True):
                # 使不可用动作的 logits 设置为极低,以尽量减少它们对 softmax 的影响
                reshaped_avail_actions = avail_actions.reshape(ep_batch.batch_size * self.n_agents, -1)
                agent_outs[reshaped_avail_actions == 0] = -1e10

            agent_outs = th.nn.functional.softmax(agent_outs, dim=-1)
            # softmax 归一化处理
            if not test_mode:
                # epsilon 下限
                epsilon_action_num = agent_outs.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    # 以 epsilon 概率从可用动作中均匀随机选择
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float().clamp_min(1)

                agent_outs = ((1 - self.action_selector.epsilon) * agent_outs
                               + th.ones_like(agent_outs) * self.action_selector.epsilon/epsilon_action_num)
                
                if getattr(self.args, "mask_before_softmax", True):
                    # 最后再确保非法动作仍为0
                    agent_outs[reshaped_avail_actions == 0] = 0.0

        return agent_outs.view(ep_batch.batch_size, self.n_agents, -1) # 恢复[B*N, A] -> [B, N, A]
    def init_hidden(self, batch_size):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)  # bav

    def parameters(self):
        return self.agent.parameters()

    def load_state(self, other_mac):
        self.agent.load_state_dict(other_mac.agent.state_dict())

    def cuda(self):
        self.agent.cuda()

    def save_models(self, path):
        th.save(self.agent.state_dict(), "{}/agent.th".format(path))

    def load_models(self, path):
        self.agent.load_state_dict(th.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage))

    def _build_agents(self, input_shape):
        self.agent = agent_REGISTRY[self.args.agent](input_shape, self.args)

    def _build_inputs(self, batch, t):
        """
        在环境时间步t上构建智能体的输入:
        1. 局部观测 obs_i
        2. 上一个动作 a_{t-1}——one-hot编码 (可选)
        3. 智能体ID (可选)
        x_i^t = [obs_i^t, a_{t-1}^i, onehot(i)]
        采用参数共享设计时,通常需要加入agent-id输入,以解决相同观测下agent执行相同动作的问题
        """
        # 假设同质智能体具有扁平观测
        bs = batch.batch_size # B
        inputs = []
        inputs.append(batch["obs"][:, t])  # [B, N, O_i]
        if self.args.obs_last_action:
            if t == 0: # 在时间步t=0时,没有上一个动作,所以用全零向量代替
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1]) # a_{t-1}执行的动作 one-hot编码
        if self.args.obs_agent_id:
            inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
        inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        return inputs # [B*N, input_shape=O + A + N]

    def _get_input_shape(self, scheme):
        """
        计算agent_i的输入维度:
        局部观测 obs_i
        若obs_last_action=True, 则输入包含上一个动作 a_{t-1} 的 one-hot 编码
        若obs_agent_id=True, 则输入包含智能体的ID one-hot编码
        """
        input_shape = scheme["obs"]["vshape"]
        if self.args.obs_last_action:
            input_shape += scheme["actions_onehot"]["vshape"][0]
        if self.args.obs_agent_id:
            input_shape += self.n_agents
        return input_shape
