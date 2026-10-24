from modules.agents import REGISTRY as agent_REGISTRY
import torch as th
from torch.autograd import Variable
from torch.distributions import Categorical
import torch.nn.functional as F

def onehot_from_logits(logits, eps=0.0):
    """
    确定性动作离散化
    将动作logits转化为 最大值下标的one-hot 编码
    返回 one-hot 样本
    e.g. 
    logits = [1.2, 3.5, 0.8]
    output = [0, 1, 0]
    """
    argmax_acs = F.one_hot(logits.argmax(dim=-1), num_classes=logits.shape[-1]).to(logits.dtype)
    return argmax_acs #.shape == logits.shape

def sample_gumbel(shape, eps=1e-20, tens_type=th.FloatTensor):
    """ 
    生成服从Gumbel(0, 1)分布的样本噪声 
    """
    U = Variable(tens_type(*shape).uniform_(), requires_grad=False) # 均匀分布U(0,1)
    # g = -log(-log(U)) 服从Gumbel(0,1)分布
    return -th.log(-th.log(U + eps) + eps)

# 基于 https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb 修改为 PyTorch 版本
def gumbel_softmax_sample(logits, temperature):
    """ 
    从 Gumbel-Softmax 分布中抽取样本 
    当temperature -> 0 时, Gumbel-Softmax 分布趋近于 one-hot 编码的离散分布
    当 temperature -> +inf 时, Gumbel-Softmax 分布趋近于均匀分布
    """
    y = logits + sample_gumbel(logits.shape, tens_type=type(logits.data)).to(logits.device)
    return F.softmax(y / temperature, dim=-1)

# 基于 https://github.com/ericjang/gumbel-softmax/blob/master/Categorical%20VAE.ipynb 修改为 PyTorch 版本
def gumbel_softmax(logits, temperature=1.0, hard=False, return_dist=False):
    """
    实现 Straight-Through Gumbel-Softmax (STGS)
    从 Gumbel-Softmax 分布中采样,并可选择将结果离散化
    参数：
      logits: [batch_size, n_class],未归一化的对数概率
      temperature: 非负标量
      hard: 若为 True,则取 argmax,但仍相对于软样本 y 求导
    返回：
      [batch_size, n_class],从 Gumbel-Softmax 分布得到的样本。
      若 hard=True,返回 one-hot 样本;否则返回在类别维度上
      概率和为 1 的概率分布。
    """

    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        y_hard = onehot_from_logits(y)
        y_keep_gradient = (y_hard - y).detach() + y # 保持梯度
    if return_dist:
        return y_keep_gradient, y
    return y_keep_gradient


# 该多智能体控制器在智能体之间共享参数
class MADDPGMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        input_shape = self._get_input_shape(scheme)
        self._build_agents(input_shape)
        self.agent_output_type = args.agent_output_type

        self.action_selector = None

        self.hidden_states = None

    def select_actions(self, ep_batch, t_ep, t_env=0, test_mode=False):
        agent_outputs = self.forward(ep_batch, t_ep)
        if test_mode:
            chosen_actions = onehot_from_logits(agent_outputs).argmax(dim=-1)
        else:
            chosen_actions = gumbel_softmax(agent_outputs, hard=True).argmax(dim=-1)
        return chosen_actions

    def target_actions(self, ep_batch, t_ep, agent_id=None):
        agent_outputs = self.forward(ep_batch, t_ep, agent_id)
        return onehot_from_logits(agent_outputs) # (bs, n_agents, n_ac) 由于使用 detach()，无需执行 softmax/Gumbel 采样

    def forward(self, ep_batch, t, agent_id=None):
        agent_inputs = self._build_inputs(ep_batch, t, agent_id)
        avail_actions = ep_batch["avail_actions"][:, t]
        agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
        if agent_id is None:
            agent_outs = agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
        agent_outs[avail_actions == 0] = -1e10
        return agent_outs # logits

    def get_repeat_actions(self, ep_batch, t, num_repeats=None):
        """
        同一状态下重复采样动作
        """
        if num_repeats is not None:
            agent_inputs = self._build_inputs(ep_batch, t).reshape(ep_batch.batch_size, self.n_agents, -1) # [B, N, input_dim]
            input_dim = agent_inputs.shape[-1]
            agent_inputs = agent_inputs.unsqueeze(2).repeat(1, 1, num_repeats, 1).reshape(-1, input_dim) # [B*N*num_repeats, input_dim]
            avail_actions = ep_batch["avail_actions"][:, t].unsqueeze(2).repeat(1, 1, num_repeats, 1)
            agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
            agent_outs = agent_outs.view(ep_batch.batch_size, self.n_agents, num_repeats, -1) # [B, N, num_repeats, A]
        else:
            agent_inputs = self._build_inputs(ep_batch, t)
            avail_actions = ep_batch["avail_actions"][:, t]
            agent_outs, self.hidden_states = self.agent(agent_inputs, self.hidden_states)
            agent_outs = agent_outs.view(ep_batch.batch_size, self.n_agents, -1)
        agent_outs[avail_actions==0] = -1e10
        chosen_actions, dist = gumbel_softmax(agent_outs, hard=True, return_dist=True) # [B, N, num_repeats, A]
        dist = Categorical(dist)
        chosen_actions_index = chosen_actions.argmax(dim=-1)
        return chosen_actions, dist.log_prob(chosen_actions_index) # [B, N, num_repeats, A], [B, N, num_repeats]
    

    def init_hidden(self, batch_size, num_repeats=None):
        self.hidden_states = self.agent.init_hidden().unsqueeze(0).expand(batch_size, self.n_agents, -1)
        if num_repeats is not None:
            self.hidden_states = self.hidden_states.unsqueeze(2).repeat(1, 1, num_repeats, 1)
    
    def init_hidden_one_agent(self, batch_size): 
        self.hidden_states = self.agent.init_hidden().expand(batch_size, -1)  # bv

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

    def _build_inputs(self, batch, t, agent_id=None):
        """
        在环境时间步t上构建智能体的输入:
        1. 局部观测 obs_i
        2. 上一个动作 a_{t-1}——one-hot编码 (可选)
        3. 智能体ID (可选)
        x_i^t = [obs_i^t, a_{t-1}^i, onehot(i)]
        采用参数共享设计时,通常需要加入agent-id输入,以解决相同观测下agent执行相同动作的问题
        """
        bs = batch.batch_size
        inputs = []
        inputs.append(batch["obs"][:, t])  # b1av
        if self.args.obs_last_action:
            if t == 0:
                inputs.append(th.zeros_like(batch["actions_onehot"][:, t]))
            else:
                inputs.append(batch["actions_onehot"][:, t-1])
        if agent_id is None:
            if self.args.obs_agent_id:
                inputs.append(th.eye(self.n_agents, device=batch.device).unsqueeze(0).expand(bs, -1, -1))
            inputs = th.cat([x.reshape(bs*self.n_agents, -1) for x in inputs], dim=1)
        else:
            if self.args.obs_agent_id:
                auxiliary_input = th.zeros(bs, self.n_agents, device=batch.device)
                auxiliary_input[:, agent_id] = 1
                inputs.append(auxiliary_input)
            inputs = th.cat(inputs, dim=1)
        return inputs
    
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
