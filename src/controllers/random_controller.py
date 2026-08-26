from torch.distributions import Categorical


# 该多智能体控制器在智能体之间共享参数
class RandomMAC:
    def __init__(self, scheme, groups, args):
        self.n_agents = args.n_agents
        self.args = args
        self.agent_output_type = args.agent_output_type

    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        # 仅对 bs 中选定的批次元素选择动作
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        chosen_actions = Categorical(probs=avail_actions.float()).sample().long() 
        # [0, 1, 1, ...], in Categorical, do "probs/probs.sum(-1, keepdim=True) "
        return chosen_actions

    def forward(self, ep_batch, t, test_mode=False):
        pass

    def init_hidden(self, batch_size):
        pass

    def parameters(self):
        pass

    def load_state(self, other_mac):
        pass

    def cuda(self):
        pass

    def save_models(self, path):
        pass

    def load_models(self, path):
        pass

    def _build_agents(self, input_shape):
        pass

    def _build_inputs(self, batch, t):
        pass

    def _get_input_shape(self, scheme):
        pass
