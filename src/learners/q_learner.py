import copy
from modules.mixers.vdn import VDNMixer
from modules.mixers.qmix import QMixer
from modules.mixers.nmix import Mixer
import torch as th
from torch.distributions import Categorical
from torch.optim import RMSprop, Adam
from components.standarize_stream import RunningMeanStd
from utils.rl_utils import build_td_lambda_targets

class QLearner:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.n_agents = args.n_agents
        self.mac = mac
        self.logger = logger

        self.params = list(mac.parameters())

        self.mixer = None
        if args.mixer is not None:
            match args.mixer:
                case "vdn":
                    self.mixer = VDNMixer()
                case "qmix":
                    self.mixer = QMixer(args)
                case "nmix":
                    self.mixer = Mixer(args)
                case _:
                    raise ValueError("Mixer {} not recognised.".format(args.mixer))
            self.params += list(self.mixer.parameters())
            self.target_mixer = copy.deepcopy(self.mixer)

        match self.args.optim_type.lower():
            case "rmsprop":
                self.optimiser = RMSprop(params=self.params, lr=self.args.lr, alpha=self.args.optim_alpha, eps=self.args.optim_eps, weight_decay=self.args.weight_decay)
            case "adam":
                self.optimiser = Adam(params=self.params, lr=self.args.lr, weight_decay=self.args.weight_decay)
            case _:
                raise ValueError("Invalid optimiser type", self.args.optim_type)

        # 深度复制有点浪费（例如重复动作选择器），但应该适用于任何 MAC
        self.target_mac = copy.deepcopy(mac)

        self.log_stats_t = -self.args.learner_log_interval - 1
        self.training_steps = 0
        self.last_target_update_step = 0
        self.last_target_update_episode = 0

        device = "cuda" if args.use_cuda else "cpu"
        if self.args.standardise_returns:
            self.ret_ms = RunningMeanStd(shape=(self.n_agents,), device=device)
        if self.args.standardise_rewards:
            self.rew_ms = RunningMeanStd(shape=(1,), device=device)

    def train(self, batch, t_env: int, episode_num: int):
        # 获取相关张量
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        # 终止点 1 -> 0
        avail_actions = batch["avail_actions"]

        if self.args.standardise_rewards:
            self.rew_ms.update(rewards)
            rewards = (rewards - self.rew_ms.mean) / th.sqrt(self.rew_ms.var)
        
        # 计算估计的 Q 值
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # 沿时间维拼接 #(bs, T-1, n_agents, n_ac)
        
        mac_out[avail_actions == 0] = -9999999
        # 为每个智能体所采取的动作选择 Q 值
        chosen_action_na_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # 移除最后一个维度

        # 计算目标所需的 Q 值
        target_mac_out = []
        self.target_mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            target_agent_outs = self.target_mac.forward(batch, t=t)
            target_mac_out.append(target_agent_outs)

        # 计算目标值时不需要首个时间步的 Q 值估计
        target_mac_out = th.stack(target_mac_out, dim=1)  # 沿时间维拼接

        # 屏蔽不可用的动作
        target_mac_out[avail_actions == 0] = -9999999

        # 对目标 Q 值取最大值
        if self.args.double_q:
            # 选择使在线 Q 最大的动作（用于 Double Q-Learning）
            mac_out_detach = mac_out.clone().detach()
            #mac_out_detach[avail_actions == 0] = -9999999 already done before
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_na_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            cons_max_q_vals = th.gather(mac_out, 3, cur_max_actions).squeeze(3)
        else:
            target_max_na_qvals = target_mac_out.max(dim=3)[0]
            cons_max_q_vals = mac_out.max(dim=3)[0]

        chosen_action_qvals = chosen_action_na_qvals
        target_max_qvals = target_max_na_qvals

        # 混合
        if self.mixer is not None:
            chosen_action_qvals = self.mixer(chosen_action_na_qvals, batch["state"][:, :-1])
            target_max_qvals = self.target_mixer(target_max_na_qvals, batch["state"])
            cons_max_q_vals = self.mixer(cons_max_q_vals, batch["state"])

        if self.args.standardise_returns:
            target_max_qvals = target_max_qvals * th.sqrt(self.ret_ms.var) + self.ret_ms.mean

        match self.args.cal_target:
            case "td_lambda":
                targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, 
                                                            self.n_agents, self.args.gamma, self.args.td_lambda)
            case "raw":
                targets = rewards + self.args.gamma * (1 - terminated) * target_max_qvals[:, 1:]
            case _:
                raise ValueError("Unknown target calculation type")
            
        if self.args.standardise_returns:
            self.ret_ms.update(targets)
            targets = (targets - self.ret_ms.mean) / th.sqrt(self.ret_ms.var)
            
        # TD 误差
        # print(batch.max_seq_length, chosen_action_qvals.shape, targets.shape)
        td_error = (chosen_action_qvals - targets.detach())

        mask = mask.expand_as(td_error) # (bs, T, )

        # 将由填充数据产生的目标值清零
        masked_td_error = td_error * mask

        # 标准 L2 损失，对实际数据取平均值
        td_loss = (masked_td_error ** 2).sum() / mask.sum()

        loss = td_loss

        if "cql" in self.args.name:
            match self.args.cql_type:
                case "individual":
                    # CQL 误差
                    assert (th.logsumexp(mac_out[:, :-1], dim=3).shape == chosen_action_na_qvals.shape)
                    cql_error = th.logsumexp(mac_out[:, :-1], dim=3) - chosen_action_na_qvals
                    cql_mask = mask.expand_as(cql_error)
                    cql_loss = (cql_error * cql_mask).sum() / mask.sum() # 最好使用 mask.sum() 而不是 cql_mask.sum()
                case "global_raw":
                    # 性能不稳定不推荐
                    # 无法枚举所有动作，因此我们随机抽取“sample_actions_num”合法联合动作
                    sample_actions_num = self.args.raw_sample_actions
                    B, T = actions.shape[:2]

                    # 情况 1：使用 Categorical 采样
                    # filled == 0（填充位置）时 avail_actions 全为 0，因此将其填为全 1 以避免概率全部为 0
                    filled = batch["filled"][:, :-1].expand(-1, -1, self.n_agents)
                    # avail_actions = avail_actions[:, :-1].sum(-1)
                    # match_tensor = avail_actions[(filled==0)] == 0
                    # print(match_tensor.all())
                    # assert 0
                    avail_actions[:, :-1][(filled==0)] = 1  
                    
                    
                    repeat_avail_actions = avail_actions[:, :-1].unsqueeze(0).expand(sample_actions_num, -1, -1, -1, -1)
                    # repeat_avail_actions = th.repeat_interleave(avail_actions[:, :-1].unsqueeze(0), repeats=sample_actions_num, dim=0)
                    # 无法直接使用，因为 mask == 0 的时间步中 avail_actions 为 [000000]
                    total_random_actions = Categorical(repeat_avail_actions.float()).sample().long().unsqueeze(-1) 
                
                    # repeat_mac_out = th.repeat_interleave(mac_out[:,:-1].unsqueeze(0), repeats=sample_actions_num, dim=0)
                    repeat_mac_out = mac_out[:,:-1].unsqueeze(0).expand(sample_actions_num, -1, -1, -1, -1)
                    
                    random_chosen_action_qvals = th.gather(repeat_mac_out, dim=-1, index=total_random_actions).squeeze(-1) 
                    random_chosen_action_qvals = random_chosen_action_qvals.reshape(sample_actions_num*B, T, -1)
                    repeat_state = batch["state"][:, :-1].unsqueeze(0).expand(sample_actions_num, -1, -1, -1).reshape(sample_actions_num*B, T, -1)
                    
                    
                    random_chosen_action_qtotal = self.mixer(random_chosen_action_qvals, repeat_state).reshape(sample_actions_num, B, T, 1)
                    negative_sampling = th.logsumexp(random_chosen_action_qtotal, dim=0) # (B, T, 1)
                    cql_error = negative_sampling - chosen_action_qvals
                    cql_loss = (cql_error * mask).sum() / mask.sum()
                   
                    # 情况 2：沿用 CFCQL
                    # repeat_avail_actions = th.repeat_interleave(avail_actions[:, :-1].unsqueeze(0), repeats=sample_actions_num, dim=0)
                    # total_random_actions = th.randint(low=0, high=self.args.n_actions, size=(sample_actions_num, B, T, self.n_agents, 1)).to(self.args.device)#san,bs,ts,na,1
                    # chosen_if_avail = th.gather(repeat_avail_actions, dim=-1, index=total_random_actions).min(-2)[0]#san,bs,ts,1
                    # repeat_mac_out = th.repeat_interleave(mac_out[:,:-1].unsqueeze(0), repeats=sample_actions_num, dim=0)
                    # random_chosen_action_qvals = th.gather(repeat_mac_out, dim=-1, index=total_random_actions).squeeze(-1) 
                    # random_chosen_action_qvals = random_chosen_action_qvals.reshape(sample_actions_num*B, T, -1)
                    # repeat_state = batch["state"][:, :-1].unsqueeze(0).expand(sample_actions_num, -1, -1, -1).reshape(sample_actions_num*B, T, -1)
                    # random_chosen_action_qtotal = self.mixer(random_chosen_action_qvals, repeat_state).reshape(sample_actions_num, B, T, 1)
                    # negative_sampling = th.logsumexp(random_chosen_action_qtotal*chosen_if_avail, dim=0) # (B, T, 1)
                    # cql_error = negative_sampling - chosen_action_qvals
                    # cql_loss = (cql_error * mask).sum() / mask.sum()
                    
                case "global_simplified":
                    assert cons_max_q_vals[:, :-1].shape == chosen_action_qvals.shape
                    cql_error = cons_max_q_vals[:, :-1] - chosen_action_qvals
                    cql_loss = (cql_error * mask).sum() / mask.sum()
                case _:
                    raise ValueError("Unknown cql type")
            loss += self.args.cql_alpha * cql_loss
        

        # 优化
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        self.training_steps += 1
        if self.args.target_update_interval_or_tau > 1 and (episode_num - self.last_target_update_episode) / self.args.target_update_interval_or_tau >= 1.0:
            self._update_targets_hard()
            self.last_target_update_episode = episode_num
        elif self.args.target_update_interval_or_tau <= 1.0:
            self._update_targets_soft(self.args.target_update_interval_or_tau)

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss", loss.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            self.logger.log_stat("td_loss", td_loss.item(), t_env)
            if "cql" in self.args.name:
                self.logger.log_stat("cql_loss", cql_loss.item(), t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env
    
    def _update_targets_hard(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _update_targets_soft(self, tau):
        for target_param, param in zip(self.target_mac.parameters(), self.mac.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
        if self.mixer is not None:
            for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()

    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # 不太正确，但我不想保存目标网络
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
