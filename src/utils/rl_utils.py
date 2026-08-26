import torch as th


def build_td_lambda_targets(rewards, terminated, mask, target_qs, n_agents, gamma, td_lambda):
    # 假设 B*T*A 中的 <target_qs > 和（至少）B*T-1*1 中的 <reward >、<terminated >、<mask >
    # 为未终止回合初始化最后一个 lambda-return
    ret = target_qs.new_zeros(*target_qs.shape)
    ret[:, -1] = target_qs[:, -1] * (1 - th.sum(terminated, dim=1))
    # “前向视图”的向后递归更新
    for t in range(ret.shape[1] - 2, -1,  -1):
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
                    * (rewards[:, t] + (1 - td_lambda) * gamma * target_qs[:, t + 1] * (1 - terminated[:, t]))
    # 返回从 t=0 到 t=T-1 的 lambda-return，即形状为 B*T-1*A
    return ret[:, 0:-1]

def build_gae_targets(rewards, masks, values, gamma, lambd):
    B, T, A, _ = values.size()
    T-=1
    advantages = th.zeros(B, T, A, 1).to(device=values.device)
    advantage_t = th.zeros(B, A, 1).to(device=values.device)

    for t in reversed(range(T)):
        delta = rewards[:, t] + values[:, t+1] * gamma * masks[:, t] - values[:, t]
        advantage_t = delta + advantage_t * gamma * lambd * masks[:, t]
        advantages[:, t] = advantage_t

    returns = values[:, :T] + advantages
    return advantages, returns


def build_q_lambda_targets(rewards, terminated, mask, exp_qvals, qvals, gamma, td_lambda):
    # 假设 B*T*A 中的 <target_qs > 和（至少）B*T-1*1 中的 <reward >、<terminated >、<mask >
    # 为未终止回合初始化最后一个 lambda-return
    ret = exp_qvals.new_zeros(*exp_qvals.shape)
    ret[:, -1] = exp_qvals[:, -1] * (1 - th.sum(terminated, dim=1))
    # “前向视图”的向后递归更新
    for t in range(ret.shape[1] - 2, -1,  -1):
        reward = rewards[:, t] + exp_qvals[:, t] - qvals[:, t] #离策略修正
        ret[:, t] = td_lambda * gamma * ret[:, t + 1] + mask[:, t] \
                    * (reward + (1 - td_lambda) * gamma * exp_qvals[:, t + 1] * (1 - terminated[:, t]))
    # 返回从 t=0 到 t=T-1 的 lambda-return，即形状为 B*T-1*A
    return ret[:, 0:-1]

def build_icq_target_q(td_q, target_q, mac, mask, gamma, td_lambda, n):
    aug = th.zeros_like(td_q[:, :1])
    mac = mac[:, :-1] 
    tree_q_vals = th.zeros_like(td_q)
    coeff = 1.0
    t1 = td_q[:]
    for _ in range(n):
        tree_q_vals += t1 * coeff
        t1 = th.cat(((t1 * mac)[:, 1:], aug), dim=1)
        coeff *= gamma * td_lambda
    return target_q + tree_q_vals

