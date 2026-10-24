import torch.nn as nn
import torch.nn.functional as F


# RNN通过隐藏状态h保存历史记忆,输出的n_actions含义由agent_output_type决定
# 当agent_output_type="q",表示每个动作的Q值,应用在CFCQL、IQL、QMIX_CQL、QMIX、QTRAN、VDN、VDN_CQL.
# 主要是基于价值优化的算法: 通过模型学习每个动作的Q值,然后通过epsilon-greedy/argmax策略选择动作
# agent_output_type="pi_logits",表示每个动作未归一化的分数(logits)
# 主要基于策略优化的算法: 通过模型学习每个动作的logits,再通过softmax归一化表示概率,最后通过argmax选择最优动作
class RNNAgent(nn.Module):
    def __init__(self, input_shape, args):
        super(RNNAgent, self).__init__()
        self.args = args
        self.use_rnn = getattr(args, "use_rnn", True)

        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        if self.use_rnn:
            self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        else:
            self.rnn = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def init_hidden(self):
        # 在与模型相同的设备上创建隐藏状态
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        x = F.relu(self.fc1(inputs))
        if self.use_rnn:
            h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
            h = self.rnn(x, h_in) # h_in为上一时刻隐藏状态
        else:
            h = F.relu(self.rnn(x))
        q = self.fc2(h)
        return q, h
