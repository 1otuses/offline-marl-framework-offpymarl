from .coma import COMACritic
from .maddpg import MADDPGCritic
from .icq import ICQCritic
from .itd3 import ITD3Critic
from .double import DoubleCritic

REGISTRY = {}

# 目前learner可能用到的Critic组件
# MADDPG、ITD3确定性策略,Critic需要适配
# Double Critic方法
# ICQ Q(s,a)=A(s,a)+V(s)框架
REGISTRY["coma_critic"] = COMACritic
REGISTRY["maddpg_critic"] = MADDPGCritic
REGISTRY["icq_critic"] = ICQCritic
REGISTRY["itd3_critic"] = ITD3Critic
REGISTRY["double_critic"] = DoubleCritic
