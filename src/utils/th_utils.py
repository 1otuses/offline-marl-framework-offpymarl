from torch import nn

def clip_by_tensor(t,t_min,t_max):
    """
    按张量范围裁剪。
    :param t: 待裁剪张量
    :param t_min: 最小值
    :param t_max: 最大值
    :return: 裁剪后的张量
    """
    t=t.float()
    t_min=t_min.float()
    t_max=t_max.float()
 
    result = (t >= t_min).float() * t + (t < t_min).float() * t_min
    result = (result <= t_max).float() * result + (result > t_max).float() * t_max
    return result

def get_parameters_num(param_list):
    return str(sum(p.numel() for p in param_list) / 1000) + 'K'


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def orthogonal_init_(m, gain=1):
    if isinstance(m, nn.Linear):
        init(m, nn.init.orthogonal_,
                    lambda x: nn.init.constant_(x, 0), gain=gain)