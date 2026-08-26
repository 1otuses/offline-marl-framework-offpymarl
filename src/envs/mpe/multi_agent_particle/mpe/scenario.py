import numpy as np

# 定义世界构建的场景
class BaseScenario(object):
    # 创建世界中的元素
    def make_world(self):
        raise NotImplementedError()
    # 创建世界的初始条件
    def reset_world(self, world):
        raise NotImplementedError()
