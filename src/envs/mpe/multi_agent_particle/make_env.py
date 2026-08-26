"""
使用 ./scenarios/ 中列出的场景创建多智能体环境。
调用示例：
    env = make_env('simple_speaker_listener')
生成 env 对象后，其用法与 OpenAI Gym 环境类似。

使用此环境的策略必须以列表形式输出所有智能体的动作。列表中的每个元素
都应为大小为 (env.world.dim_p + env.world.dim_c, 1) 的 NumPy 数组；
数组中的物理动作位于通信动作之前。更多细节请参阅 environment.py。
"""

def make_env(scenario_name, benchmark=False):
    '''
    创建 MultiAgentEnv 对象 env。它的用法与 Gym 环境类似，可调用
    env.reset() 和 env.step()；使用 env.render() 在屏幕上查看环境。

    输入：
        scenario_name   :   ./scenarios/ 中的场景名称（不含 .py 扩展名）
        benchmark       :   是否生成基准测试数据（通常仅在评估时启用）

    常用 env 属性（参见 environment.py）：
        .observation_space  :   返回每个智能体的观测空间
        .action_space       :   返回每个智能体的动作空间
        .n                  :   返回智能体数量
    '''
    from mpe.environment import MultiAgentEnv
    import mpe.scenarios as scenarios

    # 从脚本加载场景
    scenario = scenarios.load(scenario_name + ".py").Scenario()
    # 创建世界
    world = scenario.make_world()
    # 创建多智能体环境
    if benchmark:        
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation, scenario.benchmark_data)
    else:
        env = MultiAgentEnv(world, scenario.reset_world, scenario.reward, scenario.observation)
    return env
