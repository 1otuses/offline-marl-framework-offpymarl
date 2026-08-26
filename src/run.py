import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
import json

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot


def run(_run, _config, _log):

    # 检查参数有效性
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    
    # 设置记录器
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # 已经在 main.py 中设置
    #unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    #args.unique_token = unique_token

    results_save_dir = args.results_save_dir

    if args.use_wandb:
        args.use_tensorboard = False
    # assert args.use_tensorboard and args.use_wandb
    
    
    if args.use_tensorboard and not args.evaluate:
        # 仅在训练模式下记录 TensorBoard 日志
        tb_exp_direc = os.path.join(results_save_dir, 'logs')
        logger.setup_tb(tb_exp_direc)
        
    
    
    if args.use_wandb and not args.evaluate:
        wandb_run_name = args.results_save_dir.split('/')
        wandb_run_name = "/".join(wandb_run_name[wandb_run_name.index("results")+1:])
        wandb_exp_direc = os.path.join(results_save_dir, 'logs')
        logger.setup_wandb(wandb_exp_direc, project=args.wandb_project_name, name=wandb_run_name,
                           run_id=args.resume_id, config=args)
    # 写入配置文件
    config_str = json.dumps(vars(args), indent=4)
    with open(os.path.join(results_save_dir, "config.json"), "w") as f:
        f.write(config_str)
    # 设置模型保存目录
    args.model_save_dir = os.path.join(results_save_dir, 'models')

    
    # 默认启用 Sacred
    logger.setup_sacred(_run)

    # 运行并训练
    run_sequential(args=args, logger=logger)

    # 完成后清理资源
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # 确保框架进程真正退出
    os._exit(os.EX_OK)


def evaluate_sequential(args, runner):

    for _ in range(args.test_nepisode):
        runner.run(test_mode=True)

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args, logger):

    # 初始化运行器以便我们可以获得环境信息
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # 在这里设置方案和组
    env_info = runner.get_env_info()
    for k, v in env_info.items():
        setattr(args, k, v)

    # 默认/基本方案
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)

    # 在这里设置多智能体控制器
    mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)

    # 将数据方案交给运行器
    runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # 学习器
    learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return

        # 遍历 args.checkpoint_path 中的所有文件
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # 检查名称为数字的目录
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # 选择最大时间步长
            timestep_to_load = max(timesteps)
        else:
            # 选择最接近 load_step 的时间步
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

    # 开始训练
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time

    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))

    while runner.t_env <= args.t_max:

        # 每次运行一个完整回合
        episode_batch = runner.run(test_mode=False)
        buffer.insert_episode_batch(episode_batch)

        if buffer.can_sample(args.batch_size):
            # 在并行运行与按回合运行之间平衡
            # 采样 batch_size 个回合，用于大小为 batch_size 的训练批次
            terminated = False
            for _run in range(runner.batch_size):
                episode_sample = buffer.sample(args.batch_size)

                # 将批次截断为仅填充的时间步长
                max_ep_t = episode_sample.max_t_filled()
                episode_sample = episode_sample[:, :max_ep_t]

                if episode_sample.device != args.device:
                    episode_sample.to(args.device)

                terminated = learner.train(episode_sample, runner.t_env, episode)
                if terminated is not None and terminated:
                    break

        # 偶尔执行一次测试运行
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            for _ in range(n_test_runs):
                runner.run(test_mode=True)

        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            
            save_path = os.path.join(args.model_save_dir, str(runner.t_env))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))
            learner.save_models(save_path)
            model_save_time = runner.t_env
            
            # model_save_time = runner.t_env
            # local_results_path = os.path.join(dirname(dirname(abspath(__file__))), "results") # default.yaml -> results by default
            
            # match args.env:
            #     case "sc2":
            #         save_path = os.path.join(local_results_path, args.env, args.env_args["map_name"], args.name, "models", args.unique_token, str(runner.t_env))
            #     case "mpe":
            #         save_path = os.path.join(local_results_path, args.env, args.env_args["scenario_name"], args.name, "models", args.unique_token, str(runner.t_env))
            #     case _:
            #         save_path = os.path.join(local_results_path, args.env, args.name, "models", args.unique_token, str(runner.t_env))
            
            # # 结果/sc2/2s3z/qmix/models/unqiue_token/t_env
            # os.makedirs(save_path, exist_ok=True)
            # logger.console_logger.info("Saving models to {}".format(save_path))

            # # 学习者应该处理保存/加载——将 actor 保存/加载委托给 mac，
            # # 使用适当的文件名保存 Critic 和优化器状态
            # learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    # 关闭环境
    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # 设置 CUDA 标志
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config
