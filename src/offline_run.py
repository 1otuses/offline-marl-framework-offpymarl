import os
import pprint
import time
import threading
import torch as th
from tqdm import tqdm

from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from utils.checkpointing import load_training_state, save_training_state, select_checkpoint
import json

from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.offline_buffer import OfflineBuffer
from components.transforms import OneHot

# 用收集或下载的离线数据进行offline算法训练
def run(_run, _config, _log):

    # 检查参数有效性
    print(_config["use_cuda"])
    _config = args_sanity_check(_config, _log)
    
    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"
    

    logger = Logger(_log)

    _log.info("实验参数:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

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
    # 在离线训练中，我们使用 t_max 来表示迭代次数
    
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
        print("use cuda")
        learner.cuda()


    resume_state = {"t_env": 0, "episode": 0, "restored": False}
    resume_checkpoint = None
    if args.checkpoint_path != "":
        model_path, timestep_to_load = select_checkpoint(args.checkpoint_path, args.load_step)

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)

        if args.evaluate or args.save_replay:
            evaluate_sequential(args, runner)
            return

        if args.continue_training:
            resume_checkpoint = (model_path, timestep_to_load)
    
    # 创建离线数据
    match args.env:
        case "sc2":
            args.map_name = args.env_args["map_name"]
        case "gymma":
            env_name, args.map_name = args.env_args['key'].split(':')
            args.env = env_name
        case _:
                raise NotImplementedError("Do not support such envs: {}".format(args.env))
        
    offline_buffer = OfflineBuffer(args, args.map_name, args.offline_data_quality,
                                   args.offline_bottom_data_path, args.offline_max_buffer_size, 
                                   shuffle=args.offline_data_shuffle) # 设备默认CPU

    if getattr(args, "train_behaviour", False):
        train_behaviour_sequential(args, logger, learner, runner, offline_buffer)

    if resume_checkpoint is not None:
        model_path, timestep_to_load = resume_checkpoint
        resume_state = load_training_state(model_path, learner, timestep_to_load)
        if not resume_state["restored"]:
            logger.console_logger.warning(
                "training_state.th was not found; resume t_env from the checkpoint directory"
            )
        logger.console_logger.info(
            "Continuing training from t_env={} and episode={}".format(
                resume_state["t_env"], resume_state["episode"]
            )
        )
    
    
    logger.console_logger.info("Beginning  offline training with {} iterations".format(args.t_max))
    final_t_env, final_episode = train_sequential(
        args,
        logger,
        learner,
        runner,
        offline_buffer,
        resume_state["t_env"],
        resume_state["episode"],
    )

    if args.save_model:
        save_path = os.path.join(args.model_save_dir, str(final_t_env))
        os.makedirs(save_path, exist_ok=True)
        logger.console_logger.info("Save final model checkpoint in {}".format(save_path))
        learner.save_models(save_path)
        save_training_state(save_path, final_t_env, final_episode, learner)
    
    runner.close_env()
    logger.console_logger.info("训练完成")

def train_sequential(args, logger, learner, runner, offline_buffer, start_t_env=0, start_episode=0):
    t_env = start_t_env
    episode = start_episode
    t_max = args.t_max
    model_save_time = t_env if args.continue_training else 0
    last_test_T = t_env
    last_log_T = t_env
    start_time = time.time()
    last_time = start_time
    test_time_total = 0

    batch_size_train = args.offline_batch_size
    batch_size_run = args.batch_size_run # 并行环境数
    n_test_runs = max(1, args.test_nepisode//batch_size_run)
    test_start_time = time.time()

    with th.no_grad():
        runner.t_env = t_env
        for _ in range(n_test_runs):
            runner.run(test_mode=True)
    
    test_time_total += time.time() - test_start_time

    training_progress = tqdm(
        total=t_max,
        initial=min(t_env, t_max),
        desc="Offline training",
        unit="update",
        dynamic_ncols=True,
    )
    while t_env < t_max:
        episode_sample = offline_buffer.sample(batch_size_train)
        if episode_sample.device != args.device:
            episode_sample.to(args.device)
            
   
        learner.train(episode_sample, t_env, episode)
        
        t_env += 1
        training_progress.update(1)
        episode += batch_size_run
        # 偶尔执行一次测试运行并进行最终评估
        if (t_env - last_test_T) / args.test_interval >= 1 or t_env >= t_max:
            test_start_time = time.time()
            with th.no_grad():
                runner.t_env = t_env
                for _ in range(n_test_runs):
                    runner.run(test_mode=True)
            test_time_total += time.time() - test_start_time

            logger.console_logger.info("Step: {}/{}".format(t_env, t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}. FPS {:.2f}. Test time cost: {}".format(
                time_left(last_time, last_test_T, t_env, t_max), time_str(time.time() - start_time), (t_env - last_test_T) / (time.time() - last_time), time_str(test_time_total)
            ))
            last_time = time.time()
            last_test_T = t_env
        
        if args.save_model and (t_env-model_save_time >= args.save_model_interval or model_save_time==0):
            save_path = os.path.join(args.model_save_dir, str(t_env))
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))
            learner.save_models(save_path)
            save_training_state(save_path, t_env, episode, learner)
            model_save_time = t_env
        
        if (t_env - last_log_T) >= args.log_interval:
            last_log_T = t_env
            logger.log_stat("episode", episode, t_env)
            logger.print_recent_stats()

    training_progress.close()
    logger.console_logger.info("Finish training sequential")
    return t_env, episode
            

def train_behaviour_sequential(args, logger, learner, runner, offline_buffer):
    if args.behaviour_model_path == "":
        default_behaviour_model_path = os.path.join("dataset", args.env, args.map_name, args.offline_data_quality, "bc_model")
    else:
        default_behaviour_model_path = args.behaviour_model_path
    
    if os.path.exists(default_behaviour_model_path):
        logger.console_logger.info("Loading behaviour model from {}".format(default_behaviour_model_path))
        learner.load_behaviour_model(default_behaviour_model_path)
        return
    else:
        logger.console_logger.info("Beginning training behaviour proxy policy as CFCQL does with {} steps".format(args.behaviour_t_max))
        training_steps = 0
        batch_size_train = args.offline_batch_size
        episode = 0
        last_log_T = - args.log_interval - 1
        behaviour_progress = tqdm(
            total=max(args.behaviour_t_max + 1, 0),
            desc="Behaviour policy training",
            unit="update",
            dynamic_ncols=True,
        )
        while training_steps <= args.behaviour_t_max:
            episode_sample = offline_buffer.sample(batch_size_train)
            if episode_sample.device != args.device:
                episode_sample.to(args.device)
            learner.train_behaviour(episode_sample)
            training_steps += 1 # CFCQL计算“填充”但我直接+1
            behaviour_progress.update(1)
            
            episode += batch_size_train
            if (training_steps - last_log_T) >= args.log_interval:
                last_log_T = training_steps
                logger.log_stat("episode", episode, training_steps)
                logger.print_recent_stats()

        behaviour_progress.close()
        os.makedirs(default_behaviour_model_path, exist_ok=True)
        logger.console_logger.info("Saving behaviour models to {}".format(default_behaviour_model_path))
        learner.save_behaviour_model(default_behaviour_model_path)
        logger.console_logger.info("Finish training behaviour proxy policy")
            
def args_sanity_check(config, _log):

    if config.get("continue_training", False) and config.get("checkpoint_path", "") == "":
        raise ValueError("continue_training=True requires checkpoint_path")
    if config.get("continue_training", False) and not config.get("save_model", False):
        _log.warning("continue_training is enabled but save_model is False; no new checkpoint will be saved")

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
