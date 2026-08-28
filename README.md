# Offline MARL 框架 - OffPyMARL
> 🚧 本仓库尚未准备好正式发布，benchmarking 仍在进行中。🚧

OffPyMARL 为以下部分 Offline MARL 算法提供了非官方、经 benchmark 的 PyTorch 实现：

- [BC](https://arxiv.org/abs/1805.01954)
- [VDN/QMIX+CQL](https://arxiv.org/abs/2006.04779)（本项目提供三种 conservatism term 计算方式，推荐使用 `"individual"` 和 `"global_simplified"`）
- [ITD3/MATD3+BC](https://arxiv.org/abs/2106.06860)
- [MAICQ](https://arxiv.org/abs/2106.03400)
- [OMAR](https://arxiv.org/abs/2111.11188)（Centralized Critic）
- [CFCQL](https://arxiv.org/abs/2309.12696)
- [OMIGA](https://arxiv.org/abs/2307.11620)

我们还为 BC、QMIX+CQL 和 MATD3+BC 实现了相应的 [Multi-Task 版本](https://github.com/zzq-bot/mt_offpymarl)，用于处理 population-invariance 问题。

## 安装

```bash
conda create -n offpymarl python=3.10 -y
conda activate offpymarl
pip install -r requirements.txt
bash install_sc2.sh # 如果本机尚未安装 StarCraftII
bash install_smac.sh
```

## 收集数据

```bash
python src/main.py --collect --config=<alg> --env-config=sc2_collect --map_name=<map_name> --offline_data_quality=<quality> --save_replay_buffer=<whether_to_save_replay>
--num_episodes_collected=<num_episodes_per_collection> --stop_winrate=<stop_winrate> --seed=<seed>
```
`quality` 可从 `['random', 'medium', 'expert', 'full']` 中选择。

如果设置了 `save_replay_buffer`，程序会根据相应的 `offline_data_quality` 参数生成 `'medium_replay'`、`'expert_replay'` 离线数据。

## 离线数据集

我们在 [Google Drive](https://drive.google.com/drive/folders/1FzSetZJ89Vq99o8LQHXiIxU9_tS70laE?usp=sharing) 中提供了小规模数据集（少于 4,000 个 episodes），用于快速开始实验。
将完整数据集放入 `dataset` 目录后，即可使用我们预定义的任务集运行实验。

此外，本项目现已支持使用 [OG-MARL](https://github.com/instadeepai/og-marl) 数据集。为将其接入 (Off)PyMARL pipeline，我们将数据转换为 H5 文件；转换示例位于 `src/transform_data.ipynb`，详细信息请参阅该文件。

基于 OG-MARL 的 benchmarking 目前仍在进行中……

## 离线训练

```bash
python src/main.py --offline --config=<alg_name> --env-config=sc2_offline --map_name=<sc2_map>  --offline_data_quality=<data_quality> --seed=<seed> --t_max=40000 --test_interval=250 --log_interval=250 --runner_log_interval=250 --learner_log_interval=250 --save_model_interval=100001 
```
更多信息请参阅 `test/offline_train`。

## 引用 OffPyMARL

如果您在工作中使用了 OffPyMARL，请使用以下 BibTeX 进行引用：

```tex
@misc{OffPyMARL,
  author = {Ziqian Zhang},
  title = {OffPyMARL: Benchmarked Implementations of Offline Reinforcement Learning Algorithms},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/zzq-bot/offline-marl-framwork-offpymarl}},
}
```

## 致谢

感谢 [ODIS](https://github.com/LAMDA-RL/ODIS) 提供数据收集相关代码，并感谢 [EPyMARL](https://github.com/uoe-agents/epymarl) 提供 MADDPG 相关代码。
