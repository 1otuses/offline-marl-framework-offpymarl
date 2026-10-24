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

此外，本项目现已支持使用 [OG-MARL](https://github.com/instadeepai/og-marl) 数据集。为将其接入 (Off)PyMARL pipeline，可使用 `src/transform_ogmarl_h5.py` 将 Vault 转换为 HDF5。JAX/Flashbax 的现行依赖与主训练环境的 `numpy<1.24` 约束不兼容，因此请使用独立转换环境：

```bash
conda create -n offpymarl-convert --clone offpymarl -y
conda activate offpymarl-convert
python -m pip install --only-binary=:all: -r requirements-transform.txt

python src/transform_ogmarl_h5.py \
  --dataset-path /path/to/Vaults/og_marl/smac_v1 \
  --map-name 3m \
  --og-quality Good
```

`src/transform_data.ipynb` 保留为早期 notebook 示例，新的批量转换建议使用上述 CLI。

基于 OG-MARL 的 benchmarking 目前仍在进行中……

## 离线训练

```bash
python src/main.py --offline --config=<alg_name> --env-config=sc2_offline --map_name=<sc2_map>  --offline_data_quality=<data_quality> --seed=<seed> --t_max=40000 --test_interval=250 --log_interval=250 --runner_log_interval=250 --learner_log_interval=250 --save_model_interval=100001 
```
更多信息请参阅 `test/offline_train`。

算法配置现按 `src/config/algs/online/` 和 `offline/` 分类。推荐显式指定
`--config=offline/cfcql`；原有 `--config=cfcql` 在算法名称唯一时仍可使用。
在线训练和数据采集可使用 `--config=online/qmix`。

`3m/expert` 的统一离线启动方式（先激活 `offpymarl` 环境）：

```bash
# 参数依次为算法、地图、数据质量、seed、GPU；额外参数放在末尾覆盖默认设置。
bash test/offline_train/train.sh bc 3m expert 1 0
bash test/offline_train/train.sh vdn_cql 3m expert 1 0
bash test/offline_train/train.sh itd3_bc 3m expert 1 0
bash test/offline_train/train.sh matd3_bc 3m expert 1 0
bash test/offline_train/train.sh icq 3m expert 1 0
bash test/offline_train/train.sh omar 3m expert 1 0
bash test/offline_train/train.sh omiga 3m expert 1 0
# qmix_cql、cfcql 使用相同接口。

# 模型目录需要替换为本次训练生成的 models 目录。
bash test/offline_train/evaluate.sh bc 3m expert /path/to/models 200000 1000 0
```

训练脚本使用 200000 次 Q/策略更新，每 10000 次更新评估及保存。
CFCQL 额外进行 `behaviour_t_max=20000` 次行为模型批次更新，行为模型默认保存在
本次运行目录的 `behaviour/`；不同运行不自动共享。续训旧实验时可通过
`--behaviour_model_path=/path/to/original/bc_model` 指定原行为模型。
独立评估脚本运行 100 个 episode，结果写入评估运行目录的 `evaluation.json`。

离线数据量由 `sc2_offline.yaml` 中 `offline_max_buffer_size` 控制，当前 `0`
表示载入指定目录全部 HDF5 episodes。`offline_batch_size` 控制训练批次：ICQ 为 16，
其余离线配置为 32。`batch_size`、`buffer_size` 不控制此离线采样路径。
ICQ 和 OMAR 保留并行 runner 配置；当前 hard target 判据沿用 episode 计数，
分别对应约每 60 和 250 次更新同步，改变 `batch_size_run` 会改变该周期。

当前默认 `use_corrected_terminated=False`，对末行仍为 terminal transition 的 episode
在采样时补零 bootstrap 占位帧，保留最后一步奖励；原 HDF5 文件保持原样。
`True` 可选择旧的提前终止处理，但会屏蔽末次奖励。
新默认与此前 QMIX+CQL/CFCQL 实验的数据处理口径不同，正式横向比较应统一口径后重跑。
本轮修复通过 9 个离线 learner 的 CPU 小批次更新及模型保存/加载检查，
算法参数仍是本仓库起始配置，论文指标及多种子性能需要另行实验确认。

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
