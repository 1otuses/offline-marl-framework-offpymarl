# Decision Diffuser / trajectory diffusion 向多智能体扩展的文献核对

> 核对日期：2026-08-29  
> 范围：只采用论文、会议论文集、OpenReview、作者官方项目页或官方代码仓库。重点是 cooperative offline MARL；不把交通轨迹预测、multi-robot motion planning、online MARL 自动归入同一类。  
> 仓库落点：本仓库此前没有通用 `notes/` 或 `docs/` 约定（只有第三方 SMAC 文档），因此新建此独立记录，不修改现有代码。

## 结论先行

1. 用户所称 **“Dof” 基本可以确定是 DoF**：*DoF: A Diffusion Factorization Framework for Offline Multi-Agent Reinforcement Learning*（ICLR 2025），而不是一个泛指缩写。其官方实现同时给出 `DoF_Policy` 和 `DoF_Trajectory` 两条路线。[ICLR 论文页](https://proceedings.iclr.cc/paper_files/paper/2025/hash/5623c35f3ab5e2c72aeb3abce27dc28f-Abstract-Conference.html)；[官方代码](https://github.com/xmu-rl-3dv/DoF)
2. 除 MADiff、DoF 外，已经有一批真正的 diffusion-based offline MARL 方法；但其中只有少数把 diffusion 直接当执行策略或轨迹规划器，大量工作只是用 diffusion **合成/增强离线数据**，随后仍训练常规 offline MARL policy。
3. 对本仓库而言，完整移植 MADiff 不是“换一个 agent 网络”的小改动。MADiff 的轨迹窗口、扩散训练、inverse dynamics、在线重规划、归一化与数据格式会横跨数据层、learner、controller、runner 和 checkpoint。若研究创新本身围绕轨迹扩散与执行信息结构，继续在 MADiff 项目中做通常风险最低；若创新围绕数据增强或希望复用 offpymarl 的 ICQ/CQL/OMAR/QMIX 评估，则把生成器做成离线预处理模块更自然。

## 方法表：哪些是真正的 offline MARL

表中“直接”表示 diffusion 参与最终动作策略或测试时规划；“增强”表示 diffusion 只在训练期生成数据，最终执行仍由另一个 MARL policy 完成。

| 方法 | 年份/状态 | diffusion 的对象与角色 | 任务与信息结构 | 真正 offline MARL？ | 与 Decision Diffuser 的关系 |
|---|---|---|---|---|---|
| **DOM2** | TMLR 2026 | **直接策略**：每个 agent 用 few-step diffusion policy 生成当前动作；轨迹回报用于 data reweighting，不是完整轨迹规划 | cooperative MPE、MA-MuJoCo；论文明确称 fully decentralized training and execution，每个 agent 基于自己的 observation/action 学习 | 是 | 借用 diffusion policy 的多模态动作建模，但不是 DD 式整段状态轨迹 + inverse dynamics；对当前 PyMARL controller 更接近 |
| **EAQ** | ICML 2024 SPIGM workshop | **增强**：Conv1D/DDPM 生成包含多 agent observation、action、global reward、terminal 的整 episode，然后交给 CQL/BCQ 等 learner | cooperative SMAC、Dec-POMDP、共享 team reward；生成器训练阶段看 joint episode | 是，但 diffusion 不是最终 policy | 与 trajectory diffusion 接近的是“整 episode 生成”，不是测试时 receding-horizon planner |
| **MADiTS** | ICLR 2025 | **增强**：diffusion 生成高质量 coordination segments，再以 forward/backward dynamics constraint、offline credit assignment 反复 stitch 成 episode | cooperative MPE/SMAC 等 imbalanced offline datasets；合成阶段使用 joint segment，最终 policy 可沿用 offline MARL | 是 | 把 trajectory diffusion 用作 trajectory stitching，不直接控制环境 |
| **INS** | ICLR 2025 | **增强**：对 joint transition `(o,a,o',r)` 做 interaction-aware diffusion synthesis，稀疏 attention 建模 agent interactions，再按 value 选择样本 | cooperative MPE、SMAC；支持连续和离散动作；最终 policy 是任意 offline MARL learner | 是 | transition-level synthesis，不是 DD 式 horizon planner |
| **MCGD** | ICML 2025 | **直接策略**：对 coordination graph 的离散 edge 做 categorical diffusion，对 action node 做 neighbor-dependent anisotropic diffusion；最终抽取每个 agent 动作 | cooperative MPE、MA-MuJoCo、SMAC；论文声称 decentralized execution，并支持离散/连续动作 | 是 | 是 action/coordination-graph diffusion，而非整段 trajectory diffusion；对动态拓扑和鲁棒性更聚焦 |
| **OMSD / Sequential Score Decomposition** | ICML 2026 accepted | **直接策略正则**：diffusion 学 joint behavior-policy scores，再按 agent 顺序分解 score，注入 joint-action critic 以约束 decentralized policies | cooperative continuous-control MPE、MA-MuJoCo；shared team reward，decentralized execution | 是 | 使用 diffusion score 做 multimodal behavior regularization，不生成 DD 风格状态轨迹 |
| **GCD** | AAMAS 2026 | **直接 trajectory policy/planner**：以 team communication graph embedding 条件化 multi-agent diffusion，并用 classifier-free guidance 适应 team composition | SMACv2；论文目标是在 decentralized execution 下利用显式通信，对 unseen team compositions 泛化 | 是 | 与 MADiff 最接近的后续路线之一，但执行时加入 communication cue，信息权利不等同于 strict-local MADiff variant |
| **CODI** | AAMAS 2026 | **增强**：LLM-distilled agent-quality label + return 条件的 compositional trajectory diffusion，生成 segment 后 stitch 成完整轨迹 | imbalanced cooperative MPE、SMAC、SMACv2；最终用 offpymarl 中的常规 offline learner | 是 | DD 式 trajectory generator 被用于 fine-grained agent-level composition；不是最终 policy |
| **CODA** | RLC 2026 accepted / arXiv | **增强**：full-horizon centralized joint trajectory diffusion；按当前 joint policy 对采样分布做 on-policy guidance，动态扩充 replay data | cooperative continuous polynomial games、MA-MuJoCo；训练可用 centralized generator，测试仍由 base decentralized policies 执行 | 是；“on-policy”指生成条件随 learner 更新，不代表访问真实环境采新数据 | 与整段 joint trajectory diffusion高度相关，但目标是恢复 offline training 中缺失的 co-adaptation，而不是以 diffusion planner 直接执行 |
| **MF-Diffuser** | arXiv 2026 preprint | **直接 trajectory planner**：在单 agent trajectory 的分布/Wasserstein mean field 上扩散，以 coarse-to-fine 方式扩到上千 agents | mean-field stage games、sequential dynamics、team-competitive battle；依赖 exchangeability/mean-field 信息结构 | 是（preprint） | 明确从 trajectory diffusion 出发，但适用的是同质大规模 mean-field regime，不是普通 SMAC 小队 CTDE 的直接替代 |
| **Individual-CBF guided diffusion** | IFAC World Congress 2026 accepted | **直接 trajectory planner**：在状态轨迹 denoising 中嵌入 individual control barrier functions，再用 inverse dynamics 恢复动作 | safe MPE、Safe MA-MuJoCo；论文采用 CTDE，并为每个 agent 构造 interaction-relevant state | 是 | 是最接近 DD“状态轨迹 + inverse dynamics”的安全多智能体扩展，但研究问题是约束/安全 |

### 主表的一手来源

- DOM2：最新版题为 *Improving Generalization and Data Efficiency with Diffusion in Offline Multi-agent RL*；arXiv 页面标注 TMLR 版本，摘要与论文均明确 diffusion action generator、trajectory data reweighting、MPE/MA-MuJoCo。[arXiv](https://arxiv.org/abs/2307.01472)；[TMLR/OpenReview PDF](https://openreview.net/pdf/f32f1cdb339637fb71cb24b4f355eb89f0ab5adf.pdf)
- EAQ：论文明确固定 multi-agent episode 数据、cooperative shared objective 以及 SMAC 增强实验。[arXiv](https://arxiv.org/abs/2408.13092)；[OpenReview PDF](https://openreview.net/pdf?id=Bs8uwhKaPO)
- MADiTS：ICLR 论文摘要明确 diffusion segment generation、双向 dynamics constraint、offline credit assignment 与 trajectory stitching。[ICLR 论文页](https://proceedings.iclr.cc/paper_files/paper/2025/hash/3e7cf447f21cd11c846463affefce665-Abstract-Conference.html)；[论文 PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/3e7cf447f21cd11c846463affefce665-Paper-Conference.pdf)
- INS：论文算法是先训练 joint-transition diffusion，再生成、value-select 数据，最后训练 offline MARL policy。[ICLR PDF](https://proceedings.iclr.cc/paper_files/paper/2025/file/1cc864add5073ddac8e081d1f099e10d-Paper-Conference.pdf)；[官方代码](https://github.com/fyqqyf/INS)
- MCGD：ICML/PMLR 正式论文明确 sparse coordination graph、categorical edge diffusion、anisotropic action diffusion，以及 MPE/MA-MuJoCo/SMAC offline experiments。[PMLR 论文页](https://proceedings.mlr.press/v267/zeng25f.html)
- OMSD：论文摘要明确 diffusion joint behavior score、sequential decomposition、joint-action critics 与 decentralized execution，arXiv 当前标注 ICML 2026 accepted。[arXiv](https://arxiv.org/abs/2505.05968)；[官方代码](https://github.com/qiaodan-cuhk/OMSD)
- GCD：AAMAS 2026 正式论文与作者代码；官方代码同时包含 GCD、MADiff、HiSSD/ODIS 和传统 offline MARL baselines。[AAMAS 论文 PDF](https://ifaamas.csc.liv.ac.uk/Proceedings/aamas2026/pdfs/BMST1644.pdf)；[官方代码](https://github.com/CORE-Robotics-Lab/GCD)
- CODI：官方仓库明确包含 `CODI_diffusion`、agent-quality labeler、offpymarl 数据采集/训练以及 synthetic dataset 阶段。[AAMAS 2026 论文集目录](https://ifaamas.csc.liv.ac.uk/Proceedings/aamas2026/forms/contents.htm)；[论文附录](https://www.lamda.nju.edu.cn/lilh/file/codi.pdf)；[官方代码](https://github.com/lilh76/CODI)
- CODA：论文明确 full-horizon joint trajectory diffusion，生成 `(s,o,a,r,...)`，并用 current joint policy guidance 产生随训练变化的 synthetic data；真实环境不参与训练。[arXiv](https://arxiv.org/abs/2604.23308)；[OpenReview PDF](https://openreview.net/pdf?id=XfAR4v6iZM)
- MF-Diffuser：[arXiv](https://arxiv.org/abs/2605.30190)
- Individual-CBF guided diffusion：[arXiv](https://arxiv.org/abs/2606.12640)

## 不应混入主表的相邻工作

### 1. Online MARL

**DDPL (Decentralized Diffusion Policy Learning)** 用 diffusion policy 增强 cooperative MARL exploration，但训练过程需要持续环境交互，属于 online MARL，不是 offline MARL，也不是 Decision Diffuser 式 trajectory planning。[arXiv](https://arxiv.org/abs/2605.07101)

### 2. Multi-agent motion prediction / motion planning

- **MotionDiffuser** 学的是交通参与者未来轨迹的 joint predictive distribution，评测是 Waymo motion prediction；它没有从 offline reward dataset 学 cooperative policy。[arXiv](https://arxiv.org/abs/2306.03083)
- **SMD / Simultaneous Multi-Robot Motion Planning**、**DG-MAP** 解决碰撞约束下的多机器人几何路径规划，不是 Dec-POMDP offline policy learning。[SMD/ICML 2025](https://proceedings.mlr.press/v267/liang25e.html)；[DG-MAP/CoRL 2025](https://proceedings.mlr.press/v305/parimi25a.html)

这些工作在 permutation equivariance、pairwise constraints、projected denoising 方面可能提供模块灵感，但不能作为 SMAC offline MARL 的同类 baseline。

### 3. 尚未稳定的候选或非 diffusion 邻域

- **MASTARS** 使用 MADiff 架构进行 agent-wise sequential trajectory inpainting 和数据增强，但截至核对日作者主页仍标注 NeurIPS 2026 under review，不能当作已发表结论。[OpenReview PDF](https://openreview.net/pdf?id=8mQqCCxKZa)
- **Flow Matching for Offline RL with Discrete Actions** 将 continuous-time Markov chain / flow matching 扩展到 factorized multi-agent discrete actions，和 diffusion policy 邻近，但严格说不是 DDPM/Decision Diffuser。[arXiv](https://arxiv.org/abs/2602.06138)
- 已撤回的 *Diffusion Models for Offline Multi-agent Reinforcement Learning with Safety Constraints* 不应作为有效 baseline；arXiv 明示 “experiment and method plan are abolished and need to be redesigned”。[撤回页面](https://arxiv.org/abs/2407.00741)

## 对当前 offpymarl 的工程映射与风险

### 当前仓库已有的有利基础

- `OfflineBufferH5` 已能把 episode 组织为 `[B,T,N,...]`，按 `filled` 得到有效 horizon，并包含 `obs/state/actions/actions_onehot/avail_actions/reward/terminated` 等轨迹字段（`src/components/offline_buffer.py`）。
- `EpisodeBatch` 已有固定时序 tensor 与 `filled` mask（`src/components/episode_buffer.py`）。
- `EpisodeRunner` 的测试路径是真实 SMAC rollout，可报告 return/win-rate，而不是只在静态数据上算 loss（`src/runners/episode_runner.py`）。
- 当前 `3m` OG-MARL H5 转换已经给出了适合完整 episode 学习的输入，因此**数据本身不是最大障碍**。

### 完整迁移 MADiff：高工程风险

主要 seam 不是 mixer，而是执行接口：

1. **数据适配与归一化**：MADiff 官方代码把 OG-MARL 轨迹预处理为 `.npy` full episodes，并有自己的 dataset/normalizer；本仓库只加载 H5。需要定义唯一的 trajectory schema、padding/death mask、condition mask 和 normalization contract。[MADiff 官方代码](https://github.com/zbzhu99/madiff)
2. **模型与 loss**：需要新增 multi-agent denoiser、diffusion schedule、return/condition encoder、inverse dynamics，并决定 CTCE 或 CTDE。它不是把 `RNNAgent` 换成 UNet 就结束。
3. **controller/runner**：当前 `BasicMAC.select_actions()` 假设“一次 forward -> 当前 timestep logits -> action selector”。trajectory diffusion 需要构造 history condition、反复 denoise、选择第一步、下一环境步重新规划；通常应新增独立 MAC，而不是塞进现有 `BasicMAC`。
4. **SMAC 离散动作合法性**：生成或 inverse-dynamics 输出必须处理 `avail_actions`；死亡 agent、padding step、episode tail 均要 mask。仅在最终 logits 后屏蔽非法动作，可能使训练目标与执行分布不一致。
5. **信息权利**：训练时可用 global `state` 不等于执行时可用。必须分别验证 strict-local observation/history、显式 communication/public cue、centralized controller 三种 regime；否则结果不能称 CTDE decentralized execution。
6. **评估成本**：每环境步多轮 denoising 会显著改变 rollout latency。除 return/win-rate 外应记录 diffusion steps、每步推理时延、显存和非法动作率。
7. **保存/恢复**：当前 MAC 通常只保存 `agent.th`；完整方案还要保存 denoiser、inverse model、EMA、normalizer 和 diffusion schedule，并验证 checkpoint reload 后真实 rollout 等价。

因此，“只迁移 MADiff 中一部分思路”只有在部分被严格限定时才算中等工程量。例如：

- 只移植 **inter-agent attention denoiser**，离线输入/输出 contract 不变；
- 只使用 MADiff 生成 synthetic episodes，再由现有 learner 训练；
- 只做独立研究 prototype，不先接完整 Sacred/PyMARL 训练与评估生命周期。

若目标是复现 MADiff 的 CTDE/CTCE、history conditioning、inverse dynamics、receding-horizon execution 与 SMAC evaluation，则应视为新算法栈，而非局部 patch。

### 三条实施路线的相对风险

| 路线 | 对现仓库改动 | 科研适用性 | 建议 |
|---|---|---|---|
| A. 在 MADiff 仓库上创新 | 复用其完整 trajectory diffusion、SMAC configs、OG-MARL 转换和 evaluator | 创新点涉及 coordination denoising、mode conditioning、decentralized trajectory generation、inverse dynamics | **优先**；先在成熟链路验证创新，再决定是否回迁 benchmark |
| B. 在 offpymarl 外部做 diffusion augmentation | 新增独立 generator，把 synthetic H5 写回现有 `dataset/sc2/...`；原 learners/controllers 不变 | 创新点类似 EAQ/MADiTS/INS/CODI/CODA，或想横向提升 CQL/ICQ/OMAR | **工程风险最低**，但必须验证生成轨迹的 dynamics consistency 和无信息泄漏 |
| C. 在 offpymarl 内实现完整 MADiff-like policy | 新 MAC、learner、models、normalizer、runner planning path、checkpoint 与 configs | 目标是让所有算法共享相同数据/测试框架，或研究 PyMARL-native discrete trajectory diffusion | **风险最高**；只有框架统一本身是必要研究目标时才值得 |

## 可验证的最小迁移门槛

若以后选择路线 C，建议把成功标准固定为以下顺序，而不是直接跑大实验：

1. 单 batch 能 round-trip：H5 episode -> normalized trajectory -> denoiser loss，padding/death 不贡献 loss。
2. inverse dynamics 在 held-out dataset 上输出合法动作，并报告 accuracy 与 invalid-action rate。
3. 仅用每个 agent 的允许执行信息，重复采样得到 `[B,H,N,*]`，shape/mask 恒成立。
4. `3m` 单 episode 真实 rollout 能结束、保存 replay，并在固定 seed/checkpoint reload 后复现动作序列。
5. 小数据 overfit 后 return 明显高于 random；再进入 Good/Medium/Poor 多 seed 正式评估。

这五项通过之前，训练 loss 下降不能证明 multi-agent diffusion policy 可执行或协调有效。
