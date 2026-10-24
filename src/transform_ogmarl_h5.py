#!/usr/bin/env python3
"""Convert an OG-MARL SMAC Vault into OffPyMARL episode HDF5 files.

Example:
    python src/transform_ogmarl_h5.py \
        --dataset-path /home/lotus/lotus/lhh/offline_datasets/Vaults/og_marl/smac_v1 \
        --map-name 3m \
        --og-quality Good

``dataset-path`` must contain ``<map-name>.vlt``. By default, output is
written to ``dataset/sc2/<map-name>/<quality>`` under the repository root.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from types import SimpleNamespace as SN


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ENV_CONFIG = Path(__file__).resolve().parent / "config" / "envs" / "sc2.yaml"
QUALITY_MAP = {
    "Good": "expert",
    "Medium": "medium",
    "Poor": "poor",
}


# 将 OG-MARL 下载的数据.vlt 形式转化为 OffPyMARL 的 .h5 形式
# .vlt数据保存agents的transition全部数据,通过terminated和truncated来判断episode是否结束
# .h5数据需要显示保存每个agent的episode完整轨迹,根据terminated和truncated来剪断成若干episodes
# 每个episode有一个最大长度,而数据集的每个episode长度不等;需要拓展成固定最长长度.
# 拓展后会伴随生成一个filled,用来区分哪些是数据,哪些是拓展数据.
# 还会生成一个actions_onehot数据,用来表示每个时间步agent执行的动作下标位置.
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert an OG-MARL SMAC .vlt dataset to OffPyMARL .h5 shards."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        required=True,
        help="Directory containing <map-name>.vlt, for example .../og_marl/smac_v1.",
    )
    parser.add_argument("--map-name", required=True, help="SMAC map name, for example 3m.")
    parser.add_argument(
        "--og-quality",
        choices=tuple(QUALITY_MAP),
        default="Good",
        help="OG-MARL Vault UID.",
    )
    parser.add_argument(
        "--episodes-per-file",
        type=int,
        default=10000,
        help="Maximum number of episodes in each part_<n>.h5 shard.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Final output directory. Defaults to "
            "dataset/sc2/<map-name>/<mapped-quality> under the repository root."
        ),
    )
    parser.add_argument(
        "--env-config",
        type=Path,
        default=DEFAULT_ENV_CONFIG,
        help="SMAC environment YAML used to validate the dataset schema.",
    )
    return parser


def _load_runtime_dependencies():
    import numpy as np

    if int(np.__version__.split(".", maxsplit=1)[0]) >= 2:
        raise RuntimeError(
            "OG-MARL conversion requires NumPy 1.x, but the active environment "
            f"has numpy=={np.__version__}. Use the dedicated conversion "
            "environment described in README.md."
        )

    try:
        import torch as th
        import yaml
        from flashbax.vault import Vault
        from tqdm import tqdm

        from components.episode_buffer import EpisodeBatch
        from components.offline_buffer import DataSaver
        from components.transforms import OneHot
        from envs import REGISTRY as env_REGISTRY
    except Exception as exc:
        raise RuntimeError(
            "Failed to import conversion dependencies. Clone the offpymarl Conda "
            "environment and install requirements-transform.txt as described in "
            "README.md. The conversion stack is intentionally isolated from the "
            "legacy training dependencies."
        ) from exc

    return SN(
        np=np,
        th=th,
        yaml=yaml,
        Vault=Vault,
        tqdm=tqdm,
        EpisodeBatch=EpisodeBatch,
        DataSaver=DataSaver,
        OneHot=OneHot,
        env_REGISTRY=env_REGISTRY,
    )


def _shape_tuple(shape) -> tuple[int, ...]:
    return (shape,) if isinstance(shape, int) else tuple(shape)


def _prepare_experience(experience, np):
    required = ("actions", "observations", "rewards", "terminals", "truncations")
    missing = [key for key in required if key not in experience]
    if missing:
        raise KeyError(f"Vault is missing required fields: {missing}")

    infos = experience.get("infos")
    if infos is None or "legals" not in infos or "state" not in infos:
        raise KeyError("Vault must contain infos.legals and infos.state")

    arrays = {
        "actions": np.asarray(experience["actions"]),
        "observations": np.asarray(experience["observations"]),
        "rewards": np.asarray(experience["rewards"]),
        "terminals": np.asarray(experience["terminals"]),
        "truncations": np.asarray(experience["truncations"]),
        "avail_actions": np.asarray(infos["legals"]),
        "states": np.asarray(infos["state"]),
    }

    observations = arrays["observations"]
    if observations.ndim != 4:
        raise ValueError(
            "observations must have shape (B, T, n_agents, obs_dim); "
            f"received {observations.shape}"
        )

    batch_size, time_size, n_agents, _ = observations.shape
    if batch_size != 1:
        raise ValueError(
            "This converter follows the original notebook and currently requires "
            f"Vault batch size 1; received {batch_size}"
        )

    expected_agent_shape = (batch_size, time_size, n_agents)
    for key in ("actions", "rewards", "terminals", "truncations"):
        if arrays[key].shape != expected_agent_shape:
            raise ValueError(
                f"{key} must have shape {expected_agent_shape}; received {arrays[key].shape}. "
                "Continuous action datasets are not supported."
            )

    if not np.issubdtype(arrays["actions"].dtype, np.integer):
        raise TypeError(
            f"actions must contain discrete integer indices; received {arrays['actions'].dtype}"
        )

    if arrays["avail_actions"].shape[:3] != expected_agent_shape:
        raise ValueError(
            "infos.legals must have shape (B, T, n_agents, n_actions); "
            f"received {arrays['avail_actions'].shape}"
        )
    if arrays["states"].shape[:2] != (batch_size, time_size):
        raise ValueError(
            "infos.state must have shape (B, T, state_dim); "
            f"received {arrays['states'].shape}"
        )

    done = np.logical_or(arrays["terminals"], arrays["truncations"])
    if not np.all(done == done[..., :1]):
        raise ValueError("All agents must terminate or truncate synchronously")
    if not np.allclose(arrays["rewards"], arrays["rewards"][..., :1]):
        raise ValueError("OffPyMARL expects one team reward, but agent rewards differ")

    episode_ends = np.flatnonzero(done[0, :, 0])
    if episode_ends.size == 0:
        raise ValueError("Vault does not contain a completed episode")

    episode_lengths = np.diff(np.concatenate((np.asarray([-1]), episode_ends)))
    if np.any(episode_lengths < 2):
        raise ValueError(
            "The corrected termination strategy requires every episode to contain "
            "at least two stored timesteps"
        )

    arrays["terminated"] = done[..., 0].astype(np.uint8)
    arrays["rewards"] = arrays["rewards"][..., 0]
    return arrays, episode_ends, episode_lengths


def _load_env_info(args, runtime):
    with args.env_config.open("r", encoding="utf-8") as file:
        env_config = runtime.yaml.safe_load(file)

    env_args = SN(**env_config)
    env_args.env_args["map_name"] = args.map_name
    env = runtime.env_REGISTRY[env_args.env](**env_args.env_args)
    try:
        return env.get_env_info()
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()


def _validate_against_env(arrays, episode_lengths, env_info):
    observations = arrays["observations"]
    states = arrays["states"]
    actions = arrays["actions"]
    avail_actions = arrays["avail_actions"]

    checks = {
        "n_agents": (observations.shape[2], env_info["n_agents"]),
        "obs_shape": (observations.shape[3:], _shape_tuple(env_info["obs_shape"])),
        "state_shape": (states.shape[2:], _shape_tuple(env_info["state_shape"])),
        "n_actions": (avail_actions.shape[3], env_info["n_actions"]),
    }
    mismatches = [
        f"{name}: dataset={actual}, env={expected}"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    if mismatches:
        raise ValueError("Dataset/environment schema mismatch: " + "; ".join(mismatches))

    if actions.min() < 0 or actions.max() >= env_info["n_actions"]:
        raise ValueError(
            f"Action indices must be in [0, {env_info['n_actions'] - 1}]"
        )
    if episode_lengths.max() > env_info["episode_limit"]:
        raise ValueError(
            f"Longest episode has {episode_lengths.max()} transitions, exceeding "
            f"environment limit {env_info['episode_limit']}"
        )


def convert(args) -> Path:
    if args.episodes_per_file <= 0:
        raise ValueError("--episodes-per-file must be positive")
    if not args.dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset directory does not exist: {args.dataset_path}")
    if not args.env_config.is_file():
        raise FileNotFoundError(f"Environment config does not exist: {args.env_config}")

    vault_path = args.dataset_path / f"{args.map_name}.vlt" / args.og_quality
    if not vault_path.is_dir():
        raise FileNotFoundError(f"Vault UID directory does not exist: {vault_path}")

    mapped_quality = QUALITY_MAP[args.og_quality]
    output_dir = args.output_dir or (
        PROJECT_ROOT / "dataset" / "sc2" / args.map_name / mapped_quality
    )
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Output directory already exists: {output_dir}. Choose a new --output-dir "
            "to avoid overwriting an existing dataset."
        )

    runtime = _load_runtime_dependencies()
    vault = runtime.Vault(
        rel_dir=str(args.dataset_path),
        vault_name=f"{args.map_name}.vlt",
        vault_uid=args.og_quality,
    )
    arrays, episode_ends, episode_lengths = _prepare_experience(
        vault.read().experience, runtime.np
    )
    env_info = _load_env_info(args, runtime)
    _validate_against_env(arrays, episode_lengths, env_info)

    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": runtime.th.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": runtime.th.int,
        },
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": runtime.th.uint8},
        "corrected_terminated": {"vshape": (1,), "dtype": runtime.th.uint8},
    }
    groups = {"agents": env_info["n_agents"]}
    preprocess = {
        "actions": (
            "actions_onehot",
            [runtime.OneHot(out_dim=env_info["n_actions"])],
        )
    }

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary_dir = Path(
        tempfile.mkdtemp(prefix=f".{mapped_quality}-h5-", dir=output_dir.parent)
    )
    saver = runtime.DataSaver(
        str(temporary_dir), logger=None, max_size=args.episodes_per_file
    )

    start_idx = 0
    try:
        for end_idx in runtime.tqdm(episode_ends, desc="Converting episodes"):
            end_idx = int(end_idx)
            episode_slice = slice(start_idx, end_idx + 1)
            episode_length = end_idx - start_idx + 1
            time_slice = slice(0, episode_length)

            batch = runtime.EpisodeBatch(
                scheme,
                groups,
                batch_size=1,
                max_seq_length=env_info["episode_limit"] + 1,
                preprocess=preprocess,
                device="cpu",
            )
            terminated = arrays["terminated"][:, episode_slice]
            corrected_terminated = terminated.copy()
            corrected_terminated[0, -2] = 1

            batch.update(
                {
                    "state": arrays["states"][:, episode_slice],
                    "obs": arrays["observations"][:, episode_slice],
                    "actions": arrays["actions"][:, episode_slice, :, None],
                    "avail_actions": arrays["avail_actions"][:, episode_slice],
                    "reward": arrays["rewards"][:, episode_slice, None],
                    "terminated": terminated,
                    "corrected_terminated": corrected_terminated,
                },
                ts=time_slice,
            )
            saver.append(
                {
                    key: batch[key].clone().cpu()
                    for key in batch.data.transition_data.keys()
                }
            )
            start_idx = end_idx + 1

        saver.close()
        dropped_tail = arrays["observations"].shape[1] - start_idx
        manifest = {
            "source": "og_marl_vault",
            "source_path": str(vault_path),
            "map_name": args.map_name,
            "vault_uid": args.og_quality,
            "offpymarl_quality": mapped_quality,
            "n_episodes": int(len(episode_ends)),
            "n_source_timesteps": int(arrays["observations"].shape[1]),
            "n_converted_timesteps": int(episode_lengths.sum()),
            "dropped_incomplete_tail_timesteps": int(dropped_tail),
            "episode_length_min": int(episode_lengths.min()),
            "episode_length_max": int(episode_lengths.max()),
            "episode_limit": int(env_info["episode_limit"]),
            "n_agents": int(env_info["n_agents"]),
            "n_actions": int(env_info["n_actions"]),
            "obs_shape": env_info["obs_shape"],
            "state_shape": env_info["state_shape"],
            "termination_policy": (
                "terminated = terminals OR truncations; corrected_terminated also "
                "marks the penultimate stored timestep to avoid bootstrapping from a "
                "missing final next-state frame"
            ),
        }
        (temporary_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2), encoding="utf-8"
        )
        temporary_dir.rename(output_dir)
    except Exception:
        shutil.rmtree(temporary_dir, ignore_errors=True)
        raise

    print(f"Converted {len(episode_ends)} episodes to {output_dir}")
    return output_dir


def main() -> None:
    args = build_parser().parse_args()
    convert(args)


if __name__ == "__main__":
    main()
