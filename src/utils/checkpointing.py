import os

import numpy as np
import torch as th


TRAINING_STATE_FILE = "training_state.th"

_LEARNER_COUNTERS = (
    "training_steps",
    "critic_training_steps",
    "last_target_update_step",
    "last_target_update_episode",
    "log_stats_t",
    "behaviour_training_steps",
    "behaviour_log_stats_t",
)

_NORMALISERS = ("ret_ms", "rew_ms")


def select_checkpoint(checkpoint_root, load_step=0):
    if not os.path.isdir(checkpoint_root):
        raise ValueError("Checkpoint directory {} doesn't exist".format(checkpoint_root))

    timesteps = [
        int(name)
        for name in os.listdir(checkpoint_root)
        if name.isdigit() and os.path.isdir(os.path.join(checkpoint_root, name))
    ]
    if not timesteps:
        raise ValueError(
            "Checkpoint directory {} contains no numeric step directories".format(
                checkpoint_root
            )
        )

    if load_step == 0:
        timestep = max(timesteps)
    else:
        timestep = min(timesteps, key=lambda step: abs(step - load_step))

    return os.path.join(checkpoint_root, str(timestep)), timestep


def save_training_state(checkpoint_path, t_env, episode, learner):
    learner_counters = {
        name: getattr(learner, name)
        for name in _LEARNER_COUNTERS
        if hasattr(learner, name)
    }
    normalisers = {}
    for name in _NORMALISERS:
        normaliser = getattr(learner, name, None)
        if normaliser is not None:
            normalisers[name] = {
                "mean": normaliser.mean.detach().cpu(),
                "var": normaliser.var.detach().cpu(),
                "count": normaliser.count,
            }

    state = {
        "version": 1,
        "t_env": int(t_env),
        "episode": int(episode),
        "learner_counters": learner_counters,
        "normalisers": normalisers,
        "numpy_rng_state": np.random.get_state(),
        "torch_rng_state": th.get_rng_state(),
    }
    if th.cuda.is_available():
        state["cuda_rng_state"] = th.cuda.get_rng_state()

    os.makedirs(checkpoint_path, exist_ok=True)
    state_path = os.path.join(checkpoint_path, TRAINING_STATE_FILE)
    temporary_path = state_path + ".tmp"
    th.save(state, temporary_path)
    os.replace(temporary_path, state_path)


def load_training_state(checkpoint_path, learner, fallback_t_env=0):
    state_path = os.path.join(checkpoint_path, TRAINING_STATE_FILE)
    if not os.path.isfile(state_path):
        return {"t_env": int(fallback_t_env), "episode": 0, "restored": False}

    try:
        state = th.load(state_path, map_location="cpu", weights_only=False)
    except TypeError:
        state = th.load(state_path, map_location="cpu")

    for name, value in state.get("learner_counters", {}).items():
        if hasattr(learner, name):
            setattr(learner, name, value)

    for name, values in state.get("normalisers", {}).items():
        normaliser = getattr(learner, name, None)
        if normaliser is not None:
            normaliser.mean = values["mean"].to(normaliser.mean.device)
            normaliser.var = values["var"].to(normaliser.var.device)
            normaliser.count = values["count"]

    if "numpy_rng_state" in state:
        np.random.set_state(state["numpy_rng_state"])
    if "torch_rng_state" in state:
        th.set_rng_state(state["torch_rng_state"])
    if th.cuda.is_available() and "cuda_rng_state" in state:
        th.cuda.set_rng_state(state["cuda_rng_state"])

    return {
        "t_env": int(state.get("t_env", fallback_t_env)),
        "episode": int(state.get("episode", 0)),
        "restored": True,
    }
