#!/usr/bin/env bash
set -Eeuo pipefail

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
SMAC_DIR="$REPO_DIR/3rdparty/smac"
PATCH_DIR="$REPO_DIR/patches/smac"
SMAC_URL="https://github.com/oxwhirl/smac.git"

# 本机已有 VOMASD 环境验证通过的 SMAC 1.0.0 commit
SMAC_COMMIT="d6aab33f76abc3849c50463a8592a84f59a5ef84"

die() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

[[ "${CONDA_DEFAULT_ENV:-}" == "offpymarl" ]] ||
    die "请先激活 offpymarl 环境"

command -v git >/dev/null || die "未找到 git"
command -v python >/dev/null || die "未找到 python"
[[ -n "${SC2PATH:-}" ]] || die "SC2PATH 未设置"

SC2_ROOT="$(realpath -e -- "$SC2PATH")" ||
    die "无效的 SC2PATH: $SC2PATH"

SC2_BIN="$(
    find "$SC2_ROOT/Versions" \
        -maxdepth 2 -type f -name SC2_x64 -executable \
        -print -quit 2>/dev/null || true
)"

[[ -n "$SC2_BIN" ]] ||
    die "$SC2_ROOT/Versions 下未找到可执行的 SC2_x64"

[[ -f "$PATCH_DIR/smac_maps.py" ]] ||
    die "缺少 patches/smac/smac_maps.py"

[[ -d "$PATCH_DIR/SMAC_Maps" ]] ||
    die "缺少 patches/smac/SMAC_Maps"

export SC2PATH="$SC2_ROOT"

mkdir -p "$REPO_DIR/3rdparty"

if [[ ! -e "$SMAC_DIR" ]]; then
    git clone "$SMAC_URL" "$SMAC_DIR"
    git -C "$SMAC_DIR" checkout --detach "$SMAC_COMMIT"

    cp -a \
        "$PATCH_DIR/." \
        "$SMAC_DIR/smac/env/starcraft2/maps/"
else
    [[ -d "$SMAC_DIR/.git" ]] ||
        die "$SMAC_DIR 已存在，但不是 Git 仓库"

    [[ "$(git -C "$SMAC_DIR" rev-parse HEAD)" == "$SMAC_COMMIT" ]] ||
        die "$SMAC_DIR 不是预期的 SMAC commit"

    cmp -s \
        "$PATCH_DIR/smac_maps.py" \
        "$SMAC_DIR/smac/env/starcraft2/maps/smac_maps.py" ||
        die "现有 SMAC 地图注册表与项目补丁不同"

    diff -qr \
        "$PATCH_DIR/SMAC_Maps" \
        "$SMAC_DIR/smac/env/starcraft2/maps/SMAC_Maps" \
        >/dev/null ||
        die "现有 SMAC 地图与项目补丁不同"
fi

SC2_MAP_DIR="$SC2_ROOT/Maps/SMAC_Maps"

if [[ -e "$SC2_MAP_DIR" ]]; then
    diff -qr "$PATCH_DIR/SMAC_Maps" "$SC2_MAP_DIR" >/dev/null ||
        die "$SC2_MAP_DIR 已存在，但与项目地图不同"
else
    mkdir -p "$SC2_ROOT/Maps"
    cp -a "$PATCH_DIR/SMAC_Maps" "$SC2_ROOT/Maps/"
fi

python -m pip install -e "$SMAC_DIR"

python - <<'PY'
from pathlib import Path
import os

from smac.env import StarCraft2Env
from smac.env.starcraft2.maps.smac_maps import get_smac_map_registry

registry = get_smac_map_registry()
map_dir = Path(os.environ["SC2PATH"]) / "Maps" / "SMAC_Maps"
map_files = list(map_dir.glob("*.SC2Map"))

assert len(registry) == 73, (
    f"预期注册 73 张地图，实际为 {len(registry)}"
)
assert len(map_files) == 73, (
    f"预期存在 73 个地图文件，实际为 {len(map_files)}"
)

env = StarCraft2Env(map_name="3m", seed=1)
try:
    observations, state = env.reset()
    print(
        "SMAC smoke test passed: "
        f"agents={len(observations)}, state_size={len(state)}"
    )
finally:
    env.close()
PY

# git clone git@github.com:oxwhirl/smac.git
# pip install -e smac/
# cp -r patches/smac/* smac/smac/env/starcraft2/maps 
# cp patches/smac/SMAC_Maps/* $SC2PATH/Maps/SMAC_Maps/