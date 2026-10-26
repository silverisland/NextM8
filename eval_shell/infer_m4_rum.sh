#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# 与训练保持一致；不可用时明确报错，不自动切换 Python 环境。
PYTHON_BIN="${PYTHON:-/home/ma-user/ts3/bin/python}"
ARGS=()
while [ $# -gt 0 ]; do
  case "$1" in
    --venv-python)
      if [ $# -lt 2 ] || [ -z "$2" ] || [[ "$2" == --* ]]; then
        echo "缺少参数值: $1" >&2; exit 2
      fi
      PYTHON_BIN="$2"; shift 2 ;;
    *) ARGS+=("$1"); shift ;;
  esac
done
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 解释器不可执行: ${PYTHON_BIN}（可用 --venv-python 或 PYTHON 显式指定；不自动回退）" >&2
  exit 1
fi
# HF datasets 的 Arrow 缓存默认放在 work 下的 NFS 路径，避免写入根文件系统触发磁盘配额。
# 外部已设 HF_DATASETS_CACHE 时尊重外部值
if [ -z "${HF_DATASETS_CACHE:-}" ]; then
  export HF_DATASETS_CACHE="${HOME:-/home/ma-user}/work/hanjs/hf_cache/datasets"
  mkdir -p "$HF_DATASETS_CACHE"
fi

echo "Python=$PYTHON_BIN"
exec "$PYTHON_BIN" -m forecast_backbones.m4_rum.infer "${ARGS[@]}"
