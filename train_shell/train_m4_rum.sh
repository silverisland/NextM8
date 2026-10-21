#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

# 与 M8 使用同一服务器环境；本地没有该环境时使用 PATH 中的 python。
PYTHON_BIN="${PYTHON:-/home/ma-user/ts3/bin/python}"
if [ -z "${PYTHON:-}" ] && [ ! -x "$PYTHON_BIN" ]; then PYTHON_BIN=python; fi
PRE_CUDA_LD="/usr/local/cuda-12.1/lib64:/usr/lib/x86_64-linux-gnu"
GPUS=auto
PORT=29700
CPU_THREADS="${OMP_NUM_THREADS:-}"
ALLOW_CPU=0
ARGS=()

usage() {
  cat <<'USAGE'
用法: bash train_shell/train_m4_rum.sh [启动选项] [训练选项]
  --gpus <csv|auto>       指定 GPU，例如 0,1；默认 auto 使用全部当前可见卡
  --gpu <id>              单卡，同 --gpus
  --venv-python <path>    Python 解释器（默认 $PYTHON、M8 服务器环境或 python）
  --port <n>              单机多卡通信端口（默认 29700）
  --cpu_threads <n>       每进程 CPU 线程数（默认 $OMP_NUM_THREADS 或 CPU 数 / 进程数）
  --allow-cpu             auto 未检测到 GPU 时允许 CPU 训练
训练选项原样传给 forecast_backbones.m4_rum.train，例如：
  --config <yaml> --splits-dir <dir> --out <dir>
  --train-parquet <path...> --valid-parquet <path...> --out <dir>
  --station-meta <csv>    覆盖站表并自动识别标准 station_meta.csv / 线上 info.csv 表头
  --max-steps <n> --save-every <n>
训练/验证路径支持目录、文件或加引号的通配符；只给训练路径时全量训练、不验证。
同时兼容 --splits_dir、--train_parquet、--valid_parquet、--max_steps 等下划线写法。
batch_size、precision、lr 等训练超参数在 YAML 的 train.schedule 下配置。
CUDA 库路径前缀与 M8 一致，固定为 /usr/local/cuda-12.1/lib64:/usr/lib/x86_64-linux-gnu。
USAGE
}

while [ $# -gt 0 ]; do
  case "$1" in
    --gpus|--gpu|--venv-python|--port|--cpu_threads|--cpu-threads)
      if [ $# -lt 2 ] || [ -z "$2" ] || [[ "$2" == --* ]]; then
        echo "缺少参数值: $1" >&2; exit 2
      fi
      case "$1" in
        --gpus|--gpu) GPUS="$2" ;;
        --venv-python) PYTHON_BIN="$2" ;;
        --port) PORT="$2" ;;
        *) CPU_THREADS="$2" ;;
      esac
      shift 2 ;;
    --allow-cpu) ALLOW_CPU=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --splits_dir|--train_parquet|--valid_parquet|--max_steps|--save_every|--hint_col|--station_meta)
      ARGS+=("${1//_/-}"); shift ;;
    *) ARGS+=("$1"); shift ;;
  esac
done

if ! [[ "$PORT" =~ ^[0-9]{1,5}$ ]] || (( 10#$PORT < 1 || 10#$PORT > 65535 )); then
  echo "--port 必须在 1–65535 之间" >&2; exit 2
fi
if [ -n "$CPU_THREADS" ] && ! [[ "$CPU_THREADS" =~ ^[1-9][0-9]*$ ]]; then
  echo "--cpu_threads 必须是正整数" >&2; exit 2
fi
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python 解释器不可执行: $PYTHON_BIN（可用 --venv-python 指定）" >&2; exit 1
fi
if [ "$GPUS" != auto ]; then
  if ! [[ "$GPUS" =~ ^(0|[1-9][0-9]*)(,(0|[1-9][0-9]*))*$ ]]; then
    echo "--gpus 必须是 auto 或不重复的 GPU 编号，如 0,1" >&2; exit 2
  fi
  IFS=',' read -r -a GPU_LIST <<< "$GPUS"
  SEEN=,
  for gpu in "${GPU_LIST[@]}"; do
    if [[ "$SEEN" == *",$gpu,"* ]]; then
      echo "--gpus 包含重复编号: $gpu" >&2; exit 2
    fi
    SEEN+="$gpu,"
  done
  export CUDA_VISIBLE_DEVICES="$GPUS"
fi

# 动态库路径必须在首次 import torch 前配置，探卡及所有训练子进程共用此环境。
export LD_LIBRARY_PATH="${PRE_CUDA_LD}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# 在设置可见卡之后，由训练所用解释器检测；auto 尊重外部 CUDA_VISIBLE_DEVICES。
NP="$("$PYTHON_BIN" -c 'import torch; print(torch.cuda.device_count())')"
if [ "$GPUS" != auto ] && [ "$NP" -ne "${#GPU_LIST[@]}" ]; then
  echo "指定了 ${#GPU_LIST[@]} 张 GPU，但 PyTorch 仅检测到 $NP 张，请检查编号和 CUDA 环境" >&2
  exit 1
fi
if [ "$NP" -eq 0 ] && [ "$ALLOW_CPU" -ne 1 ]; then
  echo "未检测到 GPU；如需 CPU 训练，请加 --allow-cpu" >&2; exit 1
fi
if [ -z "$CPU_THREADS" ]; then
  NPROC=$(( NP > 0 ? NP : 1 ))
  NCPU="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo "$NPROC")"
  CPU_THREADS=$(( NCPU / NPROC ))
  [ "$CPU_THREADS" -ge 1 ] || CPU_THREADS=1
fi
export OMP_NUM_THREADS="$CPU_THREADS"
echo "Python=$PYTHON_BIN LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}"
echo "M4-Rum: GPUs=$NP CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-auto} CPU_THREADS=$CPU_THREADS"
if [ "$NP" -gt 1 ]; then
  exec "$PYTHON_BIN" -m torch.distributed.run --nnodes=1 --nproc_per_node="$NP" \
    --master_addr=127.0.0.1 --master_port="$PORT" \
    -m forecast_backbones.m4_rum.train "${ARGS[@]}"
else
  exec "$PYTHON_BIN" -m forecast_backbones.m4_rum.train "${ARGS[@]}"
fi
