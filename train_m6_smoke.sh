#!/bin/bash
#
# train_m6_smoke.sh —— M6 路线 B（只训 LoRA、不做 conformal）的单卡冒烟 / 正式训练脚本。
#
# 与 train_shell.sh（M1–M4）的对应关系：
#   - 同一个 python 解释器与同一条 LD_LIBRARY_PATH（否则 torch 报 Error 803、cuda 不可用，训练会悄悄落到 CPU）。
#   - 同一份切分目录（pipelines.time_splits 落的 train/valid.parquet + split_manifest.json）。
#   - M6 不走 train_script.py / accelerate launch：自带入口 forecast_backbones.m6.train，单进程单卡。
#   - 配置同样只改运行时副本，不改仓库里的 yaml。
#
# 用法：
#   bash train_m6_smoke.sh                                   # GPU 0，只训 2 步（冒烟）
#   bash train_m6_smoke.sh --gpu 3                           # 换一张卡
#   bash train_m6_smoke.sh --max_steps 0 --run_id lora_v1    # 0 = 不限步数，按配置的 epochs_max 正式训练
#   bash train_m6_smoke.sh --splits_dir /data/xxx/splits --checkpoint /data/xxx/timesfm-3.0-pytorch
#
# 注意：开训前会先在全部验证块上打一次零样本基线分，这一步不受 --max_steps 限制，期间没有输出；
#       用 `watch -n 2 nvidia-smi` 看 GPU 利用率确认它在跑。
#
set -u

# 仓库根目录：脚本放在根目录或 train_shell/ 下都能找到
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$_HERE/forecast_backbones" ]; then BASE_DIR="$_HERE"; else BASE_DIR="$(cd "$_HERE/.." && pwd)"; fi

SPLITS_DIR="${BASE_DIR}/data/splits/m1_m4_v1"
CONFIG="${BASE_DIR}/forecast_backbones/m6/config/lora_no_conformal.yaml"
RUNS_BASE="${BASE_DIR}/runs"
RUN_ID="lora_smoke"
GPU=0
MAX_STEPS=2                         # 0 = 不限（正式训练）
PRECISION="fp32"                    # V100 没有原生 bf16；A100/H100 上可传 --precision bf16
BATCH_SIZE=""                       # 留空 = 用配置里的 finetune.schedule.batch_size
CHECKPOINT=""                       # 留空 = 用配置里的 timesfm3.checkpoint
ALLOW_CPU=0
VENV_PY="/home/ma-user/ts3/bin/python"
PRE_CUDA_LD="/usr/local/cuda-12.1/lib64:/usr/lib/x86_64-linux-gnu"

usage() {
  cat <<'EOF'
用法:
  bash train_m6_smoke.sh [选项]

选项:
  --splits_dir <dir>     pipelines.time_splits 落的切分目录（默认 <仓库>/data/splits/m1_m4_v1）
  --config <yaml>        M6 配置（默认 forecast_backbones/m6/config/lora_no_conformal.yaml）
  --run_id <name>        产物目录名：<runs_base>/m6/<run_id>（默认 lora_smoke）
  --runs_base <dir>      产物根目录（默认 <仓库>/runs）
  --gpu <id>             使用哪一张卡（默认 0）；M6 只用单卡
  --max_steps <n>        训练步数上限（默认 2 = 冒烟）；0 = 不限，按配置的 epochs_max 训练
  --precision <fp32|bf16>  训练精度（默认 fp32；V100 不支持原生 bf16）
  --batch_size <n>       覆盖 finetune.schedule.batch_size
  --checkpoint <dir>     覆盖 timesfm3.checkpoint（TimesFM-3 权重目录）
  --venv-python <path>   python 解释器（默认 /home/ma-user/ts3/bin/python）
  --allow-cpu            检测不到 GPU 时也继续（很慢，仅排查用）
  -h|--help              显示本帮助
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --splits_dir)   SPLITS_DIR="$2";  shift 2 ;;
    --config)       CONFIG="$2";      shift 2 ;;
    --run_id)       RUN_ID="$2";      shift 2 ;;
    --runs_base)    RUNS_BASE="$2";   shift 2 ;;
    --gpu)          GPU="$2";         shift 2 ;;
    --max_steps)    MAX_STEPS="$2";   shift 2 ;;
    --precision)    PRECISION="$2";   shift 2 ;;
    --batch_size)   BATCH_SIZE="$2";  shift 2 ;;
    --checkpoint)   CHECKPOINT="$2";  shift 2 ;;
    --venv-python)  VENV_PY="$2";     shift 2 ;;
    --allow-cpu)    ALLOW_CPU=1;      shift   ;;
    -h|--help)      usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; usage; exit 1 ;;
  esac
done

# ============================================================
# 前置校验
# ============================================================
[ -f "$VENV_PY" ] || { echo "❌ python 解释器不存在: $VENV_PY" >&2; exit 1; }
[ -f "$CONFIG" ]  || { echo "❌ 找不到 M6 配置: $CONFIG" >&2; exit 1; }
for f in train.parquet valid.parquet split_manifest.json; do
  [ -f "$SPLITS_DIR/$f" ] || { echo "❌ 切分目录缺 $f: $SPLITS_DIR" >&2; exit 1; }
done
case "$PRECISION" in fp32|bf16) ;; *) echo "❌ --precision 只能是 fp32 或 bf16" >&2; exit 1 ;; esac

export LD_LIBRARY_PATH="${PRE_CUDA_LD}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="$GPU"
cd "$BASE_DIR" || exit 1

RUN_DIR="$RUNS_BASE/m6/$RUN_ID"
LOG_FILE="$BASE_DIR/logger/m6_${RUN_ID}_train.log"
TMP_CFG="$RUNS_BASE/.m6_cfg_${RUN_ID}.yaml"
mkdir -p "$RUNS_BASE" "$BASE_DIR/logger"

# GPU 是否可用（在套好 LD_LIBRARY_PATH 之后查）
CUDA_INFO="$("$VENV_PY" -c "import torch; ok = torch.cuda.is_available(); print(('cuda ' + torch.cuda.get_device_name(0)) if ok else 'cpu')" 2>&1 | tail -1)"
if [ "${CUDA_INFO%% *}" != "cuda" ]; then
  if [ "$ALLOW_CPU" -eq 1 ]; then
    echo "⚠️  torch 看不到 GPU（${CUDA_INFO}），按 --allow-cpu 继续，会很慢"
  else
    echo "❌ torch 看不到 GPU（${CUDA_INFO}）。检查 --gpu、LD_LIBRARY_PATH=${PRE_CUDA_LD}、--venv-python；确要用 CPU 请加 --allow-cpu" >&2
    exit 1
  fi
fi

# 运行时配置副本：只改精度 / batch_size / checkpoint，不动仓库里的 yaml
"$VENV_PY" - "$CONFIG" "$TMP_CFG" "$PRECISION" "$BATCH_SIZE" "$CHECKPOINT" <<'PYEOF'
import sys, yaml
src, dst, precision, batch_size, checkpoint = sys.argv[1:6]
cfg = yaml.safe_load(open(src, encoding="utf-8")) or {}
sch = cfg.setdefault("finetune", {}).setdefault("schedule", {})
sch["precision"] = precision
if batch_size:
    sch["batch_size"] = int(batch_size)
if checkpoint:
    cfg.setdefault("timesfm3", {})["checkpoint"] = checkpoint
yaml.safe_dump(cfg, open(dst, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
import os
ck = cfg.get("timesfm3", {}).get("checkpoint")
meta = cfg.get("dataset", {}).get("station_meta")
print(f"CHECKPOINT={ck}"); print(f"STATION_META={meta}")
sys.exit(0 if (ck and os.path.exists(ck)) and (meta and os.path.exists(meta)) else 3)
PYEOF
RC=$?
if [ $RC -eq 3 ]; then
  echo "❌ 配置里的 timesfm3.checkpoint 或 dataset.station_meta 路径不存在（见上两行）；权重路径可用 --checkpoint 覆盖" >&2
  rm -f "$TMP_CFG"; exit 1
elif [ $RC -ne 0 ]; then
  echo "❌ 生成运行时配置失败" >&2; rm -f "$TMP_CFG"; exit 1
fi

echo "=============================================================="
echo "M6 LoRA 训练（不做 conformal）"
echo "BASE_DIR   : $BASE_DIR"
echo "VENV_PY    : $VENV_PY"
echo "SPLITS_DIR : $SPLITS_DIR"
echo "CONFIG     : $CONFIG  (运行时副本: $TMP_CFG)"
echo "RUN_DIR    : $RUN_DIR"
echo "设备       : $CUDA_INFO  (CUDA_VISIBLE_DEVICES=$GPU)   精度: $PRECISION"
if [ "$MAX_STEPS" -gt 0 ]; then echo "max_steps  : ${MAX_STEPS}（冒烟）"; else echo "max_steps  : 不限（正式训练）"; fi
echo "日志       : $LOG_FILE"
echo "=============================================================="

rm -rf "$RUN_DIR"; mkdir -p "$RUN_DIR"

cmd=(
  "$VENV_PY" -u -m forecast_backbones.m6.train
  --config "$TMP_CFG"
  --splits-dir "$SPLITS_DIR"
  --out "$RUN_DIR"
)
[ "$MAX_STEPS" -gt 0 ] && cmd+=(--max-steps "$MAX_STEPS")

echo "CMD: ${cmd[*]}"
"${cmd[@]}" 2>&1 | tee "$LOG_FILE"
rc=${PIPESTATUS[0]}
rm -f "$TMP_CFG"

if [ $rc -ne 0 ]; then
  echo "❌ M6 训练失败 (exit=$rc)，日志: $LOG_FILE" >&2
  exit 1
fi
for f in lora_weights.pt station_routing.json train_log.json; do
  [ -f "$RUN_DIR/$f" ] || { echo "❌ 训练结束但缺少 $RUN_DIR/$f" >&2; exit 1; }
done
echo "✅ M6 完成: $RUN_DIR/lora_weights.pt"
echo "   对比表: $RUN_DIR/reports/ladder_lora.csv"
