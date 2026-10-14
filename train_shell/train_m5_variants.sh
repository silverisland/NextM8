#!/bin/bash
#
# train_m5_variants.sh —— 批量训练 M5 的多个变体：每个变体一张卡，卡数不够时排队；跑完汇总成一张表（collect_runs）。
#
# 用法（基线）：
#   bash train_m5_variants.sh --splits_dir data/splits/m8_v1 --variants z0_zeroshot --gpus 0 --batch_tag m5_base
#   bash train_m5_variants.sh --splits_dir data/splits/m8_v1 --variants l1_lora_r8 --seeds 0,1,2 --gpus 1,2,3 --batch_tag m5_base
#   冒烟：... --variants z0_zeroshot,l1_lora_r8 --max_steps 20 --eval_every 10 --gpus 0 --batch_tag smoke
# d2_pstar_point / g2_cs_ceiling 只动决策层，不用训：对已训好的 run 用 eval_m5.sh --decision_yaml
# 基线的 z0 与 l1 可同时开（共用 --batch_tag）：暂存目录按调用隔离（.runs_$$）。
# 产物：runs/m5/<batch_tag>/<变体>[_s<seed>]/…，汇总 runs/m5/<batch_tag>/variants_summary.csv
set -u
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$_HERE/forecast_backbones" ]; then BASE_DIR="$_HERE"; else BASE_DIR="$(cd "$_HERE/.." && pwd)"; fi
SPLITS_DIR="${BASE_DIR}/data/splits/m8_v1"
TRAIN_PARQUET=""       # 给了就是全量训练（透传 --train_parquet，不用 --splits_dir）
VARIANT_DIR="${BASE_DIR}/forecast_backbones/m5/config/variants"
VARIANTS=""            # 逗号分隔的变体名（不带 .yaml）；留空 = 目录下全部
GPUS="auto"
MAX_STEPS=""
SEEDS=""             # 逗号分隔；留空 = 用配置里的 run.seeds[0]
BATCH_TAG="variants_$(date +%Y%m%d_%H%M%S)"
RUNS_BASE="${BASE_DIR}/runs"
VENV_PY="/home/ma-user/ts3/bin/python"
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --splits_dir) SPLITS_DIR="$2"; shift 2 ;;
    --train_parquet) TRAIN_PARQUET="$2"; shift 2 ;;
    --variants)   VARIANTS="$2";   shift 2 ;;
    --variant_dir) VARIANT_DIR="$2"; shift 2 ;;
    --gpus)       GPUS="$2";       shift 2 ;;
    --max_steps)  MAX_STEPS="$2";  shift 2 ;;
    --seeds)      SEEDS="$2";      shift 2 ;;
    --batch_tag)  BATCH_TAG="$2";  shift 2 ;;
    --runs_base)  RUNS_BASE="$2";  shift 2 ;;
    --venv-python) VENV_PY="$2";   shift 2 ;;
    *) EXTRA+=("$1"); shift ;;                      # 其余参数原样传给 train_m5.sh（如 --precision、--batch_size）
  esac
done
if [ "$GPUS" = "auto" ]; then
  GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ' | paste -sd, -)"
fi
[ -n "$GPUS" ] || { echo "❌ 没检测到 GPU；用 --gpus 指定" >&2; exit 1; }
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
if [ -z "$VARIANTS" ]; then
  FILES=( $(ls "$VARIANT_DIR"/*.yaml | sort) )
else
  FILES=(); IFS=',' read -r -a NAMES <<< "$VARIANTS"
  for n in "${NAMES[@]}"; do f="$VARIANT_DIR/${n%.yaml}.yaml"; [ -f "$f" ] || { echo "❌ 变体不存在: $f" >&2; exit 1; }; FILES+=("$f"); done
fi
OUT_BASE="$RUNS_BASE/m5/$BATCH_TAG"; mkdir -p "${OUT_BASE}"
# 暂存目录按调用隔离（$$ = 本脚本 PID）：多个调用可共用同一个 --batch_tag 并发跑，互不删对方的在途产物
STAGE="$OUT_BASE/.runs_$$"
# 任务 = 变体 × 种子；种子留空时一个变体一个任务
JOBS=(); JOB_SEEDS=()
if [ -n "$SEEDS" ]; then IFS=',' read -r -a SEED_LIST <<< "$SEEDS"; else SEED_LIST=(""); fi
for f in "${FILES[@]}"; do for sd in "${SEED_LIST[@]}"; do JOBS+=("$f"); JOB_SEEDS+=("$sd"); done; done
echo "变体 ${#FILES[@]} 个 × 种子 ${#SEED_LIST[@]} = 任务 ${#JOBS[@]} 个，GPU ${#GPU_LIST[@]} 张：${GPUS}；产物 ${OUT_BASE}"
# 每张卡一条队列：第 i 个任务分给第 i mod G 张卡，同一张卡上的任务顺序执行
G=${#GPU_LIST[@]}
pids=()
for gi in "${!GPU_LIST[@]}"; do
  gpu="${GPU_LIST[$gi]}"
  (
    for ((i=gi; i<${#JOBS[@]}; i+=G)); do
      f="${JOBS[$i]}"; sd="${JOB_SEEDS[$i]}"; name="$(basename "$f" .yaml)${sd:+_s$sd}"
      echo "[gpu $gpu] ▶ $name"
      if [ -n "$TRAIN_PARQUET" ]; then src=(--train_parquet "$TRAIN_PARQUET"); else src=(--splits_dir "$SPLITS_DIR"); fi
      args=("${src[@]}" --config "$f" --gpus "$gpu" --runs_base "$STAGE" --run_id "$name" --tag "$name"
            --port $((29700 + gi)) --venv-python "$VENV_PY" --overwrite)
      [ -n "$sd" ] && args+=(--seed "$sd")
      [ -n "$MAX_STEPS" ] && args+=(--max_steps "$MAX_STEPS")
      bash "$_HERE/train_m5.sh" "${args[@]}" "${EXTRA[@]}" > "$OUT_BASE/${name}.log" 2>&1 \
        && echo "[gpu $gpu] ✅ $name" || echo "[gpu $gpu] ❌ ${name}（看 $OUT_BASE/${name}.log）"
      # 把产物目录提到 OUT_BASE/<name>
      [ -d "$STAGE/m5/$name" ] && { rm -rf "$OUT_BASE/$name"; mv "$STAGE/m5/$name" "$OUT_BASE/$name"; }
    done
  ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
rm -rf "$STAGE"
cd "$BASE_DIR" && "$VENV_PY" -m forecast_backbones.m5.collect_runs "${OUT_BASE}" --out "$OUT_BASE/variants_summary.csv"
