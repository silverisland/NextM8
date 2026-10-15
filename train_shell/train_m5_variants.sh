#!/bin/bash
#
# train_m5_variants.sh —— 批量训练 M5 的多个变体：自动检测 GPU、按任务数把卡平分给任务；跑完汇总成一张表（collect_runs）。
#
# 用法（基线，一条命令：z0 1 个 + l1 3 个种子 = 4 个任务；8 张卡时每个任务 2 张卡、4 个同时跑）：
#   bash train_m5_variants.sh --splits_dir data/splits/m8_v1 --variants z0_zeroshot,l1_lora_r8 --seeds 0,1,2 --batch_tag m5_base
#   冒烟：... --variants z0_zeroshot,l1_lora_r8 --max_steps 20 --eval_every 10 --batch_tag m5_smoke
#   先看计划不跑：加 --dry_run
# 分卡规则（--gpus_per_job auto）：每个任务 k = floor(卡数 / 任务数) 张卡（至少 1），再往下取到能整除全局 batch 的数；
#   卡按 k 张一组分成若干槽，任务轮流进槽，同一槽内顺序执行。1 个任务 → 全部卡；任务比卡多 → 每任务 1 张卡排队。
#   全局 batch（--batch_size，默认配置里的 16）不随卡数变：每卡 = 全局 / k，所以不同卡数跑出的结果可比。
# 零样本变体（finetune.enabled: false）与种子无关：给了 --seeds 也只跑一次，目录名不带 _s<seed>。
# 一次调用默认占满所有卡；同时开两个调用会抢卡，要并发请用 --gpus 把卡分开。
# d2_pstar_point / g2_cs_ceiling 只动决策层，不用训：对已训好的 run 用 eval_m5.sh --decision_yaml
# 产物：runs/m5/<batch_tag>/<变体>[_s<seed>]/…，汇总 runs/m5/<batch_tag>/variants_summary.csv
set -u
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$_HERE/forecast_backbones" ]; then BASE_DIR="$_HERE"; else BASE_DIR="$(cd "$_HERE/.." && pwd)"; fi
SPLITS_DIR="${BASE_DIR}/data/splits/m8_v1"
TRAIN_PARQUET=""       # 给了就是全量训练（透传 --train_parquet，不用 --splits_dir）
VARIANT_DIR="${BASE_DIR}/forecast_backbones/m5/config/variants"
VARIANTS=""            # 逗号分隔的变体名（不带 .yaml）；留空 = 目录下全部
GPUS="auto"
GPUS_PER_JOB="auto"
BATCH_SIZE=""          # 全局 batch；留空 = 配置里的值（默认 16）
MAX_STEPS=""
SEEDS=""             # 逗号分隔；留空 = 用配置里的 run.seeds[0]
BATCH_TAG="variants_$(date +%Y%m%d_%H%M%S)"
RUNS_BASE="${BASE_DIR}/runs"
VENV_PY="/home/ma-user/ts3/bin/python"
DRY_RUN=0
EXTRA=()
while [ $# -gt 0 ]; do
  case "$1" in
    --splits_dir) SPLITS_DIR="$2"; shift 2 ;;
    --train_parquet) TRAIN_PARQUET="$2"; shift 2 ;;
    --variants)   VARIANTS="$2";   shift 2 ;;
    --variant_dir) VARIANT_DIR="$2"; shift 2 ;;
    --gpus)       GPUS="$2";       shift 2 ;;
    --gpus_per_job) GPUS_PER_JOB="$2"; shift 2 ;;
    --batch_size) BATCH_SIZE="$2"; shift 2 ;;
    --max_steps)  MAX_STEPS="$2";  shift 2 ;;
    --seeds)      SEEDS="$2";      shift 2 ;;
    --batch_tag)  BATCH_TAG="$2";  shift 2 ;;
    --runs_base)  RUNS_BASE="$2";  shift 2 ;;
    --venv-python) VENV_PY="$2";   shift 2 ;;
    --dry_run)    DRY_RUN=1;       shift   ;;
    *) EXTRA+=("$1"); shift ;;                      # 其余参数原样传给 train_m5.sh（如 --precision、--eval_every）
  esac
done
if [ "$GPUS" = "auto" ]; then
  GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ' | paste -sd, -)"
fi
[ -n "$GPUS" ] || { echo "❌ 没检测到 GPU；用 --gpus 指定" >&2; exit 1; }
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
G=${#GPU_LIST[@]}
if [ -z "$VARIANTS" ]; then
  FILES=( $(ls "$VARIANT_DIR"/*.yaml | sort) )
else
  FILES=(); IFS=',' read -r -a NAMES <<< "$VARIANTS"
  for n in "${NAMES[@]}"; do f="$VARIANT_DIR/${n%.yaml}.yaml"; [ -f "$f" ] || { echo "❌ 变体不存在: $f" >&2; exit 1; }; FILES+=("$f"); done
fi
# 任务 = 变体 × 种子；种子留空时一个变体一个任务；零样本变体只跑一次（打分与种子无关）
JOBS=(); JOB_SEEDS=()
if [ -n "$SEEDS" ]; then IFS=',' read -r -a SEED_LIST <<< "$SEEDS"; else SEED_LIST=(""); fi
for f in "${FILES[@]}"; do
  if grep -Eq '^[[:space:]]+enabled:[[:space:]]*false' "$f"; then JOBS+=("$f"); JOB_SEEDS+=(""); continue; fi
  for sd in "${SEED_LIST[@]}"; do JOBS+=("$f"); JOB_SEEDS+=("$sd"); done
done
J=${#JOBS[@]}
# 每个任务几张卡：auto = floor(G / J)，至少 1，再往下取到能整除全局 batch
GB="${BATCH_SIZE:-16}"
if [ "$GPUS_PER_JOB" = "auto" ]; then
  K=$(( G / J )); [ "$K" -ge 1 ] || K=1
  while [ "$K" -gt 1 ] && [ $(( GB % K )) -ne 0 ]; do K=$(( K - 1 )); done
else
  K="$GPUS_PER_JOB"
  [ "$K" -ge 1 ] && [ "$K" -le "$G" ] || { echo "❌ --gpus_per_job=$K 不在 1..$G" >&2; exit 1; }
  [ $(( GB % K )) -eq 0 ] || { echo "❌ 全局 batch $GB 不能被 --gpus_per_job=$K 整除" >&2; exit 1; }
fi
S=$(( G / K )); [ "$S" -le "$J" ] || S=$J          # 槽数：k 张卡一组；任务少于槽数时多出的卡空着
slot_gpus() { local s=$1 out="" c; for ((c=s*K; c<(s+1)*K; c++)); do out="${out:+$out,}${GPU_LIST[$c]}"; done; echo "$out"; }
job_name() { local f="${JOBS[$1]}" sd="${JOB_SEEDS[$1]}"; echo "$(basename "$f" .yaml)${sd:+_s$sd}"; }
echo "变体 ${#FILES[@]} 个 → 任务 $J 个；GPU ${G} 张（${GPUS}）→ 每个任务 $K 张卡、$S 个槽同时跑；全局 batch ${GB}（每卡 $(( GB / K ))）"
echo "PLAN gpus_per_job=$K slots=$S jobs=$J"
for ((s=0; s<S; s++)); do
  names=""; for ((i=s; i<J; i+=S)); do names="${names:+$names }$(job_name "$i")"; done
  echo "PLAN slot=$s gpus=$(slot_gpus "$s") jobs=$names"
done
[ "$DRY_RUN" -eq 1 ] && exit 0
OUT_BASE="$RUNS_BASE/m5/$BATCH_TAG"; mkdir -p "${OUT_BASE}"
echo "产物 ${OUT_BASE}"
# 暂存目录按调用隔离（$$ = 本脚本 PID）：多个调用可共用同一个 --batch_tag 并发跑，互不删对方的在途产物
STAGE="$OUT_BASE/.runs_$$"
pids=()
for ((s=0; s<S; s++)); do
  gpus="$(slot_gpus "$s")"
  (
    for ((i=s; i<J; i+=S)); do
      f="${JOBS[$i]}"; sd="${JOB_SEEDS[$i]}"; name="$(job_name "$i")"
      echo "[gpu $gpus] ▶ $name"
      if [ -n "$TRAIN_PARQUET" ]; then src=(--train_parquet "$TRAIN_PARQUET"); else src=(--splits_dir "$SPLITS_DIR"); fi
      args=("${src[@]}" --config "$f" --gpus "$gpus" --runs_base "$STAGE" --run_id "$name" --tag "$name"
            --port $((29700 + s)) --venv-python "$VENV_PY" --overwrite)
      [ -n "$sd" ] && args+=(--seed "$sd")
      [ -n "$MAX_STEPS" ] && args+=(--max_steps "$MAX_STEPS")
      [ -n "$BATCH_SIZE" ] && args+=(--batch_size "$BATCH_SIZE")
      bash "$_HERE/train_m5.sh" "${args[@]}" "${EXTRA[@]}" > "$OUT_BASE/${name}.log" 2>&1 \
        && echo "[gpu $gpus] ✅ $name" || echo "[gpu $gpus] ❌ ${name}（看 $OUT_BASE/${name}.log）"
      # 把产物目录提到 OUT_BASE/<name>
      [ -d "$STAGE/m5/$name" ] && { rm -rf "$OUT_BASE/$name"; mv "$STAGE/m5/$name" "$OUT_BASE/$name"; }
    done
  ) &
  pids+=($!)
done
for p in "${pids[@]}"; do wait "$p"; done
rm -rf "$STAGE"
cd "$BASE_DIR" && "$VENV_PY" -m forecast_backbones.m5.collect_runs "${OUT_BASE}" --out "$OUT_BASE/variants_summary.csv"
