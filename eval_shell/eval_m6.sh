#!/bin/bash
# ============================================================
# eval_m6.sh —— M6（TimesFM-3 + LoRA）逐站 192 步并行评测
#
# 与 eval_m5.sh 同构（各 worker 独占 GPU，按连续块切分）：
#   每 worker 一个 GPU / CUDA_VISIBLE_DEVICES，串行处理自己的站块，
#   worker 日志写到 $OUT_DIR/logs/worker_<i>.log，逐站跑 eval_shell/m6_station_script.py
#   （训练产物 -> M6 模型 -> 推理 -> 指标 -> {站}-metrics.xlsx + {站}-summary.json）。
# 输出与 M1-M5 的 metrics_ens 同构，可直接供 make_models_summary 汇总。
#
# 用法（任意 cwd；相对路径都相对仓库根）：
#   CKPT_DIR=runs/m6/lora_20260922_1200 bash eval_shell/eval_m6.sh
#   RUN_ID=lora_20260922_1200 bash eval_shell/eval_m6.sh
# 环境变量：
#   CKPT_DIR     M6 训练产物目录   必给（或给 RUN_ID，取 $BASE_DIR/runs/m6/$RUN_ID）
#   M6_CONFIG    训练用 yaml       默认空（读 $CKPT_DIR/model_config.yaml；旧 run 没有时才需要）
#   STATION_DIR  test 数据          默认 $BASE_DIR/datasets/test
#   OUT_DIR      结果输出目录       默认 $CKPT_DIR/metrics_ens
#   CAP_POWER    回退容量           默认 465（自带 cap_power_on 的站会用站容量）
#   NUM_WORKERS  并行 worker 数     默认 8
#   BATCH_SIZE   推理 batch         默认空（用配置里的 eval.batch_size）
#   PYTHON       python 解释器      默认 /home/ma-user/ts3/bin/python
#   DRY_RUN      1=只打印分片       默认 0
#
# 产出：$OUT_DIR/{station}-metrics.xlsx + {station}-summary.json
# 幂等：已存在 {station}-summary.json 的站 SKIP（判据=summary 存在，非 python rc）
# ============================================================
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${BASE_DIR}"

PYTHON="${PYTHON:-/home/ma-user/ts3/bin/python}"
CUDA_LIB="LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:/usr/lib/x86_64-linux-gnu"
STATION_SCRIPT="${SCRIPT_DIR}/m6_station_script.py"

if [ -z "${CKPT_DIR:-}" ] && [ -n "${RUN_ID:-}" ]; then
  CKPT_DIR="${BASE_DIR}/runs/m6/${RUN_ID}"
fi
if [ -z "${CKPT_DIR:-}" ]; then
  echo "错误: 请设 CKPT_DIR=runs/m6/<run_id>（train_m6.sh 的产物目录）或 RUN_ID=<run_id>" >&2
  exit 1
fi
CKPT_DIR="$(cd "${CKPT_DIR}" 2>/dev/null && pwd)" || { echo "错误: 产物目录不存在" >&2; exit 1; }
M6_CONFIG="${M6_CONFIG:-}"
STATION_DIR="${STATION_DIR:-${BASE_DIR}/datasets/test}"
OUT_DIR="${OUT_DIR:-${CKPT_DIR}/metrics_ens}"
CAP_POWER="${CAP_POWER:-465}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BATCH_SIZE="${BATCH_SIZE:-}"
PARALLEL_LOGS="${OUT_DIR}/logs"

if [ -z "${M6_CONFIG}" ] && [ ! -f "${CKPT_DIR}/model_config.yaml" ]; then
  echo "错误: ${CKPT_DIR} 没有 model_config.yaml；旧 run 请用 M6_CONFIG=<训练用的 yaml> 指定" >&2
  exit 1
fi
if [ ! -f "${CKPT_DIR}/model.safetensors" ] && [ ! -f "${CKPT_DIR}/lora_weights.pt" ]; then
  echo "错误: ${CKPT_DIR} 没有 model.safetensors（旧 run 为 lora_weights.pt）" >&2
  exit 1
fi

EXTRA_ARGS=()
[ -n "${M6_CONFIG}" ] && EXTRA_ARGS+=(--config "${M6_CONFIG}")
[ -n "${BATCH_SIZE}" ] && EXTRA_ARGS+=(--batch_size "${BATCH_SIZE}")

# 收集 station=*.parquet
STATIONS=()
while IFS= read -r st; do STATIONS+=("$st"); done < <(
  for f in "${STATION_DIR}"/station=*.parquet; do [ -e "$f" ] && basename "$f" .parquet | sed 's/^station=//'; done | sort)
N_STATIONS=${#STATIONS[@]}

mkdir -p "${OUT_DIR}" "${PARALLEL_LOGS}"

echo "=========================================="
echo " M6 (TimesFM-3 + LoRA) 逐站 192 步并行评测"
echo "  CKPT_DIR   : ${CKPT_DIR}"
echo "  M6_CONFIG  : ${M6_CONFIG:-${CKPT_DIR}/model_config.yaml}"
echo "  STATION_DIR: ${STATION_DIR}"
echo "  OUT_DIR    : ${OUT_DIR}"
echo "  CAP_POWER  : ${CAP_POWER}"
echo "  NUM_WORKERS: ${NUM_WORKERS}  BATCH_SIZE: ${BATCH_SIZE:-配置里的 eval.batch_size}"
echo "  站数: ${N_STATIONS}     worker 日志: ${PARALLEL_LOGS}"
echo "=========================================="

if [ "$N_STATIONS" -eq 0 ]; then
  echo "错误: ${STATION_DIR} 下没有 station=*.parquet" >&2
  exit 1
fi

# ============================================================
# 切块：base = ⌊N/NW⌋，前 rem 个块各 +1
# ============================================================
base=$(( N_STATIONS / NUM_WORKERS ))
rem=$(( N_STATIONS % NUM_WORKERS ))
start=0
declare -a BLOCK_START=()
declare -a BLOCK_END=()
for ((i=0; i<NUM_WORKERS; i++)); do
  len=$base
  if [ "$i" -lt "$rem" ]; then len=$(( len + 1 )); fi
  BLOCK_START[i]=$start
  start=$(( start + len ))
  BLOCK_END[i]=$start
done

for ((i=0; i<NUM_WORKERS; i++)); do
  s=${BLOCK_START[i]}
  e=${BLOCK_END[i]}
  if [ "$s" -eq "$e" ]; then
    echo "  worker[$i] GPU${i} : (空)"
  else
    echo "  worker[$i] GPU${i} : ${STATIONS[@]:$s:$((e-s))}"
  fi
done

if [ "${DRY_RUN:-0}" = "1" ]; then
  echo "[DRY_RUN] 仅打印分片，不启动推理。退出。"
  exit 0
fi

# ============================================================
# 单站任务：
#   - summary 已存在 -> SKIP
#   - 跑 m6_station_script.py，OK/FAIL 以 summary 是否生成为准
# ============================================================
run_one() {
  local st="$1"
  local summary_file="${OUT_DIR}/${st}-summary.json"
  if [ -f "$summary_file" ]; then
    echo "SKIP $st (已存在 summary)"
    return 0
  fi
  echo "===== 开始: $st ====="
  env $CUDA_LIB CUDA_VISIBLE_DEVICES="$GPU_ID" \
    "$PYTHON" "$STATION_SCRIPT" \
      --checkpoint_dir "$CKPT_DIR" \
      --station_path "${STATION_DIR}/station=${st}.parquet" \
      --out_dir "$OUT_DIR" \
      --cap_power "$CAP_POWER" \
      "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}" \
      2>&1 | grep -avE "it/s\]|s/it\]|batch/s\]|s/batch\]|Generating train split"
  local rc=${PIPESTATUS[0]}
  if [ -f "$summary_file" ]; then
    echo "OK   $st (rc=$rc)"
  else
    echo "FAIL $st (rc=$rc)"
  fi
  return 0
}

# ============================================================
# 启动 NUM_WORKERS 个后台 worker，各 worker 串行处理自己的块。
# worker 计数 OK/FAIL/SKIP；结束行打印最终统计；任一站 FAIL 则该 worker 退出码 1
# ============================================================
declare -a WORKER_PIDS=()
for ((i=0; i<NUM_WORKERS; i++)); do
  GPU_ID=$i
  s=${BLOCK_START[i]}
  e=${BLOCK_END[i]}
  (
    OK=0; FAIL=0; SKIP=0
    echo "===== worker[$i] 启动 (GPU${GPU_ID}, ${s}..${e}) ====="
    for ((k=s; k<e; k++)); do
      out=$(run_one "${STATIONS[$k]}")
      echo "$out"
      last="${out##*$'\n'}"                 # 只看 run_one 的最后一行，python 日志里的字样不影响计数
      case "$last" in
        "SKIP "*) SKIP=$(( SKIP + 1 ));;
        "OK   "*) OK=$(( OK + 1 ));;
        *) FAIL=$(( FAIL + 1 ));;
      esac
    done
    echo "===== worker[$i] 结束 (GPU${GPU_ID}): OK=$OK FAIL=$FAIL SKIP=$SKIP ====="
    [ "$FAIL" -eq 0 ]
  ) > "${PARALLEL_LOGS}/worker_${i}.log" 2>&1 &
  WORKER_PIDS+=("$!")
done

FAIL_TOTAL=0
for p in "${WORKER_PIDS[@]}"; do
  wait "$p" || FAIL_TOTAL=$(( FAIL_TOTAL + 1 ))
done

echo ""
echo "=========================================="
echo " M6 评测全部完毕"
echo "  站点总计   : ${N_STATIONS}"
echo "  summary 数 : $(find "${OUT_DIR}" -maxdepth 1 -name '*-summary.json' | wc -l)"
echo "  失败 worker: ${FAIL_TOTAL}"
echo "  结果目录   : ${OUT_DIR}"
echo "=========================================="
[ "$FAIL_TOTAL" -eq 0 ]
