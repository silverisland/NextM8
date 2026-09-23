#!/bin/bash
# ============================================================
# conformal_m5.sh —— M5 离线滚动 conformal（收缩加权）测试
#   第 1 步：NUM_GPUS 个进程各占一张卡，按站分片推理，落盘原始分位到 $OUT_DIR/raw/{站}.parquet（已有的站跳过）
#   第 2 步：全部站推理完后，CPU 滚动回放，报表写到 $OUT_DIR/replay/
#
# 用法（cwd 任意）：
#   bash eval_shell/conformal_m5.sh
# 环境变量（均有默认）：
#   STATION_DIR  测试集目录（station=*.parquet）  默认 $BASE_DIR/datasets/test
#   OUT_DIR      输出目录                        默认 $BASE_DIR/runs/m5_conformal
#   NUM_GPUS     并行卡数                        默认 8
#   BATCH_SIZE   推理 batch                      默认 64
#   STATION_META 站点经纬度表                    默认 forecast_backbones/m6/station_meta.csv
#   KS           收缩强度 k，空格分隔             默认 "5 15 30"
#   POOL         season | calendar               默认 season
#   CLUSTERS     station,cluster 两列 csv         默认空 = 全部站一类
#   SKIP_PREDICT 1 = 跳过第 1 步，只重跑回放      默认 0
# ============================================================
set -u
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${BASE_DIR}"

PYTHON="${PYTHON:-/home/ma-user/ts3/bin/python}"
export LD_LIBRARY_PATH="/usr/local/cuda-12.1/lib64:/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

STATION_DIR="${STATION_DIR:-${BASE_DIR}/datasets/test}"
OUT_DIR="${OUT_DIR:-${BASE_DIR}/runs/m5_conformal}"
NUM_GPUS="${NUM_GPUS:-8}"
BATCH_SIZE="${BATCH_SIZE:-64}"
STATION_META="${STATION_META:-${BASE_DIR}/forecast_backbones/m6/station_meta.csv}"
KS="${KS:-5 15 30}"
POOL="${POOL:-season}"
CLUSTERS="${CLUSTERS:-}"
LOG_DIR="${OUT_DIR}/logs"
mkdir -p "${OUT_DIR}/raw" "${LOG_DIR}"

N_STATIONS=$(find "${STATION_DIR}" -maxdepth 1 -name 'station=*.parquet' | wc -l)
echo "=========================================="
echo " M5 离线 conformal"
echo "  STATION_DIR : $STATION_DIR（$N_STATIONS 站）"
echo "  OUT_DIR     : $OUT_DIR"
echo "  NUM_GPUS    : $NUM_GPUS   BATCH_SIZE: $BATCH_SIZE"
echo "  STATION_META: $STATION_META"
echo "  KS=$KS  POOL=$POOL  CLUSTERS=${CLUSTERS:-<全部一类>}"
echo "=========================================="
if [ "$N_STATIONS" -eq 0 ]; then
  echo "错误: $STATION_DIR 下没有 station=*.parquet" >&2
  exit 1
fi

if [ "${SKIP_PREDICT:-0}" != "1" ]; then
  PIDS=()
  for ((i=0; i<NUM_GPUS; i++)); do
    CUDA_VISIBLE_DEVICES=$i "$PYTHON" -m forecast_backbones.m5.conformal_replay predict \
      --station_dir "$STATION_DIR" --out_dir "$OUT_DIR/raw" \
      --station_meta "$STATION_META" --batch_size "$BATCH_SIZE" \
      --shard "$i" --num_shards "$NUM_GPUS" \
      > "${LOG_DIR}/predict_${i}.log" 2>&1 &
    PIDS+=("$!")
    echo "  worker[$i] GPU$i -> ${LOG_DIR}/predict_${i}.log"
  done
  FAIL=0
  for p in "${PIDS[@]}"; do wait "$p" || FAIL=$((FAIL + 1)); done
  N_RAW=$(find "${OUT_DIR}/raw" -maxdepth 1 -name '*.parquet' | wc -l)
  echo "推理完毕：原始分位表 $N_RAW / $N_STATIONS 站，失败 worker $FAIL 个"
  grep -h "没有坐标" "${LOG_DIR}"/predict_*.log | sort -u
  if [ "$FAIL" -ne 0 ] || [ "$N_RAW" -ne "$N_STATIONS" ]; then
    echo "有站没推理成功，先看 ${LOG_DIR}/predict_*.log；修好后重跑本脚本（已完成的站会跳过）" >&2
    exit 1
  fi
fi

EXTRA=()
if [ -n "$CLUSTERS" ]; then EXTRA+=(--clusters "$CLUSTERS"); fi
# shellcheck disable=SC2086
"$PYTHON" -m forecast_backbones.m5.conformal_replay replay \
  --raw_dir "$OUT_DIR/raw" --out_dir "$OUT_DIR/replay" --k $KS --pool "$POOL" ${EXTRA[@]+"${EXTRA[@]}"} \
  2>&1 | tee "${LOG_DIR}/replay.log"
