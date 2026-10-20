#!/bin/bash
#
# train_m5.sh —— M5（TimesFM-3 + LoRA 微调；finetune.enabled=false 时只做零样本打分）单个配置的训练脚本。
# 与 train_m8.sh 同一套约定：同一个 python 与 LD_LIBRARY_PATH、同一份切分目录、accelerate launch、配置只改运行时副本。
# 底座 ~330M 参数只训 LoRA + output_head；V100 用 fp32。batch_size 是全局 batch（所有卡合计），每卡 = batch_size / 卡数，
# 卡数变了配方不变；显存不够就多给卡（每卡变小），或把 --batch_size 降到 8。
#
# 用法：
#   bash train_m5.sh --splits_dir data/splits/m8_v1 --station_meta forecast_backbones/m5/station_meta.csv --gpus 0   # 默认 f1_final
#   bash train_m5.sh --splits_dir data/splits/m8_v1 --config forecast_backbones/m5/config/z0_zeroshot.yaml --gpus 0
#   f 系列 yaml 的站表是线上 info.csv（南网站号 1780…）；家里切分的站号是 1001…，所以家里训练要 --station_meta 换回 station_meta.csv
#   bash train_m5.sh --splits_dir data/splits/m8_v1 --gpus 0 --max_steps 20 --eval_every 10 --run_id smoke   # 冒烟
#   批量跑变体 / 多种子见 train_m5_variants.sh
set -u
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$_HERE/forecast_backbones" ]; then BASE_DIR="$_HERE"; else BASE_DIR="$(cd "$_HERE/.." && pwd)"; fi

SPLITS_DIR="${BASE_DIR}/data/splits/m8_v1"
TRAIN_PARQUET=""                    # 给了就是全量训练：文件夹 / 通配符（加引号）/ 单个文件，不走切分目录，隐含 --refit
CONFIG="${BASE_DIR}/forecast_backbones/m5/config/f1_final.yaml"   # 默认 = 最终配方；未写的字段取 config.py 的 DEFAULT_CFG
STATION_META=""                     # 站表路径：覆盖配置 dataset.station_meta（线上 info.csv 表头自动识别）
RUNS_BASE="${BASE_DIR}/runs"
RUN_ID=""                           # 留空 = 配置 run.tag（或 变体文件名）+ 时间戳
GPUS="auto"
MAX_STEPS=""
EVAL_EVERY=""
BATCH_SIZE=""
PRECISION="fp32"
PORT=29700
CPU_THREADS=""
OVERWRITE=0
ALLOW_CPU=0
TAG=""
SEED=""
REFIT=0
VENV_PY="/home/ma-user/ts3/bin/python"
PRE_CUDA_LD="/usr/local/cuda-12.1/lib64:/usr/lib/x86_64-linux-gnu"

usage() {
  cat <<'USAGE'
用法: bash train_m5.sh [选项]
  --splits_dir <dir>       pipelines.time_splits 落的切分目录（train/valid.parquet + split_manifest.json；不需要 cal.parquet）
  --train_parquet <path>   全量训练：现成的训练 parquet（文件夹 / 加引号的通配符 / 文件），不切分、不剔除任何行、无验证集，
                           固定 --max_steps 步后存 checkpoint（隐含 --refit）。与 --splits_dir 二选一
  --config <yaml>          M5 配置（默认 forecast_backbones/m5/config/f1_final.yaml；各变体直接放在 config/ 下，只写与 config.py 的
                           DEFAULT_CFG 不同的字段；config/configs.yaml 是线上零样本配置，不能拿来训练）
  --station_meta <csv>     站表（覆盖 dataset.station_meta，列名映射随之清空、表头自动识别）。家里切分用
                           forecast_backbones/m5/station_meta.csv；也可直接给线上的 info.csv
  --run_id <name>          产物目录名：<runs_base>/m5/<run_id>（默认 <tag>_<日期_时间>）
  --runs_base <dir>        产物根目录（默认 <仓库>/runs）
  --gpus <csv|auto>        用哪些卡（默认 auto = 全部可见卡）；每张卡一个进程
  --gpu <id>               同 --gpus <id>
  --max_steps <n>          覆盖 train.schedule.max_steps；0 = 不限（按 epochs_max）
  --eval_every <n>         覆盖 train.schedule.eval_every_steps
  --batch_size <n>         覆盖 train.schedule.batch_size（全局 batch，所有卡合计；须能被卡数整除）
  --precision <fp32|bf16>  训练精度（默认 fp32；V100 不支持 bf16）
  --tag <name>             写进 reports/summary.csv 的变体名（默认取配置 run.tag）
  --seed <n>               覆盖 run.seeds（模型初始化 + 数据顺序 + 增强随机数）
  --refit                  全量重训：train ∪ valid 全当训练行，不验证、不早停，固定 --max_steps 步后存最终权重
                           （配方选定后给最终模型；切分用 data/splits/m8_refit.yaml 落的目录，valid.parquet 为 0 行；零样本 z0 不需要 refit）
  --port <n>               多进程通信端口（默认 29700）
  --cpu_threads <n>        每个进程的 CPU 线程数
  --overwrite              产物目录已存在时删掉重训
  --venv-python <path>     python 解释器（默认 /home/ma-user/ts3/bin/python）
  --allow-cpu              没有 GPU 时用 CPU 跑（仅排查）
USAGE
}
while [ $# -gt 0 ]; do
  case "$1" in
    --splits_dir)   SPLITS_DIR="$2";  shift 2 ;;
    --train_parquet) TRAIN_PARQUET="$2"; shift 2 ;;
    --config)       CONFIG="$2";      shift 2 ;;
    --station_meta) STATION_META="$2"; shift 2 ;;
    --run_id)       RUN_ID="$2";      shift 2 ;;
    --runs_base)    RUNS_BASE="$2";   shift 2 ;;
    --gpus|--gpu)   GPUS="$2";        shift 2 ;;
    --max_steps)    MAX_STEPS="$2";   shift 2 ;;
    --eval_every)   EVAL_EVERY="$2";  shift 2 ;;
    --batch_size)   BATCH_SIZE="$2";  shift 2 ;;
    --precision)    PRECISION="$2";   shift 2 ;;
    --tag)          TAG="$2";         shift 2 ;;
    --seed)         SEED="$2";        shift 2 ;;
    --refit)        REFIT=1;          shift   ;;
    --port)         PORT="$2";        shift 2 ;;
    --cpu_threads)  CPU_THREADS="$2"; shift 2 ;;
    --overwrite)    OVERWRITE=1;      shift   ;;
    --venv-python)  VENV_PY="$2";     shift 2 ;;
    --allow-cpu)    ALLOW_CPU=1;      shift   ;;
    -h|--help)      usage; exit 0 ;;
    *) echo "未知参数: $1" >&2; usage; exit 1 ;;
  esac
done

[ -f "$VENV_PY" ] || { echo "❌ python 解释器不存在: $VENV_PY" >&2; exit 1; }
[ -f "$CONFIG" ]  || { echo "❌ 找不到配置: $CONFIG" >&2; exit 1; }
if [ -n "$TRAIN_PARQUET" ]; then
  REFIT=1
  if [ -d "$TRAIN_PARQUET" ]; then n=$(find "$TRAIN_PARQUET" -name "*.parquet" 2>/dev/null | wc -l); else n=$(compgen -G "$TRAIN_PARQUET" | wc -l); fi   # 目录递归（date=xxx 分区）
  [ "$n" -gt 0 ] || { echo "❌ --train_parquet 没有匹配到 parquet: $TRAIN_PARQUET" >&2; exit 1; }
  [ -n "$MAX_STEPS" ] || { echo "❌ 全量训练没有验证集、不能早停：请给 --max_steps（按验证切分上各种子 best_step 的上限给）" >&2; exit 1; }
else
  for f in train.parquet valid.parquet split_manifest.json; do
    [ -f "$SPLITS_DIR/$f" ] || { echo "❌ 切分目录缺 $f: $SPLITS_DIR" >&2; exit 1; }
  done
fi
case "$PRECISION" in fp32|bf16) ;; *) echo "❌ --precision 只能是 fp32 或 bf16" >&2; exit 1 ;; esac

if [ "$GPUS" = "auto" ]; then
  GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ' | paste -sd, -)"
fi
if [ -z "$GPUS" ]; then
  if [ "$ALLOW_CPU" -eq 1 ]; then echo "⚠️  没检测到 GPU，按 --allow-cpu 用 CPU 跑"; else
    echo "❌ 没检测到 GPU；用 --gpus 指定，或确要用 CPU 请加 --allow-cpu" >&2; exit 1; fi
fi
IFS=',' read -r -a GPU_LIST <<< "$GPUS"
NP="${#GPU_LIST[@]}"; [ "$NP" -ge 1 ] || NP=1
if [ -z "$CPU_THREADS" ]; then
  NCPU="$(nproc 2>/dev/null || getconf _NPROCESSORS_ONLN 2>/dev/null || echo "$NP")"
  CPU_THREADS=$(( NCPU / NP )); [ "$CPU_THREADS" -ge 1 ] || CPU_THREADS=1
fi
export LD_LIBRARY_PATH="${PRE_CUDA_LD}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
[ -n "$GPUS" ] && export CUDA_VISIBLE_DEVICES="$GPUS"
cd "$BASE_DIR" || exit 1

CFG_TAG="$("$VENV_PY" -c "import yaml,sys; c=yaml.safe_load(open(sys.argv[1],encoding='utf-8')) or {}; print((c.get('run') or {}).get('tag') or '')" "$CONFIG" 2>/dev/null)"
[ -z "$CFG_TAG" ] && CFG_TAG="$(basename "$CONFIG" .yaml)"
[ -z "$TAG" ] && TAG="$CFG_TAG"
[ -z "$RUN_ID" ] && RUN_ID="${TAG}_$(date +%Y%m%d_%H%M%S)"
RUN_DIR="$RUNS_BASE/m5/$RUN_ID"
LOG_FILE="$BASE_DIR/logger/m5_${RUN_ID}_train.log"
TMP_CFG="$RUNS_BASE/.m5_cfg_${RUN_ID}.yaml"
mkdir -p "$RUNS_BASE" "$BASE_DIR/logger"
if [ -d "$RUN_DIR" ] && [ -n "$(ls -A "$RUN_DIR" 2>/dev/null)" ]; then
  if [ "$OVERWRITE" -eq 1 ]; then rm -rf "$RUN_DIR"; else
    echo "❌ 产物目录已存在且非空: ${RUN_DIR}（换 --run_id，或加 --overwrite）" >&2; exit 1; fi
fi

"$VENV_PY" - "$CONFIG" "$TMP_CFG" "$PRECISION" "$BATCH_SIZE" "$EVAL_EVERY" "$TAG" "$SEED" "$NP" "$STATION_META" <<'PYEOF'
import sys, yaml, os
src, dst, precision, batch_size, eval_every, tag, seed, n_proc, station_meta = sys.argv[1:10]
cfg = yaml.safe_load(open(src, encoding="utf-8")) or {}
sch = cfg.setdefault("train", {}).setdefault("schedule", {})
sch["precision"] = precision
if batch_size: sch["batch_size"] = int(batch_size)
if eval_every: sch["eval_every_steps"] = int(eval_every)
cfg.setdefault("run", {})["tag"] = tag
if seed != "": cfg["run"]["seeds"] = [int(seed)]
if station_meta:
    cfg.setdefault("dataset", {})["station_meta"] = os.path.abspath(station_meta)
    cfg["dataset"]["station_meta_cols"] = None      # 换了站表，配置里的列名映射（info.csv 表头）不再适用：标准表头 / info.csv 表头都自动识别
gb = int(sch.get("batch_size") or 16)                     # 变体没写就是默认 16（config.py DEFAULT_CFG）
if gb % int(n_proc):
    print(f"❌ 全局 batch {gb} 不能被卡数 {n_proc} 整除：换 --gpus 的卡数，或换 --batch_size", file=sys.stderr); sys.exit(4)
yaml.safe_dump(cfg, open(dst, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
meta = (cfg.get("dataset") or {}).get("station_meta", "forecast_backbones/m5/station_meta.csv")
print(f"STATION_META={meta}")
sys.exit(0 if os.path.exists(meta) else 3)
PYEOF
RC=$?
if [ $RC -eq 3 ]; then echo "❌ dataset.station_meta 路径不存在（见上一行）；请先填好站表" >&2; rm -f "$TMP_CFG"; exit 1
elif [ $RC -eq 4 ]; then rm -f "$TMP_CFG"; exit 1
elif [ $RC -ne 0 ]; then echo "❌ 生成运行时配置失败" >&2; rm -f "$TMP_CFG"; exit 1; fi

echo "=============================================================="
echo "M5 训练   tag=$TAG${SEED:+   seed=$SEED}$([ "$REFIT" -eq 1 ] && echo '   [全量重训 refit：不验证、不早停]')"
echo "CONFIG     : $CONFIG  (运行时副本: $TMP_CFG)"
if [ -n "$TRAIN_PARQUET" ]; then echo "TRAIN_PARQUET: $TRAIN_PARQUET  （全量：不切分、不剔行、无验证）"; else echo "SPLITS_DIR : $SPLITS_DIR"; fi
echo "RUN_DIR    : $RUN_DIR"
if [ -n "$GPUS" ]; then echo "设备       : $NP 张卡 (CUDA_VISIBLE_DEVICES=$GPUS)   精度: $PRECISION   每进程 CPU 线程: $CPU_THREADS"; else echo "设备       : CPU"; fi
echo "日志       : $LOG_FILE"
echo "=============================================================="
mkdir -p "$RUN_DIR"
cmd=("$VENV_PY" -m accelerate.commands.launch --num_processes="$NP" --num_machines=1 --dynamo_backend=no --mixed_precision=no
     --main_process_port="$PORT" --num_cpu_threads_per_process="$CPU_THREADS")
[ "$NP" -gt 1 ] && cmd+=(--multi_gpu)
[ -z "$GPUS" ] && cmd+=(--cpu)
cmd+=(-m forecast_backbones.m5.train --config "$TMP_CFG" --out "$RUN_DIR" --tag "$TAG")
if [ -n "$TRAIN_PARQUET" ]; then cmd+=(--train-parquet "$TRAIN_PARQUET"); else cmd+=(--splits-dir "$SPLITS_DIR"); fi
[ -n "$MAX_STEPS" ] && cmd+=(--max-steps "$MAX_STEPS")
[ "$REFIT" -eq 1 ] && cmd+=(--refit)
echo "CMD: ${cmd[*]}"
"${cmd[@]}" 2>&1 | tee "$LOG_FILE"
rc=${PIPESTATUS[0]}
rm -f "$TMP_CFG"
if [ $rc -ne 0 ]; then echo "❌ M5 训练失败 (exit=$rc)，日志: $LOG_FILE" >&2; exit 1; fi
for f in model_config.yaml train_log.json reports/summary.csv; do
  [ -f "$RUN_DIR/$f" ] || { echo "❌ 训练结束但缺少 $RUN_DIR/$f" >&2; exit 1; }
done
# 零样本 run（train_log.json 里 "zero_shot": true）不写权重；LoRA run 必须有 model.safetensors
if ! grep -q '"zero_shot": true' "$RUN_DIR/train_log.json" && [ ! -f "$RUN_DIR/model.safetensors" ]; then
  echo "❌ 训练结束但缺少 $RUN_DIR/model.safetensors" >&2; exit 1
fi
echo "✅ M5 完成: $RUN_DIR   摘要: $RUN_DIR/reports/summary.csv"
