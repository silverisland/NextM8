#!/bin/bash
# ============================================================
# eval_m7_decision.sh —— M7（Timestral-209M 零样本）决策层消融：同一个模型，只换上报点规则
#
# 每个变体 = forecast_backbones/m7/config/configs.yaml 深合并 config/decision/<变体>.yaml，
# 写成 $OUT_ROOT/<变体>/model_config.yaml，再跑 eval_pretrain.sh（17 站并行，指标口径与 M7 基线评测相同）。
# 决策层没有可训练参数，各变体只差 logits 的取法；但每个变体会重新推理一遍。
#
# 变体（config/decision/）：
#   a1_median         中位数（排序 + 钳 [0,cap] 后的 Q(0.5)）
#   a2_mean           分布均值（21 档 → 200 点网格的均值）
#   a3_pstar_9        p*，只用 9 档 0.1~0.9（= M5 d2_pstar_point 口径）
#   a4_pstar_21       p*，模型全部 21 档 0.01~0.99 → 200 点网格
#   a5_pstar_21_g300  p*，21 档 → 300 点网格
# 旧基线（decision.point: null，模型 0.5 档直出、不钳 cap）= 直接 MODEL=m7 bash eval_shell/eval_pretrain.sh。
#
# 用法（cwd 任意）：
#   bash eval_shell/eval_m7_decision.sh                                  # 全部变体
#   VARIANTS="a1_median a4_pstar_21" bash eval_shell/eval_m7_decision.sh # 指定变体
# 环境变量：
#   VARIANTS   变体名列表        默认 config/decision/ 下全部
#   OUT_ROOT   输出根目录        默认 $BASE_DIR/runs/m7/abl_decision
#   PYTHON / STATION_DIR / NUM_WORKERS / BATCH_SIZE 原样传给 eval_pretrain.sh
#
# 产出：$OUT_ROOT/<变体>/model_config.yaml、$OUT_ROOT/<变体>/metrics_ens/{站}-summary.json，
#       $OUT_ROOT/summary.csv（每变体一行：各指标的 17 站平均 + 站数）
# 幂等：eval_pretrain.sh 对已有 summary 的站 SKIP；改了变体 yaml 要重跑请先删 $OUT_ROOT/<变体>/metrics_ens
# ============================================================
set -u
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${BASE_DIR}"

PYTHON="${PYTHON:-/home/ma-user/ts3/bin/python}"
export PYTHON
M7_CFG="${BASE_DIR}/forecast_backbones/m7/config/configs.yaml"
VAR_DIR="${BASE_DIR}/forecast_backbones/m7/config/decision"
OUT_ROOT="${OUT_ROOT:-${BASE_DIR}/runs/m7/abl_decision}"
if [ -z "${VARIANTS:-}" ]; then
  VARIANTS="$(find "${VAR_DIR}" -maxdepth 1 -name '*.yaml' -exec basename {} .yaml \; | sort | tr '\n' ' ')"
fi

echo "=========================================="
echo " M7 决策层消融"
echo "  变体    : ${VARIANTS}"
echo "  OUT_ROOT: ${OUT_ROOT}"
echo "=========================================="

FAIL=0
for v in ${VARIANTS}; do
  vy="${VAR_DIR}/${v}.yaml"
  if [ ! -f "$vy" ]; then
    echo "错误: 变体 $vy 不存在" >&2; FAIL=$(( FAIL + 1 )); continue
  fi
  d="${OUT_ROOT}/${v}"
  mkdir -p "$d"
  "$PYTHON" - "$M7_CFG" "$vy" "$d/model_config.yaml" <<'EOF' || { echo "错误: 合并 $v 配置失败" >&2; FAIL=$(( FAIL + 1 )); continue; }
import sys, yaml
def merge(a, b):
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = merge(a[k], v) if isinstance(v, dict) and isinstance(a.get(k), dict) else v
    return out
base, var, dst = sys.argv[1:4]
cfg = merge(yaml.safe_load(open(base, encoding="utf-8")) or {}, yaml.safe_load(open(var, encoding="utf-8")) or {})
yaml.safe_dump(cfg, open(dst, "w", encoding="utf-8"), allow_unicode=True, sort_keys=False)
print(f"[{dst}] decision = {cfg.get('decision')}")
EOF
  echo "===== 变体 $v ====="
  MODEL=m7 POINT= CKPT_DIR="$d" OUT_DIR="$d/metrics_ens" bash "${SCRIPT_DIR}/eval_pretrain.sh" || FAIL=$(( FAIL + 1 ))   # 上报点只由变体 yaml 决定
done

# 汇总：每变体各指标的站平均
"$PYTHON" - "$OUT_ROOT" ${VARIANTS} <<'EOF'
import glob, json, sys
from pathlib import Path
import pandas as pd
root, variants = Path(sys.argv[1]), sys.argv[2:]
rows = []
for v in variants:
    files = sorted(glob.glob(str(root / v / "metrics_ens" / "*-summary.json")))
    if not files:
        continue
    df = pd.DataFrame([json.load(open(f, encoding="utf-8")) for f in files])
    num = df.select_dtypes("number").drop(columns=["cap_power", "rows"], errors="ignore")
    rows.append({"variant": v, "n_stations": len(df), **num.mean().round(4).to_dict()})
if rows:
    out = pd.DataFrame(rows)
    out.to_csv(root / "summary.csv", index=False)
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(out.to_string(index=False))
    print(f"[summary] {root / 'summary.csv'}")
else:
    print("没有可汇总的 summary.json")
EOF

echo "失败变体数: $FAIL"
[ "$FAIL" -eq 0 ]
