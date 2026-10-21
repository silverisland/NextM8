"""Model-independent YAML configuration utilities."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import yaml


def deep_merge(base: dict, override: dict) -> dict:
    """dict 递归合并；非 dict（含 list）整体替换。"""
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def interpolate(cfg: dict) -> dict:
    """把字符串里的 ${run.out_dir} 替换成 cfg['run']['out_dir']。只支持这一个占位符。"""
    out_dir = str(cfg.get("run", {}).get("out_dir", ""))

    def _walk(x):
        if isinstance(x, dict):
            return {k: _walk(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_walk(v) for v in x]
        if isinstance(x, str) and "${run.out_dir}" in x:
            return x.replace("${run.out_dir}", out_dir)
        return x

    return _walk(cfg)


def dump_config(cfg: dict, path) -> None:
    """配置写成 yaml（元组转列表，键序不变）。"""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plain = json.loads(json.dumps(cfg, ensure_ascii=False, default=str))
    Path(path).write_text(
        yaml.safe_dump(plain, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )


def raw_override(config_path) -> dict:
    """yaml 路径 / dict / None → 尚未合并、未插值的 override dict。"""
    if config_path is None:
        return {}
    if isinstance(config_path, dict):
        return config_path
    return yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
