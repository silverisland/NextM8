#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
exec "${PYTHON:-python}" -m forecast_backbones.m4_rum.evaluate "$@"
