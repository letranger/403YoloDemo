#!/usr/bin/env bash

set -euo pipefail

ASSET_DIR="${HOME}/.cache/trt_pose"
WEIGHT_URL="https://github.com/NVIDIA-AI-IOT/trt_pose/releases/download/v0.0.1/resnet18_baseline_att_224x224_A_epoch_249.pth"
TOPO_URL="https://raw.githubusercontent.com/NVIDIA-AI-IOT/trt_pose/master/tasks/human_pose/human_pose.json"

mkdir -p "${ASSET_DIR}"
cd "${ASSET_DIR}"

echo "Downloading to: ${ASSET_DIR}"

# -nc: no-clobber（已存在則不覆寫）
if command -v wget >/dev/null 2>&1; then
  wget -nc "${WEIGHT_URL}"
  wget -nc "${TOPO_URL}"
else
  curl -L -O "${WEIGHT_URL}"
  curl -L -O "${TOPO_URL}"
fi

WEIGHT_PATH="${ASSET_DIR}/resnet18_baseline_att_224x224_A_epoch_249.pth"
TOPO_PATH="${ASSET_DIR}/human_pose.json"

echo ""
echo "WEIGHT=${WEIGHT_PATH}"
echo "TOPO=${TOPO_PATH}"
echo ""
echo "已完成 trt_pose 模型與拓撲下載。"

