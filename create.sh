#!/usr/bin/env bash
set -euo pipefail

ENV_FILE="torch_cuda128_environment.yml"
ENV_NAME="torch_cuda128"
DOWNLOAD_SCRIPT="modelscope_download.py"
TRAIN_SCRIPT="main_train.py"

echo "===== [1/7] 检查 conda ====="
if ! command -v conda >/dev/null 2>&1; then
    echo "错误：未找到 conda，请先安装并配置好 conda。"
    exit 1
fi

echo "===== [2/7] 初始化 conda ====="
CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

echo "===== [3/7] 创建环境（如果已存在则跳过） ====="
if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
    echo "环境 ${ENV_NAME} 已存在，跳过创建。"
else
    conda env create -f "${ENV_FILE}"
fi
pip install modelscope
echo "===== [4/7] 激活环境 ====="
conda activate "${ENV_NAME}"

echo "===== [5/7] 下载数据集 ====="
python "${DOWNLOAD_SCRIPT}"
echo "数据集下载脚本执行完成。"

echo "===== [6/7] 查找 project_dataset 目录 ====="
PROJECT_DATASET_DIR="$(find "$(pwd)" -type d -name "BLOP" 2>/dev/null | head -n 1 | xargs -r dirname)"

if [ -z "${PROJECT_DATASET_DIR}" ]; then
    PROJECT_DATASET_DIR="$(find / -type d -name "project_dataset" 2>/dev/null | head -n 1)"
fi

if [ -z "${PROJECT_DATASET_DIR}" ]; then
    echo "错误：未找到 project_dataset 目录，也没有找到 BLOP 线索目录。"
    exit 1
fi

echo "找到 project_dataset 目录：${PROJECT_DATASET_DIR}"

PROJECT_PARENT_DIR="$(dirname "${PROJECT_DATASET_DIR}")"
PROJECT_MODEL_DIR="${PROJECT_PARENT_DIR}/project_model"
OUTPUT_DIR="${PROJECT_MODEL_DIR}/huber/1"
PROBLEM_DIR="${PROJECT_DATASET_DIR}/huber_dataset"

echo "===== [7/7] 创建输出目录并启动训练 ====="
mkdir -p "${OUTPUT_DIR}"

if [ ! -d "${PROBLEM_DIR}" ]; then
    echo "错误：未找到训练数据目录 ${PROBLEM_DIR}"
    exit 1
fi

echo "训练数据目录：${PROBLEM_DIR}"
echo "模型输出目录：${OUTPUT_DIR}"
