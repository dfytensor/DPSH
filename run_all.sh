#!/usr/bin/env bash
# DPSH vs ROSA 实验一键复现
# 用法: bash run_all.sh            # 复现已验证的基线结果
#       bash run_all.sh --full     # 追加聚类准则消融（下一步实验，未跑完）
set -e
cd "$(dirname "$0")"

FLAG="${1:-}"
echo "==> [1/3] 密集场景（事实句比例 0.32）"
python3 -u dpsh_experiment.py --scenario dense $FLAG 2>&1 | tee logs/dense.log

echo "==> [2/3] 稀疏场景（事实句比例 0.05）"
python3 -u dpsh_experiment.py --scenario sparse $FLAG 2>&1 | tee logs/sparse.log

echo "==> [3/3] 绘图"
python3 plot_results.py

echo "==> 完成。结果: results/results_dense.json, results/results_sparse.json, results/dpsh_results.png"
