"""根据 results_dense.json / results_sparse.json 生成对比图。"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

SCEN = [("密集", os.path.join(HERE, "results", "results_dense.json")),
        ("稀疏", os.path.join(HERE, "results", "results_sparse.json"))]
TASKS = ["T1_verbatim", "T2_paraphrase", "T3_mixed"]
LABELS = ["T1 精确召回（ROSA 主场）", "T2 改写召回（ROSA 软肋）", "T3 混合预测"]

res = {tag: json.load(open(p)) for tag, p in SCEN}
order = list(res["密集"].keys())

fig, axes = plt.subplots(2, 3, figsize=(20, 11))
for r, (tag, _) in enumerate(SCEN):
    for c, (t, lab) in enumerate(zip(TASKS, LABELS)):
        ax = axes[r][c]
        accs = np.array([res[tag][n][t]["acc"] * 100 for n in order])
        cols = ["#d62728" if ("ROSA" in n) else
                ("#2ca02c" if n.startswith("DPSH-full") else
                 ("#7f7f7f" if n == "Unigram" else "#1f77b4")) for n in order]
        y = np.arange(len(order))
        ax.barh(y, accs, color=cols)
        ax.set_yticks(y)
        ax.set_yticklabels(order, fontsize=9)
        ax.set_xlabel("Top-1 准确率 (%)", fontsize=10)
        ax.set_title(f"[{tag}] {lab}", fontsize=12)
        ax.set_xlim(0, 120)
        for i, v in enumerate(accs):
            ax.text(v + 2, i, f"{v:.1f}", va="center", fontsize=8.5)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
plt.tight_layout()
out_path = os.path.join(HERE, "results", "dpsh_results.png")
plt.savefig(out_path, dpi=150)
print("saved:", out_path)
