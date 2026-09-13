<div align="center">

# DPSH — Differentiable Probabilistic Suffix Hypergraph

**可微概率后缀超图：检索硬、聚合软 —— 与 ROSA 的对比及真实链路验证**

英文 | 中文 · 合成基准 + 真实文本 PPL + 神经融合 · 全部结果可复现

</div>

---

## 1. 这是什么

**ROSA**：离散后缀自动机——最长精确匹配 → 硬复制后继 token。无参数、无泛化。

**DPSH**：可微概率后缀超图——检索保持离散高效（后缀索引 + 聚类超边），聚合全程连续可微（可学习多阶混合 + 语义近邻 + 超边归并 + 证据门控），输出完整 next-token 分布而非硬 token。

本仓库包含：

1. **合成基准**：判别性任务上证明机制因果（T1 精确 / T2 改写 / T3 混合）
2. **真实链路验证**（Phase 1–5）：真实语料 PPL、规模×种子网格、跨语言、真预训练 LM 融合
3. **最终配方**：`后缀多阶检索 + 证据门控 ⊕ λ 融合神经 LM logits`

### 核心结果一句话

| 断言 | 证据 |
|---|---|
| ROSA 在真实文本上不可用 | wikitext-103 上 32.5 bits/token（比 unigram 差 3 倍） |
| DPSH 检索后验优于经典 PPM | 4 格规模×种子网格全部成立（-29.6±6.5 mbits） |
| 检索⊕神经融合互补 | 中英双语、跨规模、真预训练 LM 上全部成立 |
| 检索在真预训练 LM 之上有增量 | Qwen2.5-0.5B 6.393 → 融合 **5.966** bits（+427 mbits） |

---

## 2. 目录结构

```
dpsh_experiment_pack/
├── dpsh_experiment.py        # 合成基准主实验（语料/索引/模型/评测/消融）
├── plot_results.py           # 读取 results/*.json 出对比图
├── run_all.sh                # 合成基准一键复现
├── dpsh_real_pg19.py         # Phase 1: 真实文本 PPL（wikitext-103，词级）
├── dpsh_real_phase2.py       # Phase 2: GRU 神经分支融合（GPU）
├── dpsh_real_phase3.py       # Phase 3: 规模×种子网格（{150k,500k}×{s1,s2}）
├── dpsh_real_phase4.py       # Phase 4: 中文泛化 + 自适应 μ 门控
├── dpsh_real_phase5.py       # Phase 5: 真预训练 LM（Qwen2.5-0.5B）分支融合
├── results/
│   ├── results_dense.json            # 合成-密集
│   ├── results_sparse.json           # 合成-稀疏（--full 完整版）
│   ├── results_*_baseline.json       # 修复 bug 前的基线备份
│   ├── dpsh_results.png              # 12 方法对比图
│   ├── results_real_pg19.json        # Phase 1
│   ├── results_real_phase2.json      # Phase 2
│   ├── results_real_phase3.json      # Phase 3
│   ├── results_real_phase4.json      # Phase 4
│   └── results_real_phase5.json      # Phase 5
└── logs/                     # 运行日志
```

## 3. 环境

```bash
# 合成基准 + Phase 1（纯 CPU）
pip install numpy scipy scikit-learn matplotlib

# Phase 2–4 神经分支（GPU 推荐）
pip install torch --index-url https://download.pytorch.org/whl/cu130

# Phase 5 真预训练 LM
pip install "transformers<5"   # 5.x 与 torch<=2.13 不兼容
```

中文语料（Phase 4）默认读本地 `minimind_data/pretrain_t2t_mini.jsonl`（路径在
`dpsh_real_phase4.py` 中可改），需要 `jieba`；英文语料自动从 HF（可用
`HF_ENDPOINT=https://hf-mirror.com` 镜像）流式拉取并缓存。

## 4. 快速开始

```bash
# 合成基准（约 8 分钟，CPU）
bash run_all.sh            # 基线
bash run_all.sh --full     # 追加聚类准则消融（succ/both 聚类）

# 真实链路（各 10–20 分钟）
python dpsh_real_pg19.py       # Phase 1: ROSA vs PPM vs DPSH（真实 PPL）
python dpsh_real_phase2.py     # Phase 2: + GRU 融合
python dpsh_real_phase3.py     # Phase 3: 规模×种子网格
python dpsh_real_phase4.py     # Phase 4: 中文 + 自适应 μ
python dpsh_real_phase5.py     # Phase 5: Qwen 融合（需 GPU + transformers）
```

> **注意**：合成基准的 dense/sparse 两场景必须分进程跑（`run_all.sh` 已处理）。
> `dpsh_experiment.py` 的模块级 `rng` 只在 import 时播种一次，同进程连跑会让
> 第二个场景的语料漂移。

## 5. 合成基准

### 5.1 判别性任务

| 任务 | 构造 | 意图 |
|---|---|---|
| **T1 Verbatim** | 待预测上下文在历史中精确出现过 | ROSA 主场，DPSH **不能退化** |
| **T2 Paraphrase** | 上下文换成同义 key，精确匹配断裂 | ROSA 软肋，检验语义泛化 |
| **T3 Mixed** | 通用 next-token 预测 | 综合表现 |

语料 = filler 模板段落 + 事实句 `[Q0, Q1, K, V]`。每个同义组 3 个 key，
只有 2 个在历史中作过事实句主语，剩下 1 个构成 paraphrase 测试集。
`fact_ratio` 控制证据密度：**密集** 0.32（节点平均 13.5 次观测）/
**稀疏** 0.05（2.3 次）。

公平性：所有含自由参数的方法（PPM / DPSH 各变体）在同一 dev split 上用
L-BFGS 最小化 NLL 自动拟合，不手工调参；ROSA / ROSA+WB 无参数。

### 5.2 结果（完整版，含聚类准则消融）

密集场景（acc / bits）：

| 方法 | T1 精确 | T2 改写 | T3 混合 |
|---|---|---|---|
| ROSA-hard | 100.0% / 0.00 | **0.0% / 39.86** | 55.7% / 17.65 |
| PPM | 100.0% / 0.11 | 0.0% / 11.10 | 56.4% / 3.57 |
| DPSH 硬超边 | 100.0% / 0.10 | 89.3% / 1.03 | 56.4% / 3.10 |
| DPSH 自适应(后继聚类) | 100.0% / 0.04 | 97.3% / 0.40 | 57.9% / 3.07 |
| DPSH 自适应(混合聚类) | 100.0% / 0.06 | 100.0% / 0.07 | 58.3% / 2.78 |
| DPSH w/o超边 | 100.0% / 0.04 | 100.0% / 0.08 | **58.8% / 2.89** |

稀疏场景：

| 方法 | T1 精确 | T2 改写 | T3 混合 |
|---|---|---|---|
| ROSA-hard | 85.6% / 5.74 | **0.0% / 39.86** | 57.7% / 16.86 |
| PPM | 85.6% / 2.38 | 0.0% / 14.09 | 58.0% / 3.34 |
| DPSH 硬超边 | 85.6% / 1.42 | 3.6% / 5.26 | 57.6% / 3.04 |
| DPSH 自适应(后继聚类) | 92.9% / 1.53 | 80.4% / 3.91 | 60.2% / 2.86 |
| DPSH 自适应(混合聚类) | 92.9% / 1.59 | 80.4% / 3.42 | 61.3% / 2.62 |
| DPSH w/o超边 | **92.9%** / 1.63 | 80.4% / 4.09 | **61.7% / 2.62** |

（bits 略有跨环境波动，来自 MiniBatchKMeans 的 sklearn 版本间非确定性；
参数无关方法逐位可复现。）

### 5.3 合成结论

1. **DPSH 严格优于 ROSA。** T1 不退化；T2 上 ROSA 从 100% 掉到 0%
   （39.86 bits 是概率塌到数值下限的截断值——精确匹配一断就彻底失明）。
2. **命门是语义近邻检索**：去掉近邻 → T2 归零；去掉多阶 → 70~80%；
   去掉证据门控（硬超边）→ 崩到 3.6%。
3. **超边无增量价值（负面结论）**：按后继分布/混合聚类能救回 T2，
   但 T3 始终不超过完全不用超边 → 正式设计**砍掉超边**。

## 6. 真实链路验证（Phase 1–5）

### Phase 1 — 真实文本 PPL（wikitext-103，词级 5000，书本级 held-out）

| 方法 | acc | bits/token |
|---|---|---|
| Unigram | 16.0% | 8.470 |
| ROSA-hard | 17.5% | **32.56** 💥 |
| ROSA+WB | 17.5% | 32.51 |
| PPM | 19.9% | 7.899 |
| DPSH w/o近邻 | **20.4%** | **7.850** |
| DPSH-full | 19.9% | 7.852 |

- **ROSA 判死刑**：硬复制在真实文本上"匹配到但复制错"是常态
- **DPSH 的多阶混合+证据门控有真实增量**（vs PPM +0.6%）
- **语义近邻无增量**（换更强嵌入也一样，见 Phase 2）→ 正式版砍掉

### Phase 2 — GRU 神经分支融合（检索硬 + 神经软）

| 方法 | bits/token |
|---|---|
| DPSH w/o近邻 (PPMI) | 7.850 |
| NNLM(GRU) 单独 | 8.100 |
| **DPSH ⊕ NNLM 融合** | **7.780** |

μ=0.65（dev 网格拟合，非手调），两个分支都被真实使用。
另：训练出的 NN 词嵌入替换 PPMI 后近邻检索仍无增量（7.854→7.936），
证实"近邻无增量"不是嵌入弱的问题。

### Phase 3 — 规模 × 种子网格（可靠性）

{150k, 500k} 索引 × {切片1, 切片2}：

| 格 | PPM | DPSH | NNLM | 融合 | C1 | C2 |
|---|---|---|---|---|---|---|
| 150k_s1 | 7.795 | 7.776 | 7.800 | **7.564** | ✓ | ✓ |
| 150k_s2 | 7.786 | 7.752 | 7.833 | **7.559** | ✓ | ✓ |
| 500k_s1 | 7.349 | 7.317 | 7.484 | **7.090** | ✓ | ✓ |
| 500k_s2 | 7.333 | 7.299 | 7.481 | **7.086** | ✓ | ✓ |

- **C1（DPSH<PPM）**：4/4 成立，增量 -29.6±6.5 mbits
- **C2（融合<所有单分支）**：4/4 成立，增益 211.5±12.3 mbits
- 种子间融合差异仅 ~0.005 bits；μ 稳定在 0.50–0.55

### Phase 4 — 跨语言（中文 minimind 语料，jieba 词级）+ 自适应 μ

| 语料 | PPM | DPSH | NNLM | 融合(标量μ) | C1 | C2 |
|---|---|---|---|---|---|---|
| en 500k | 7.103 | 7.072 | 7.249 | **6.855** | ✓ | ✓ |
| zh 500k | 6.266 | 6.225 | 5.967 | **5.682** | ✓ | ✓ |

**门控方向随语料自动翻转**：英文检索强 → μ=0.55；中文神经强 → μ=0.35。
dev 拟合自动找到正确混合方向，无需人工干预。
自适应 μ（sigmoid(a+b·nmax+c·mlen)）仅比标量好 ~2 mbits——诚实汇报。

### Phase 5 — 真预训练 LM 分支（Qwen2.5-0.5B-Instruct）

| 方法 | acc | bits/token |
|---|---|---|
| DPSH(检索) | 23.3% | 7.105 |
| GRU 分支 | 20.9% | 7.271 |
| Qwen2.5-0.5B 单独 | 27.9% | 6.393 |
| **融合：检索⊕Qwen** | 29.0% | **5.966** |
| **融合：检索⊕GRU⊕Qwen** | **29.3%** | **5.937** |

- **检索在真预训练 LM 之上有真实增量：+427 mbits**（6.393→5.966），
  权重 0.4/0.6（dev 拟合）
- 三路融合 0.3/0.1/0.6 再进半步
- Qwen 分布按"词首 BPE token"聚合到词级词表（近似；
  batch 推理必须左 padding，否则取到 PAD 位置的 logits）

### Phase 6 — NIAH 长上下文本尊（500k 真实草垛）

300 根真实针（独特 trigram），5 个深度带，原版+改写版：

| 深度 | ROSA | PPM | DPSH | Qwen-long(4k BPE窗) |
|---|---|---|---|---|
| 1k-5k | 100% | 98.3% | 98.3% | 12.5% |
| 20k-100k | 100% | 96.7% | 96.7% | 0% |
| **100k-500k** | **96.7%** | 95.0% | **96.7%** | **0%** |

- **深度不变性**：检索系 50 万 token 深处仍 ~97%，O(1) 查询无窗口衰减
- **窗口悬崖**：窗口系在 >窗口深度全灭

### Phase 7/8/9 — 改写鲁棒性三连（否定→修复→逼近上界）

| 方案 | 改写 Acc@1 均值 | 结论 |
|---|---|---|
| 精确检索（Phase 7 基线） | 1% | 改写即失明 |
| +语义近邻检索 | 0% | **否证**：真实嵌入近邻=相关词≠可互换 |
| +查询扩展 max聚合@5 | 28% | 扩展有效，原词恢复率@20=93% |
| 置信/最小熵选择 | 18%/8% | **否证**：自信=高频搭配≠正确稀有针 |
| **+命中优先选择@20** | **89%** | 最长精确阶+命中数是唯一可靠信号 |
| oracle（原词恢复） | 96% | 上界 = 恢复率 × verbatim 97% |

### Phase 10 — LongBench-lite 生成管线（诚实边界）

HotpotQA/2WikiMQA/MuSiQue 各 30 条，DPSH 全文索引逐 token 生成 ⊕ Qwen：

- Qwen-long EM 0-10% / F1 4-18%；DPSH 生成 F1 3-6%（能检索到正确证据段落，
  但词级贪心续写产生不了答案格式的字符串）
- 逐步 logit 融合在生成式 QA 上无增益 → **生成式任务的正确集成是
  retrieve-then-read**（DPSH 定位证据 → LM 在证据上作答），而非逐步混合

## 7. 可靠性证据链

```
合成基准      机制因果成立（T1 不退化 / T2 泛化 / 消融定位有效组件）
   ↓
Phase 1      真实文本：ROSA 死刑；DPSH > PPM
   ↓
Phase 3      规模×种子：C1/C2 4/4 格成立，方差 ~mbits 级
   ↓
Phase 4      跨语言复现；门控方向自适应
   ↓
Phase 5      真预训练 LM 之上仍有 +427 mbits 增量
   ↓
Phase 6      NIAH：500k 深度 97% 深度不变 vs 窗口系 0%
   ↓
Phase 7-9    改写鲁棒性 0% -> 89% (扩展+命中优先), 两次否定性结论
   ↓
Phase 10     LongBench-lite: 生成式任务应走 retrieve-then-read
   ↓
Phase 11     跨架构外挂: RWKV-7 (线性RNN) 上 +1311 mbits, 第一档普适
```

## 7.5 两种应用档位

**第一档：推理时外挂（已证明，零训练成本）**
LM 冻结不动，DPSH 作为外挂精确记忆接入：
- 探针/检索类任务：**logit 级融合**——DPSH 后验与 LM 逐 token 混合
  （Phase 5: Qwen2.5-0.5B +427 mbits；Phase 11: RWKV-7 +1311 mbits）
- 生成类任务：**retrieve-then-read**——DPSH 定位证据段落供 LM 作答（Phase 10 结论）
- 成本：查询 CPU 毫秒级、零 GPU 占用；任何开源 LM 即插即用

**跨架构外挂验证（Phase 11, RWKV-7 "Goose" 0.4B, 纯 PyTorch 实现零 fla/triton）**

| 方法 | acc | bits/token |
|---|---|---|
| DPSH(检索) | 23.3% | 7.105 |
| RWKV-7 0.4B 单独 | 19.9% | 7.885 |
| **融合：检索⊕RWKV-7** | **24.2%** | **6.574** |

μ=0.70（RWKV 分支窗口 32 词较弱 → 检索贡献自动放大，与 Phase 4
"门控方向随分支强弱自适应"一致）。注意 RWKV-7 是**无注意力 RNN**——
外挂融合对注意力系 (Transformer) 与线性 RNN 系架构均成立。

**第二档：训练时内嵌（Phase 12 首次验证：该规模下为负结果，边界已划定）**
把检索-门控机制做进模型内部（检索层接残差流/替代部分注意力），
让模型预训练时原生学会"精确回忆长程片段"。
Phase 4 的证据门控 λ 是内嵌形态雏形（当前与 unigram 混合，未来与模型隐状态混合）。
需要架构改造 + 十亿级 token 预训练算力（8×A100 数天-数周量级）。

**Phase 12 内嵌首验（Meta-ASH 30M 底座, 参数对齐, minimind）**：
将 DPSH 多阶序列内归纳检索 + 证据门控做成 +10 参数可训练层
（加性 logit 融合 `logits + g·λ·log P_retr`，g=0 起步与基线严格等价）：

| 语料 | META 基线 | META+DPSHGate | 增益 |
|---|---|---|---|
| minimind 通用 800docs | 3.02 | 3.33 | -0.31 nats |
| 检索密集（每篇 fact 四元组×3） | 2.78 | 2.94 | -0.16 nats |

诚实结论：**短文档（256 tok）+ 短程训练（3k 步）下，内嵌检索未兑现第一档的增益**
（第一档 +427~1311 mbits 的前提是跨文档 500k 大索引的充分统计计数；
内嵌形态只能看当前文档内的重复，证据密度低一个量级，且早期计数噪声污染训练）。
第二档的正路 = 路线图的 RWKV7+ROSA 长文档全预训练，而非小规模短训。

## 8. 工程备注（本次验证修复的 bug）

1. `--full` 模式新变体被静默丢弃（`ORDER` 在导入时固化）——这就是
   "已实现但从未跑完"的真正原因；修复后在 main 中扩展 `ORDER`
2. `--scenario plot` 必崩（`make_plot` 用 tuple 下标记问 JSON dict）——兼容两种格式
3. `/data/workspace/` Linux 硬编码路径 ×4 → 改为脚本相对路径
4. 中文字体 Linux-only → `Microsoft YaHei` 回退链
5. Phase 5：Qwen tokenizer 默认右 padding，batch 推理 `logits[:, -1]`
   取到 PAD 位置 → 左 padding 修复

## 9. 已知边界（不要过度外推）

- 真实链路是**词级 5000 词表 / ≤500k 索引**的机制级验证，不是生产规模
- Phase 5 的 Qwen 分支用首 BPE 聚合近似（全词 BPE 打分成本高）
- 单语种单语料各一（en=wikitext-103，zh=minimind）；无多种子显著性检验
  （Phase 3 的种子方差除外）
- 检索的增量部分来自**域内长史**（检索索引 vs Qwen 的 32 词窗口）；
  长上下文任务（改写版 NIAH / LongBench）是下一步
- 无安全对齐考量；合成语料只证明机制因果

## 10. 引用

```bibtex
@misc{dpsh,
  title={DPSH: Differentiable Probabilistic Suffix Hypergraph —
         检索硬、聚合软的语言模型组件及其与 ROSA 的对比},
  author={dfytensor},
  year={2026},
  url={https://github.com/dfytensor/DPSH}
}
```

## License

MIT
