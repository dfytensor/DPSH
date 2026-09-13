#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第二档验证: DPSH 机制内嵌 Meta-ASH 30M — 端到端训练的检索-门控层 (Phase 12)
第二档定义 (README 7.5): 把 DPSH 的"多阶后缀检索 + 证据门控"做进模型内部,
让模型预训练时原生学会"精确回忆长程片段"。

DPSHGate (可训练内嵌层, 仅 +9 参数):
  位置 t 的 logits 预测 x[t+1]; 检索窗 = x[t-l:t] (l=1..L 阶), 在序列内部
  找更早的相同窗口 (t' < t), 其后继 x[t'+1] 即归纳候选 (序列内归纳检索, 无外部索引):
    n_l[t]    = #{t' : x[t-l:t] == x[t'-l:t']}                    (该阶证据量)
    succ_l[t] = #{t' : 同上且 x[t'+1] == x[t+1]}                   (目标后继命中数)
    P_retr[t] = (Σ_l w_l·succ_l + α) / (Σ_l w_l·n_l + α·V)         w_l 可学习 (多阶混合)
    λ[t]      = sigmoid(bias + Σ_l c_l·log1p(n_l))                (证据门控, DPSH λ 内嵌)
    logP[t]   = log[(1-λ)·softmax(logits[t])[x[t+1]] + λ·P_retr[t]]
  梯度: 模型经 (1-λ) 项学习"把可检索位置让给检索"; 门控/阶权重端到端学习。

对照 (minimind 800docs×256, 种子 42, 3000 步, Meta-ASH 30M 底座):
  A. META 基线 (29.6M, hrc_validate 报告 final 2.870)
  B. META + DPSHGate (29.6M + 9 参数)  <- 第二档内嵌
诊断: fireable (序列内存在精确重复窗) 位置 loss 分解 + λ 均值。
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, r"F:\OpenASH2605")
sys.path.insert(0, r"F:\OpenASH2605\hrc_validate")
sys.path.insert(0, r"F:\OpenASH2605\copyfirst_redesign")

import torch
import torch.nn as nn
import meta_ash_30m as base
from meta_ash_30m import CLM, MetaMaxState30, build_docs, VOCAB, D, HEADS, DEV

OUT = r"F:\夸克\dpsh_experiment_pack\dpsh_experiment_pack\results\results_tier2_dpshgate.json"
L_ORD, ALPHA = 4, 1e-3


class DPSHGate(nn.Module):
    """可学习证据门控 + 多阶序列内归纳检索 (第二档内嵌层, 9 参数)."""

    def __init__(self, vocab, L=L_ORD, alpha=ALPHA):
        super().__init__()
        self.L = L
        self.alpha = alpha
        self.V = vocab
        self.w_ord = nn.Parameter(torch.zeros(L))            # 多阶混合, softmax 均匀起步
        self.evidence = nn.Parameter(torch.zeros(L))         # 证据门控系数
        self.gate_bias = nn.Parameter(torch.tensor(-1.0))    # λ 初始 ≈ 0.27
        self.g_scale = nn.Parameter(torch.tensor(0.0))       # 全局融合系数, 0 = 基线等价起步

    def forward(self, ids, logits):
        """加性 logit 融合 (第一档 Phase5/11 的训练版):
        logits_final = logits + g·λ_t·log P_retr
        g 初始 0 -> 与基线完全等价起步; λ_t = 证据门控 (可学习); P_retr = 多阶检索后验."""
        b, s = ids.shape
        dev = ids.device
        causal = torch.ones(s, s, device=dev, dtype=torch.bool).tril(-1)  # t' < t
        onehot = torch.zeros(b, s, self.V, device=dev, dtype=torch.float16)
        onehot.scatter_(2, ids.clamp(min=0).unsqueeze(-1), 1.0)
        n_list, succ_list = [], []
        succ_total = torch.zeros(b, s, self.V, device=dev, dtype=torch.float16)
        for l in range(1, self.L + 1):
            mm = torch.ones(b, s, s, device=dev, dtype=torch.bool)
            for j in range(l):
                sh = torch.cat([torch.full_like(ids[:, :j + 1], -1), ids[:, :-(j + 1)]], 1)
                mm = mm & (sh[:, :, None] == sh[:, None, :]) \
                     & (sh[:, :, None] > 0) & (sh[:, None, :] > 0)
            mm = mm & causal
            n_list.append(mm.sum(-1).float())
            succ_l = torch.bmm(mm.to(torch.float16), onehot)  # [b,s,V] 各候选后继计数
            succ_list.append(mm.sum(-1).float())
            w = torch.softmax(self.w_ord, 0)[l - 1]
            succ_total = succ_total + w * succ_l
        n_stack = torch.stack(n_list, -1)                      # [b,s,L]
        n_tot = succ_total.sum(-1, keepdim=True)               # [b,s,1]
        log_p_retr = (succ_total + 1e-3).log() - (n_tot + 1e-3 * self.V).log()
        lam = torch.sigmoid(self.gate_bias + n_stack.log1p() @ self.evidence)  # [b,s]
        logits_final = logits + self.g_scale * lam.unsqueeze(-1) * log_p_retr.float()
        fireable = (n_stack.max(-1).values >= 1)
        return logits_final, lam, fireable


class CLMGate(CLM):
    """CLM + DPSHGate (加性 logit 融合)."""

    def __init__(self, attn_cls):
        super().__init__(attn_cls)
        self.gate = DPSHGate(VOCAB)

    def forward(self, x, targets=None):
        import torch.nn.functional as F
        h = self.em(x)
        state = [None] * len(self.decoder_layers)
        for i, layer in enumerate(self.decoder_layers):
            h1, state[i] = layer(h, state[i])
            h = h1 + h
        logits = self.head(h)
        if targets is None:
            return logits, None
        logits_f, lam, fireable = self.gate(x, logits)
        V = logits_f.shape[-1]
        loss = F.cross_entropy(logits_f[:, :-1].reshape(-1, V),
                               targets[:, 1:].reshape(-1), ignore_index=0)
        # 诊断: 逐位置 CE 分解 (fireable / non)
        lg = logits_f[:, :-1]
        tg = targets[:, 1:]
        ce_vec = -torch.log_softmax(lg, -1).gather(-1, tg.clamp(min=0).unsqueeze(-1)).squeeze(-1)
        valid = (tg > 0).float()
        fire_d = fireable[:, :-1]  # 位置 t 的检索服务于 t+1, 与 CE 对齐 (长度 s-1)
        return logits, (loss, ce_vec * valid, valid, lam, fire_d)


def train_gate(model, docs, steps=3000, bs=8, seed=42, tag=""):
    torch.manual_seed(seed)
    np.random.seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=3e-5)
    # 分组裁剪: 融合损失的 1/p_mix 乘子会让门控梯度比骨干大 2-3 个量级,
    # 全局统一裁剪会把骨干梯度吞掉 (第一次运行的教训)
    gate_params = [p for p in model.gate.parameters()]
    model_params = [p for n, p in model.named_parameters() if not n.startswith("gate.")]
    model.train()
    t0 = time.time()
    for st in range(steps):
        idx = np.random.randint(0, len(docs), bs)
        x = torch.cat([docs[i] for i in idx])
        logits, (loss, loss_vec, valid, lam, fireable) = model(x, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_params, 1.0)
        torch.nn.utils.clip_grad_norm_(gate_params, 1.0)
        opt.step()
        sched.step()
        if st % 500 == 0 or st == steps - 1:
            with torch.no_grad():
                fv = fireable.float() * valid
                fl = (loss_vec * fv).sum() / fv.sum().clamp(min=1.0)
                nf = (loss_vec * (1 - fv) * valid).sum() / ((1 - fireable.float()) * valid).sum().clamp(min=1.0)
                lam_mean = (lam[:, :-1] * valid).sum() / valid.sum().clamp(min=1.0)
                fire_rate = fv.sum() / valid.sum().clamp(min=1.0)
            print("  [%s] step %d loss=%.4f | fireable=%.4f non=%.4f | lam=%.3f fire%%=%.1f%% (%.0fs)"
                  % (tag, st, loss.item(), fl.item(), nf.item(), lam_mean.item(),
                     fire_rate.item() * 100, time.time() - t0), flush=True)
    return loss.item()


def build_docs_fact():
    """检索密集型文档: minimind 文本 + 每篇插入 3 次重复的事实四元组 [K1,K2,K3,V].
    V 为每篇唯一的高位 id; 检验内嵌检索对'序列内真复现'的增益 (第二档 NIAH)."""
    import meta_ash_30m as base_mod
    from open_ash_voc import OpenASHVoc
    voc = OpenASHVoc(agent_voc_path=r"F:\OpenASH2605\open_ash_voc_agent.json")
    texts = []
    for p in [r"F:\OpenASH2605\minimind_data\pretrain_t2t_mini.jsonl",
              r"F:\OpenASH2605\minimind_data\sft_t2t_mini.jsonl"]:
        with open(p, encoding="utf-8") as f:
            for line in f:
                try:
                    dd = json.loads(line)
                    t = dd.get("text", "")
                    if not t and "conversations" in dd:
                        t = "".join(x.get("content", "") for x in dd["conversations"])
                    if len(t) > 500:
                        texts.append(t.replace("\n", ""))
                except Exception:
                    pass
                if len(texts) >= 3000:
                    break
    seq_len = 256
    docs = []
    for di, txt in enumerate(texts[:800]):
        ids = voc.encode(txt)[:seq_len]
        ids = [max(0, min(x, VOCAB - 1)) for x in ids]
        fact = [20000 + di, 20500 + di, 21000 + di, 21500 + di]  # 每篇唯一四元组
        for _ in range(3):                       # 文内重复 3 次
            pos = len(ids) // 4
            ids = ids[:pos] + fact + ids[pos:]
        ids = ids[:seq_len]
        if len(ids) < seq_len:
            ids = ids + [0] * (seq_len - len(ids))
        docs.append(torch.tensor([ids], device=DEV))
    return docs


def main():
    print("build docs...", flush=True)
    if os.environ.get("FACT_DOCS", "0") == "1":
        docs = build_docs_fact()
        print("检索密集型文档 (fact 四元组 x3)", flush=True)
    else:
        docs = build_docs()
    print("docs:", len(docs), flush=True)

    results = {}
    skip_base = os.environ.get("SKIP_BASELINE", "0") == "1"
    runs = ([("META+DPSHGate(第二档)", lambda: CLMGate(MetaMaxState30))] if skip_base else
            [("META基线", lambda: CLM(MetaMaxState30)),
             ("META+DPSHGate(第二档)", lambda: CLMGate(MetaMaxState30))])
    for name, ctor in runs:
        m = ctor().to(DEV)
        n_params = sum(p.numel() for p in m.parameters())
        n_gate = sum(p.numel() for p in m.gate.parameters()) if hasattr(m, "gate") else 0
        print("%s params: %.1fM (+%d gate params)" % (name, n_params / 1e6, n_gate), flush=True)
        t0 = time.time()
        if hasattr(m, "gate"):
            final = train_gate(m, docs, tag=name)
        else:
            final = base.train(m, docs, tag=name)
        print("%s FINAL LOSS = %.4f (%.0fs)" % (name, final, time.time() - t0), flush=True)
        results[name] = {"final_loss": final, "params": n_params, "gate_params": n_gate}
        del m
        torch.cuda.empty_cache()

    json.dump(results, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("已保存:", OUT)
    a = 3.0203 if skip_base else results.get("META基线", {}).get("final_loss", 3.0203)
    b = results["META+DPSHGate(第二档)"]["final_loss"]
    print("\n[第二档裁定]")
    print(f"  META 基线      : {a:.4f} (hrc_validate 报告 2.870)")
    print(f"  META+DPSHGate  : {b:.4f}")
    print(f"  内嵌增益        : {a-b:+.4f} nats ({(a-b)/a*100:+.1f}%)")

if __name__ == "__main__":
    main()
