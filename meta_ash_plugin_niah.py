# -*- coding: utf-8 -*-
"""
Phase 13: 插件式验证 — 冻结 Meta-ASH 30M + DPSH 外挂, NIAH 针测验
正确口径 (应指正): 插件不看 loss, 看任务能力增益。
协议:
  - 底座: MetaMaxState30 (29.6M) 在 minimind 800docs 上常规预训练 (无任何针), 冻结
  - 评测: 60 个 NIAH 样本 — 未见过文档中插入随机事实 [K1,K2,K3,V], 文末给查询 [K1,K2,K3]
  - 插件: DPSH 检索后验 (对可见上下文的精确后缀匹配, 无训练) ⊕ 模型 logits
  - 扫融合权重 mu ∈ {0, 0.05, ..., 1.0} × 针深度 {60, 128, 196}
判据: mu=0 (裸模型) acc≈随机; mu 增大 -> acc 单调升到 ~100% (插件给冻结模型装上精确回忆)
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, r"F:\OpenASH2605")
sys.path.insert(0, r"F:\OpenASH2605\hrc_validate")
sys.path.insert(0, r"F:\OpenASH2605\copyfirst_redesign")

import torch
import torch.nn.functional as F
from meta_ash_30m import CLM, MetaMaxState30, build_docs, VOCAB, D, HEADS, DEV

OUT = r"F:\夸克\dpsh_experiment_pack\dpsh_experiment_pack\results\results_tier1_plugin_niah.json"
CKPT = r"F:\夸克\dpsh_experiment_pack\dpsh_experiment_pack\results\metaash30m_minimind.pt"
SEED = 42

# ---------------- 底座训练 (常规预训练, 无针) ----------------

def get_model():
    m = CLM(MetaMaxState30).to(DEV)
    if os.path.exists(CKPT):
        m.load_state_dict(torch.load(CKPT, map_location=DEV))
        print("[底座] 加载缓存 checkpoint", flush=True)
        return m
    docs = build_docs()
    print(f"[底座] 常规预训练 3000 步 ({len(docs)} docs, 无针) ...", flush=True)
    base_train(m, docs)
    torch.save(m.state_dict(), CKPT)
    return m

def base_train(m, docs, steps=3000, bs=8, seed=SEED):
    torch.manual_seed(seed); np.random.seed(seed)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps, eta_min=3e-5)
    m.train()
    t0 = time.time()
    for st in range(steps):
        idx = np.random.randint(0, len(docs), bs)
        x = torch.cat([docs[i] for i in idx])
        _, loss = m(x, x)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sched.step()
        if st % 1000 == 0 or st == steps - 1:
            print(f"  [底座] step {st} loss={loss.item():.4f} ({time.time()-t0:.0f}s)", flush=True)

# ---------------- NIAH 构造 ----------------

def build_niah(texts, n=60, doc_len=256):
    """未见过文本 + 随机针 [K1,K2,K3,V] 插入深度 d, 文末查询 [K1,K2,K3]."""
    samples = []
    depths = [60, 128, 196]
    rng = np.random.default_rng(SEED)
    for i in range(n):
        t = texts[i]
        ids = []
        for chunk in t.replace("\n", "").split("。"):
            ids += [max(1, min(x, VOCAB - 1)) for x in _enc(chunk)]
            if len(ids) >= doc_len:
                break
        ids = ids[:doc_len]
        d = depths[i % 3]
        k1, k2, k3 = 22000 + i, 22100 + i, 22200 + i
        v = 22300 + i
        needle = [k1, k2, k3, v]
        full = ids[:d] + needle + ids[d:]
        full = full[:doc_len - 3]
        inp = full + [k1, k2, k3]                     # 文末查询
        samples.append({"ids": inp, "target": v, "depth": d, "k": (k1, k2, k3)})
    return samples

def _enc(text):
    import jieba
    jieba.setLogLevel(60)
    from open_ash_voc import OpenASHVoc
    global _VOC
    if _VOC is None:
        _VOC = OpenASHVoc(agent_voc_path=r"F:\OpenASH2605\open_ash_voc_agent.json")
    return _VOC.encode(text)

_VOC = None

# ---------------- DPSH 插件 (对可见上下文的精确检索, 无训练) ----------------

def plugin_posterior(ids, query_k, V):
    """查询 k 在可见上下文中的精确后缀匹配 -> (P_retr 向量, 命中数). 最长阶优先."""
    s = len(ids)
    for l in range(3, 0, -1):
        key = tuple(query_k[-l:])
        hits = []
        for t in range(l, s - 3 + 1):          # 查询区之前的出现
            if tuple(ids[t - l:t]) == key:
                hits.append(t)
        if hits:
            cnt = {}
            for t in hits:                     # 后继 = 窗口后紧跟的 token ids[t]
                if t < s:
                    cnt[ids[t]] = cnt.get(ids[t], 0) + 1
            P = np.full(V, 1e-3 / V, dtype=np.float32)
            tot = sum(cnt.values())
            for tok, c in cnt.items():
                P[tok] = c / tot
            return P, len(hits), l
    P = np.full(V, 1.0 / V, dtype=np.float32)
    return P, 0, 0

# ---------------- 主流程 ----------------

def main():
    print("[1] 底座 ...", flush=True)
    m = get_model()
    m.eval()

    print("[2] 构造未见过文本库 ...", flush=True)
    json_path = r"F:\OpenASH2605\minimind_data\pretrain_t2t_mini.jsonl"
    texts = []
    with open(json_path, encoding="utf-8") as f:
        for line in f:
            try:
                t = json.loads(line).get("text", "")
                if len(t) > 800:
                    texts.append(t)
            except Exception:
                pass
            if len(texts) >= 900:
                break
    eval_texts = texts[850:850 + 60]           # 训练用前 800+, 这些没见过
    samples = build_niah(eval_texts, n=min(60, len(eval_texts)))
    print(f"  {len(samples)} 个 NIAH 样本 (深度 60/128/196 各 20)", flush=True)

    print("[3] 评测: 裸模型 vs +DPSH插件 ...", flush=True)
    mus = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    acc = {mu: [] for mu in mus}
    acc5 = {mu: [] for mu in mus}
    depth_acc = {d: {0.0: [], 0.3: [], 1.0: []} for d in (60, 128, 196)}
    for si, sp in enumerate(samples):
        ids = torch.tensor([sp["ids"]], device=DEV)
        with torch.no_grad():
            logits, _ = m(ids, ids)
        lg = logits[0, -1].float()                      # 最后位置预测 (查询后继)
        P_retr, nhits, order = plugin_posterior(sp["ids"], sp["k"], VOCAB)
        p_lm = torch.softmax(lg, -1).cpu().numpy()
        tgt = sp["target"]
        for mu in mus:
            P = (1 - mu) * p_lm + mu * P_retr
            top = np.argsort(-P)[:5]
            acc[mu].append(int(top[0] == tgt))
            acc5[mu].append(int(tgt in top))
        for d in (60, 128, 196):
            if sp["depth"] == d:
                for mu in (0.0, 0.3, 1.0):
                    P = (1 - mu) * p_lm + mu * P_retr
                    depth_acc[d][mu].append(int(np.argmax(P) == tgt))
        if si < 3:
            print(f"  #{si} depth={sp['depth']} hits={nhits}(l={order}) "
                  f"裸rank={int((lg.cpu().numpy() > lg.cpu().numpy()[tgt]).sum())+1}", flush=True)

    print("\n" + "=" * 60)
    print("Phase 13 — 插件式 NIAH (冻结 Meta-ASH 30M, 未见针)")
    print(f"{'mu':>6}{'Acc@1':>10}{'Acc@5':>10}")
    print("-" * 60)
    for mu in mus:
        print(f"{mu:>6.2f}{np.mean(acc[mu])*100:9.1f}%{np.mean(acc5[mu])*100:9.1f}%")
    print("=" * 60)
    for d in (60, 128, 196):
        r = depth_acc[d]
        print(f"深度{d:<5} 裸={np.mean(r[0.0])*100:.0f}%  mu=0.3={np.mean(r[0.3])*100:.0f}%  "
              f"mu=1.0(纯检索)={np.mean(r[1.0])*100:.0f}%")

    results = {"acc_by_mu": {str(mu): float(np.mean(acc[mu])) for mu in mus},
               "acc5_by_mu": {str(mu): float(np.mean(acc5[mu])) for mu in mus},
               "depth": {str(d): {str(mu): float(np.mean(depth_acc[d][mu]))
                                  for mu in (0.0, 0.3, 1.0)} for d in (60, 128, 196)}}
    json.dump(results, open(OUT, "w", encoding="utf-8"), indent=2)
    print("已保存:", OUT)
    a0 = np.mean(acc[0.0]) * 100
    a3 = np.mean(acc[0.3]) * 100
    a1 = np.mean(acc[1.0]) * 100
    print(f"\n[裁定] 裸模型 {a0:.1f}% -> +插件(μ=0.3) {a3:.1f}% | 纯检索 {a1:.1f}% "
          f"-> {'插件给冻结 30M 装上精确回忆 ✓' if a3 > a0 + 30 else '无效'}")

if __name__ == "__main__":
    main()
