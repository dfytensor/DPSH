# -*- coding: utf-8 -*-
"""
DPSH Phase 7: 改写 NIAH 的救援 — 语义近邻检索的裁决实验
背景: Phase 6 改写版全军覆没 (检索系 0%); 合成实验称语义近邻是改写命门;
      真实 PPL 中近邻无增量。NIAH-改写是两者间的裁决场景。
设计:
  - 同 Phase 6 的 300 根针 + 改写构造 (PPMI-NN 替换前缀最稀有词)
  - 索引嵌入 = GRU 训练词嵌入 (语义空间), 支持 use_nn=True 近邻检索
  - 方法:
      A. DPSH 仅精确 (use_nn=False)     —— 预期 ~0%
      B. DPSH + 语义近邻 (use_nn=True)  —— 假设: 深度无关地 >0%
      C. B ⊕ Qwen-short 融合
      D. Qwen-short 单独
      E. verbatim 对照 (A vs B 不应伤原版)
判据: B 在改写下 Acc@1 显著 > A 且深度平坦 -> 近邻检索在"检索语义泛化"场景有真实价值
"""
import os, sys, io, json, math, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
OUT = os.path.join(PROJ, "results", "results_real_phase7_niah_nn.json")
SEED = 20260912
np.random.seed(SEED); torch.manual_seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
V, SCALE = 5000, 500000
T0 = time.time()

from dpsh_real_pg19 import tokenize as en_tokenize

BINS = [(0, 1000), (1000, 5000), (5000, 20000), (20000, 100000), (100000, 500000)]
BIN_NAMES = ["0-1k", "1k-5k", "5k-20k", "20k-100k", "100k-500k"]

class NNLM(nn.Module):
    def __init__(self, V, d=128, h=320):
        super().__init__()
        self.emb = nn.Embedding(V, d, padding_idx=0)
        self.gru = nn.GRU(d, h, num_layers=2, batch_first=True, dropout=0.1)
        self.ln = nn.LayerNorm(h)
        self.head = nn.Linear(h, V, bias=False)
    def forward(self, x):
        out, _ = self.gru(self.emb(x))
        return self.head(self.ln(out[:, -1]))

CTX = 16
def train_nnlm(tr, epochs=4):
    xs = np.lib.stride_tricks.sliding_window_view(tr, CTX + 1)
    X = torch.tensor(np.array(xs[:, :-1]), dtype=torch.long)
    Y = torch.tensor(np.array(xs[:, -1]), dtype=torch.long)
    m = NNLM(V).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bs = 1024
    for ep in range(epochs):
        perm = torch.randperm(len(X))
        tot, nb = 0.0, 0
        for i in range(0, len(X), bs):
            b = perm[i:i + bs]
            loss = nn.functional.cross_entropy(m(X[b].to(DEV)), Y[b].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        print(f"    epoch {ep+1}: loss {tot/nb:.4f} ({time.time()-T0:.0f}s)", flush=True)
    return m

def main():
    import dpsh_experiment as dp
    # ---- 数据 (与 phase5/6 同切片) ----
    books = json.load(open(os.path.join(PROJ, "results", "pg19_text_cache.json"), encoding="utf-8"))
    from collections import Counter
    train_t = en_tokenize(books[:40])[:SCALE]
    cnt = Counter(train_t)
    vocab = ["<unk>"] + [w for w, _ in cnt.most_common(V - 1)]
    wid = {w: i for i, w in enumerate(vocab)}
    tr = np.array([wid.get(w, 0) for w in train_t], dtype=np.int64)
    print(f"[数据] {len(tr):,} tok ({time.time()-T0:.0f}s)", flush=True)

    # ---- 针采样 (与 phase6 相同逻辑) ----
    freq = Counter(tr.tolist())
    tri_pos = {}
    for i in range(3, len(tr)):
        key = (int(tr[i - 2]), int(tr[i - 1]), int(tr[i]))
        tri_pos.setdefault(key, []).append(i)
    uniq = {k: v[0] for k, v in tri_pos.items() if len(v) == 1}
    cand = [(d, k) for k, d in uniq.items() if 0 not in k and 2 <= freq[k[2]] <= 50]
    needles = {}
    for b, (lo, hi) in enumerate(BINS):
        pool = [(d, k) for d, k in cand if lo <= d < hi]
        np.random.shuffle(pool)
        needles[b] = pool[:60]
    print(f"[针] {[len(needles[b]) for b in range(5)]}", flush=True)

    # ---- GRU 语义嵌入 ----
    print("[GRU] 训练语义嵌入 ...", flush=True)
    m = train_nnlm(tr); m.eval()
    E = m.emb.weight.detach().cpu().numpy().astype(np.float32)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    E_ppmi_for_swap = None  # 改写替换用 PPMI-NN (与 phase6 一致)
    from dpsh_real_pg19 import build_embeddings_sparse
    E_ppmi_for_swap = build_embeddings_sparse(list(map(int, tr)), vocab, dim=64)

    dp.VOCAB = V; dp.MIN_COUNT = 4
    ix = dp.Index(list(map(int, tr)), E, decay=0.7, K=2500)
    print(f"[索引] 节点 {len(ix.nodes):,} ({time.time()-T0:.0f}s)", flush=True)

    # ---- 拟合 (dev 常规流, 与 phase6 同) ----
    dev_idx = np.linspace(6, 25000 - 2, 800).astype(int)
    dev_s = [(tr[i - 5:i], int(tr[i])) for i in dev_idx]
    init0 = np.r_[1.0, 0.5, np.zeros(dp.L_MAX + 1), 0.0, 0.5, 1.0, 2.0]
    def fit_chunked(feats, chunk=250, iters=20):
        B = [dp.make_batch(feats[i:i + chunk]) for i in range(0, len(feats), chunk)]
        ns = np.array([len(b["tgt"]) for b in B])
        f = lambda x: sum(dp.batch_forward(b, x, ix.unigram, "adaptive")[1] * n
                          for b, n in zip(B, ns)) / ns.sum()
        return minimize(f, init0, method="L-BFGS-B",
                        options={"maxiter": iters, "maxfun": iters * 5}).x
    def mk(data, use_nn):
        feats = []
        for ctx, tgt in data:
            c = dp.select_multi_order(ix.query(list(ctx), C=6, S=4, use_nn=use_nn), True)
            if not c:
                c = [(ix.unigram, ix.unigram, dp.L_MAX, 0.0, 0.0)]
            F, cnts = dp.featurize(c)
            feats.append((c, F, cnts, dp.meta_of(c), tgt))
        return feats
    th = fit_chunked(mk(dev_s, use_nn=False))
    th_nn = fit_chunked(mk(dev_s, use_nn=True))

    from dpsh_real_phase6_niah import qwen_next_word_probs  # 复用 (模型缓存全局)

    results = {}
    for variant in ["verbatim", "paraphrase"]:
        print(f"\n===== {variant} =====", flush=True)
        rows = {}
        for b in range(5):
            pool = needles[b]
            if not pool:
                continue
            ctx5, tgts, depths = [], [], []
            for d, (a, bb, c) in pool:
                ctx = [a, bb]
                if variant == "paraphrase":
                    freqs = [ix.unigram[a], ix.unigram[bb]]
                    j = int(np.argmin(freqs))
                    orig = ctx[j]
                    sims = E_ppmi_for_swap @ E_ppmi_for_swap[orig]
                    sims[orig] = -1
                    ctx[j] = int(np.argmax(sims))
                f5 = list(tr[max(0, d - 5):d])
                f5[-2], f5[-1] = ctx[0], ctx[1]
                ctx5.append(f5); tgts.append(c); depths.append(d)
            tgts = np.array(tgts)

            def probs(ctx_list, use_nn, theta):
                out = []
                for c in ctx_list:
                    cand = dp.select_multi_order(ix.query(list(c), C=6, S=4, use_nn=use_nn), True)
                    if not cand:
                        out.append(ix.unigram); continue
                    F, cnts = dp.featurize(cand)
                    p, _ = dp.score(cand, F, cnts, dp.meta_of(cand), theta, ix.unigram, "adaptive")
                    out.append(p)
                return np.array(out)
            P_exact = probs(ctx5, False, th)
            P_nn = probs(ctx5, True, th_nn)
            texts_short = [" ".join(vocab[t] for t in tr[max(0, d - 32):d]) for d in depths]
            P_q = qwen_next_word_probs(texts_short, vocab)
            P_fuse = 0.4 * P_nn + 0.6 * P_q
            def a1(P): return float(np.mean(np.argmax(P, 1) == tgts))
            row = {"DPSH仅精确": a1(P_exact), "DPSH+语义近邻": a1(P_nn),
                   "近邻⊕Qwen融合": a1(P_fuse), "Qwen-short": a1(P_q)}
            rows[BIN_NAMES[b]] = row
            print(f"  {BIN_NAMES[b]:<10} " + "  ".join(
                f"{k}={v*100:.0f}%" for k, v in row.items()), flush=True)
        results[variant] = rows

    json.dump(results, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n已保存:", OUT, flush=True)

    for variant in results:
        print(f"\n[{variant}] Acc@1 (%)")
        ms = ["DPSH仅精确", "DPSH+语义近邻", "近邻⊕Qwen融合", "Qwen-short"]
        print(f"{'深度':<12}" + "".join(f"{m:>16}" for m in ms))
        for bn in BIN_NAMES:
            if bn in results[variant]:
                r = results[variant][bn]
                print(f"{bn:<12}" + "".join(f"{r[m]*100:15.1f}%" for m in ms))
    print("\n[裁定]")
    para = results["paraphrase"]
    nn_deep = np.mean([para[bn]["DPSH+语义近邻"] for bn in BIN_NAMES if bn in para])
    ex_deep = np.mean([para[bn]["DPSH仅精确"] for bn in BIN_NAMES if bn in para])
    print(f"  改写全深度均值: 仅精确 {ex_deep*100:.1f}% vs +语义近邻 {nn_deep*100:.1f}% "
          f"({'近邻有真实救援 ✓' if nn_deep > ex_deep + 0.05 else '近邻仍无效'})")
    verb = results["verbatim"]
    nv = np.mean([verb[bn]["DPSH+语义近邻"] for bn in BIN_NAMES if bn in verb])
    print(f"  verbatim 对照: +近邻 {nv*100:.1f}% (不伤原版)" )

if __name__ == "__main__":
    main()
