# -*- coding: utf-8 -*-
"""
DPSH Phase 2: 神经分支融合 (检索硬 + 神经软) — 真实价值放大实验

Phase 1 结论: DPSH(检索后验) 7.850 < PPM 7.899 bits, 但近邻检索无增量, λ 门控只混 unigram。
本阶段按 DPSH 设计意图补上神经分支:
  1. GPU 训练 GRU-NNLM (同 150k train, 词表 5000) -> P_nn
  2. 用 NNLM 训练出的词嵌入替换 PPMI+SVD 重建索引 -> 检验"弱嵌入拖累近邻"假设
  3. P_final = mu * P_retrieval + (1-mu) * softmax(logits/T), mu,T 在 dev 网格拟合
裁定: 融合 bits < max(单分支) 且 mu 在中间 -> 检索与神经互补, 价值放大成立
"""
import os, sys, io, json, math, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import torch
import torch.nn as nn

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
NNLM_PATH = os.path.join(PROJ, "results", "nnlm_wt103.pt")

SEED = 20260912
torch.manual_seed(SEED)
np.random.seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"

CTX = 16
V = 5000

# ---------------- 1. 数据 (与 Phase 1 完全一致的切分/词表) ----------------

from dpsh_real_pg19 import tokenize, build_embeddings_sparse

def build_data():
    books = json.load(open(os.path.join(PROJ, "results", "pg19_text_cache.json"), encoding="utf-8"))
    from collections import Counter
    train_t = tokenize(books[:40])
    cnt = Counter(train_t)
    vocab = ["<unk>"] + [w for w, _ in cnt.most_common(V - 1)]
    wid = {w: i for i, w in enumerate(vocab)}
    enc = lambda ts: [wid.get(w, 0) for w in ts]
    train = np.array(enc(train_t)[:150000])
    dev = np.array(enc(tokenize(books[40:48]))[:25000])
    test = np.array(enc(tokenize(books[48:]))[:60000])
    return train, dev, test

def stream(data, max_n, ctx=5):
    idx = np.linspace(ctx, len(data) - 2, min(max_n, len(data) - ctx - 1)).astype(int)
    return [(data[i - ctx:i], int(data[i])) for i in idx]

# ---------------- 2. NNLM (GPU) ----------------

class NNLM(nn.Module):
    def __init__(self, V, d=128, h=256):
        super().__init__()
        self.emb = nn.Embedding(V, d, padding_idx=0)
        self.gru = nn.GRU(d, h, batch_first=True)
        self.ln = nn.LayerNorm(h)
        self.head = nn.Linear(h, V, bias=False)

    def forward(self, x):
        out, _ = self.gru(self.emb(x))
        return self.head(self.ln(out[:, -1]))          # (B,V) 最后位置 logits

def train_nnlm(train):
    xs, ys = [], []
    for i in range(CTX, len(train)):
        xs.append(train[i - CTX:i]); ys.append(train[i])
    X = torch.tensor(np.array(xs), dtype=torch.long)
    Y = torch.tensor(np.array(ys), dtype=torch.long)
    m = NNLM(V).to(DEV)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    bs = 1024
    for ep in range(3):
        perm = torch.randperm(len(X))
        tot, nb = 0.0, 0
        for i in range(0, len(X), bs):
            b = perm[i:i + bs]
            logits = m(X[b].to(DEV))
            loss = nn.functional.cross_entropy(logits, Y[b].to(DEV))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        print(f"  epoch {ep+1}: loss {tot/nb:.4f}")
    return m

@torch.no_grad()
def nnlm_probs(m, ids_ctx, T=1.0):
    """ids_ctx: (N, CTX) -> (N, V) 概率"""
    out = []
    for i in range(0, len(ids_ctx), 4096):
        x = torch.tensor(np.array(ids_ctx[i:i + 4096]), dtype=torch.long, device=DEV)
        out.append(torch.softmax(m(x) / T, -1).cpu().numpy())
    return np.concatenate(out)

# ---------------- 3. 主流程 ----------------

def main():
    import dpsh_experiment as dp
    t0 = time.time()
    train, dev, test = build_data()
    dev_s = stream(dev, 1500)
    test_s = stream(test, 4000)
    print(f"[数据] train {len(train):,} | dev {len(dev_s)} | test {len(test_s)} ({time.time()-t0:.0f}s)")

    # ---- 2. NNLM 训练/加载 ----
    if os.path.exists(NNLM_PATH):
        print("[NNLM] 加载缓存权重")
        m = NNLM(V).to(DEV); m.load_state_dict(torch.load(NNLM_PATH, map_location=DEV))
    else:
        print(f"[NNLM] GPU 训练 GRU {V} 词表 ...")
        m = train_nnlm(train)
        torch.save(m.state_dict(), NNLM_PATH)
    m.eval()

    def probs_of(data_s):
        return nnlm_probs(m, [c for c, _ in data_s])
    Pnn_dev, Pnn_test = probs_of(dev_s), probs_of(test_s)

    def bits_of(P, data_s):
        tgt = np.array([t for _, t in data_s])
        p = np.clip(P[np.arange(len(tgt)), tgt], 1e-12, 1.0)
        acc = float(np.mean(np.argmax(P, 1) == tgt))
        return acc, float(-np.log(p).mean() / math.log(2))
    acc_n, bits_n = bits_of(Pnn_test, test_s)
    print(f"[NNLM] test: {acc_n*100:.1f}% / {bits_n:.3f} bits ({time.time()-t0:.0f}s)")

    # ---- 3. 训练嵌入重建索引 ----
    dp.VOCAB = V; dp.MIN_COUNT = 3
    E_nn = m.emb.weight.detach().cpu().numpy().astype(np.float32)
    E_nn = E_nn / (np.linalg.norm(E_nn, axis=1, keepdims=True) + 1e-9)
    print("[索引] NN 嵌入重建后缀索引 ...")
    ix = dp.Index(train, E_nn, decay=0.7, K=1200)
    print(f"  节点 {len(ix.nodes):,} K={ix.K} ({time.time()-t0:.0f}s)")

    N_THETA = 2 + (dp.L_MAX + 1) + 3 + 1
    init0 = np.r_[1.0, 0.5, np.zeros(dp.L_MAX + 1), 0.0, 0.5, 1.0, 2.0]
    from scipy.optimize import minimize
    def fit_chunked(feats, mode, chunk=250, iters=25):
        B = [dp.make_batch(feats[i:i + chunk]) for i in range(0, len(feats), chunk)]
        ns = np.array([len(b["tgt"]) for b in B])
        def obj(x):
            return sum(dp.batch_forward(b, x, ix.unigram, mode)[1] * n
                       for b, n in zip(B, ns)) / ns.sum()
        return minimize(obj, init0, method="L-BFGS-B",
                        options={"maxiter": iters, "maxfun": iters * 5}).x

    def mk(data, use_nn):
        feats = []
        for ctx, tgt in data:
            c = dp.select_multi_order(ix.query(ctx, C=6, S=4, use_nn=use_nn), True)
            if not c:
                c = [(ix.unigram, ix.unigram, dp.L_MAX, 0.0, 0.0)]
            F, cnts = dp.featurize(c)
            feats.append((c, F, cnts, dp.meta_of(c), tgt))
        return feats

    results = {"NNLM(GRU)": (acc_n, bits_n),
               "PPM[phase1]": (0.199, 7.899),
               "DPSH w/o近邻[phase1,PPMI]": (0.204, 7.850)}
    P_r_dev = P_r_test = None
    for name, use_nn in {"DPSH w/o近邻(NN emb)": False, "DPSH-full(NN emb)": True}.items():
        print(f"[fit] {name} ...")
        fdev = mk(dev_s, use_nn)
        th = fit_chunked(fdev, "adaptive")
        ftest = mk(test_s, use_nn)
        results[name] = dp.evaluate_fast(ftest, th, ix.unigram, "adaptive")
        print(f"  {results[name][0]*100:.1f}% / {results[name][1]:.3f} bits ({time.time()-t0:.0f}s)")
        # 保留 full 的检索后验用于融合
        if use_nn:
            def post(feats):
                out = []
                for i in range(0, len(feats), 500):
                    for (c, F, cnts, meta, tgt) in feats[i:i + 500]:
                        P, _ = dp.score(c, F, cnts, meta, th, ix.unigram, "adaptive")
                        out.append(P)
                return np.array(out)
            P_r_dev = post(fdev); P_r_test = post(ftest)

    # ---- 4. 融合: P = mu*P_r + (1-mu)*Pnn(T)  (dev 网格拟合 mu,T) ----
    print("[融合] dev 网格拟合 mu, T ...")
    tgt_d = np.array([t for _, t in dev_s]); tgt_t = np.array([t for _, t in test_s])
    best = (None, 1.0, 1e9)
    for T in [0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.5]:
        Pn = np.clip(nnlm_probs(m, [c for c, _ in dev_s], T), 1e-12, 1.0)
        for mu in np.arange(0.0, 1.01, 0.05):
            P = mu * P_r_dev + (1 - mu) * Pn
            p = np.clip(P[np.arange(len(tgt_d)), tgt_d], 1e-12, 1.0)
            b = float(-np.log(p).mean() / math.log(2))
            if b < best[2]:
                best = (mu, T, b)
    mu, T, b_dev = best
    print(f"  最优 mu={mu:.2f}, T={T}, dev bits={b_dev:.3f}")

    Pn_t = np.clip(nnlm_probs(m, [c for c, _ in test_s], T), 1e-12, 1.0)
    P_fused = mu * P_r_test + (1 - mu) * Pn_t
    results["DPSH-full ⊕ NNLM 融合"] = bits_of(P_fused, test_s)

    # ---------------- 结果 ----------------
    print("\n" + "=" * 74)
    print("Phase 2 — 真实文本 (wikitext-103, 词级 5000, NN 分支融合)")
    print(f"{'方法':<30}{'acc':>10}{'bits/token':>14}")
    print("-" * 74)
    for n in ["PPM[phase1]", "DPSH w/o近邻[phase1,PPMI]", "NNLM(GRU)",
              "DPSH w/o近邻(NN emb)", "DPSH-full(NN emb)", "DPSH-full ⊕ NNLM 融合"]:
        a, b = results[n]
        print(f"{n:<30}{a*100:9.1f}%{b:14.3f}")
    print("=" * 74)

    out = os.path.join(PROJ, "results", "results_real_phase2.json")
    json.dump({n: {"acc": results[n][0], "bits": results[n][1]} for n in results},
              open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("已保存:", out)

    b_r = results["DPSH-full(NN emb)"][1]
    print("\n[裁定]")
    print(f"  近邻增量(NN嵌入) : {results['DPSH w/o近邻(NN emb)'][1]:.3f} -> {b_r:.3f} "
          f"({'' if b_r < results['DPSH w/o近邻(NN emb)'][1] else '无'}增量)")
    print(f"  融合 vs 最强单分支: {results['DPSH-full ⊕ NNLM 融合'][1]:.3f} vs "
          f"{min(b_r, bits_n):.3f} (mu={mu:.2f} -> "
          f"{'互补成立, 价值放大' if results['DPSH-full ⊕ NNLM 融合'][1] < min(b_r, bits_n) else '未超过单分支'})")

if __name__ == "__main__":
    main()
