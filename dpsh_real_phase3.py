# -*- coding: utf-8 -*-
"""
DPSH Phase 3: 规模 × 种子 网格 — 可靠性验证
缺口: Phase1/2 只测了 150k/单切片。本实验在 {150k, 500k} × {切片1, 切片2} 网格上
完整重跑 PPM / DPSH(后缀+多阶+门控, 无近邻) / NNLM(2层GRU加强版) / 融合,
检验两个结论是否处处成立:
  C1: DPSH(检索) bits < PPM bits
  C2: 融合 bits < min(DPSH, NNLM)
每格跑完立即增量保存 results/results_real_phase3.json
"""
import os, sys, io, json, math, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import torch
import torch.nn as nn

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
OUT = os.path.join(PROJ, "results", "results_real_phase3.json")
SEED = 20260912
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CTX, V = 16, 5000

from dpsh_real_pg19 import tokenize

# ---------------- 数据: 固定 dev/test, train 切片随 (scale, seed) 变 ----------------

def build_books():
    return json.load(open(os.path.join(PROJ, "results", "pg19_text_cache.json"), encoding="utf-8"))

def make_cell_data(books, scale, seed):
    from collections import Counter
    train_t = tokenize(books[:40])
    start = seed * 20000
    train = np.array([0] * CTX + train_t[start:start + scale])
    dev = np.array([0] * CTX + tokenize(books[40:48])[:25000])
    test = np.array([0] * CTX + tokenize(books[48:])[:60000])
    cnt = Counter(train_t)
    vocab = ["<unk>"] + [w for w, _ in cnt.most_common(V - 1)]
    wid = {w: i for i, w in enumerate(vocab)}
    tr = np.array([wid.get(w, 0) for w in train_t[start:start + scale]])
    tr = np.concatenate([np.zeros(CTX, dtype=np.int64), tr])
    dv = np.concatenate([np.zeros(CTX, dtype=np.int64),
                         np.array([wid.get(w, 0) for w in tokenize(books[40:48])[:25000]])])
    te = np.concatenate([np.zeros(CTX, dtype=np.int64),
                         np.array([wid.get(w, 0) for w in tokenize(books[48:])[:60000]])])
    return tr, dv, te

def stream(data, max_n, ctx=5):
    idx = np.linspace(ctx, len(data) - 2, min(max_n, len(data) - ctx - 1)).astype(int)
    return [(data[i - ctx:i], int(data[i])) for i in idx]

# ---------------- 加强版 NNLM ----------------

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

def train_nnlm(tr, tag, epochs=4):
    xs = np.lib.stride_tricks.sliding_window_view(tr, CTX + 1)
    X = torch.tensor(xs[:, :-1], dtype=torch.long)
    Y = torch.tensor(xs[:, -1], dtype=torch.long)
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
        print(f"    epoch {ep+1}: loss {tot/nb:.4f} ({time.time()-T0:.0f}s)")
    return m

@torch.no_grad()
def nnlm_probs(m, ctx_list, T=1.0):
    out = []
    for i in range(0, len(ctx_list), 4096):
        x = torch.tensor(np.array(ctx_list[i:i + 4096], dtype=np.int64), device=DEV)
        x = x[:, -CTX:]
        out.append(torch.softmax(m(x) / T, -1).cpu().numpy())
    return np.concatenate(out)

def bits_of(P, data_s):
    tgt = np.array([t for _, t in data_s])
    p = np.clip(P[np.arange(len(tgt)), tgt], 1e-12, 1.0)
    return float(np.mean(np.argmax(P, 1) == tgt)), float(-np.log(p).mean() / math.log(2))

# ---------------- 单格实验 ----------------

def run_cell(books, scale, seed):
    import dpsh_experiment as dp
    tag = f"{scale//1000}k_s{seed}"
    print(f"\n{'#'*70}\n# 格 {tag}\n{'#'*70}")
    tr, dv, te = make_cell_data(books, scale, seed)
    dev_s, test_s = stream(dv, 1500), stream(te, 4000)

    m = train_nnlm(tr, tag)
    m.eval()
    Pnn_d, Pnn_t = nnlm_probs(m, [c for c, _ in dev_s]), nnlm_probs(m, [c for c, _ in test_s])
    acc_n, bits_n = bits_of(Pnn_t, test_s)
    print(f"  NNLM: {acc_n*100:.1f}% / {bits_n:.3f} bits")

    dp.VOCAB = V
    dp.MIN_COUNT = 3 if scale <= 200000 else 4
    K = 1200 if scale <= 200000 else 2500
    E = m.emb.weight.detach().cpu().numpy().astype(np.float32)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    ix = dp.Index(list(map(int, tr)), E, decay=0.7, K=K)
    print(f"  索引: 节点 {len(ix.nodes):,} K={ix.K} ({time.time()-T0:.0f}s)")

    # PPM
    w_ppm = dp.fit_raw(lambda x: (lambda c: dp.ppm_predict(c, ix, x)), dev_s,
                       dp.L_MAX + 1, np.ones(dp.L_MAX + 1))
    ppm = dp.evaluate(lambda c: dp.ppm_predict(c, ix, w_ppm), test_s)

    # DPSH w/o近邻 (最终配方组件)
    init0 = np.r_[1.0, 0.5, np.zeros(dp.L_MAX + 1), 0.0, 0.5, 1.0, 2.0]
    from scipy.optimize import minimize
    def fit_chunked(feats, chunk=250, iters=25):
        B = [dp.make_batch(feats[i:i + chunk]) for i in range(0, len(feats), chunk)]
        ns = np.array([len(b["tgt"]) for b in B])
        f = lambda x: sum(dp.batch_forward(b, x, ix.unigram, "adaptive")[1] * n
                          for b, n in zip(B, ns)) / ns.sum()
        return minimize(f, init0, method="L-BFGS-B",
                        options={"maxiter": iters, "maxfun": iters * 5}).x
    def mk(data):
        feats = []
        for ctx, tgt in data:
            c = dp.select_multi_order(ix.query(ctx, C=6, S=4, use_nn=False), True)
            if not c:
                c = [(ix.unigram, ix.unigram, dp.L_MAX, 0.0, 0.0)]
            F, cnts = dp.featurize(c)
            feats.append((c, F, cnts, dp.meta_of(c), tgt))
        return feats
    fdev = mk(dev_s); ftest = mk(test_s)
    th = fit_chunked(fdev)
    dpsh = dp.evaluate_fast(ftest, th, ix.unigram, "adaptive")

    def post(feats):
        out = []
        for (c, F, cnts, meta, tgt) in feats:
            P, _ = dp.score(c, F, cnts, meta, th, ix.unigram, "adaptive")
            out.append(P)
        return np.array(out)
    P_r_d, P_r_t = post(fdev), post(ftest)

    # 融合 (dev 网格 mu/T)
    tgt_d = np.array([t for _, t in dev_s])
    best = (0.5, 1.0, 1e9)
    for T in [0.8, 0.9, 1.0, 1.1, 1.3]:
        Pn = np.clip(nnlm_probs(m, [c for c, _ in dev_s], T), 1e-12, 1.0)
        for mu in np.arange(0, 1.01, 0.05):
            P = mu * P_r_d + (1 - mu) * Pn
            p = np.clip(P[np.arange(len(tgt_d)), tgt_d], 1e-12, 1.0)
            b = float(-np.log(p).mean() / math.log(2))
            if b < best[2]:
                best = (mu, T, b)
    mu, T, _ = best
    Pn_t = np.clip(nnlm_probs(m, [c for c, _ in test_s], T), 1e-12, 1.0)
    fusion = bits_of(mu * P_r_t + (1 - mu) * Pn_t, test_s)

    row = {"PPM": ppm, "DPSH": dpsh, "NNLM": (acc_n, bits_n),
           "FUSION": fusion, "mu": float(mu), "T": T}
    print(f"  [格结果] PPM {ppm[1]:.3f} | DPSH {dpsh[1]:.3f} | NNLM {bits_n:.3f} "
          f"| FUSION {fusion[1]:.3f} (mu={mu:.2f},T={T})")
    return row

T0 = time.time()
books = build_books()
grid = {(s, k): run_cell(books, s, k)
        for s in (150000, 500000) for k in (1, 2)}
json.dump({f"{a//1000}k_s{b}": {n: {"acc": v[0], "bits": v[1]} if isinstance(v, tuple) else v
                                for n, v in r.items()}
           for (a, b), r in grid.items()},
          open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

# ---------------- 汇总 ----------------
print("\n" + "=" * 88)
print(f"{'格':<12}{'PPM':>9}{'DPSH':>9}{'NNLM':>9}{'FUSION':>9}   C1: DPSH<PPM  C2: FUS<min")
print("-" * 88)
c1_all = c2_all = True
for (s, k), r in grid.items():
    c1 = r["DPSH"][1] < r["PPM"][1]
    c2 = r["FUSION"][1] < min(r["DPSH"][1], r["NNLM"][1])
    c1_all &= c1; c2_all &= c2
    print(f"{s//1000}k_s{k:<7}{r['PPM'][1]:9.3f}{r['DPSH'][1]:9.3f}{r['NNLM'][1]:9.3f}"
          f"{r['FUSION'][1]:9.3f}   {'✓' if c1 else '✗'}            {'✓' if c2 else '✗'}"
          f"   (mu={r['mu']:.2f})")
print("=" * 88)
d_ppm = [r["DPSH"][1] - r["PPM"][1] for r in grid.values()]
d_min = [min(r["DPSH"][1], r["NNLM"][1]) - r["FUSION"][1] for r in grid.values()]
print(f"DPSH-PPM 增量: {np.mean(d_ppm)*1000:.1f}±{np.std(d_ppm)*1000:.1f} mbits | "
      f"融合增益: {np.mean(d_min)*1000:.1f}±{np.std(d_min)*1000:.1f} mbits")
print(f"可靠性判定: C1 {'处处成立' if c1_all else '不成立'} | C2 {'处处成立' if c2_all else '不成立'}")
print("已保存:", OUT)
