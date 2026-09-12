# -*- coding: utf-8 -*-
"""
DPSH Phase 4: 跨语料泛化 (中文) + 上下文自适应 mu 门控
  - 中文: 本地 minimind pretrain_t2t_mini.jsonl, jieba 词级, 500k train
  - 英文: wikitext-103 500k_s1 (同 phase3, 便于直接对比)
  - 自适应门控: mu_i = sigmoid(a + b*log1p(nmax_i) + c*mlen_i), dev L-BFGS 拟合,
                对比标量 mu (phase3 用网格)
结论口径: C1/C2 在中文上是否复现; 自适应 mu 是否优于标量 mu
"""
import os, sys, io, json, math, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
OUT = os.path.join(PROJ, "results", "results_real_phase4.json")
ZH_CACHE = os.path.join(PROJ, "results", "zh_tokens.jsonl")
SEED = 20260912
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CTX, V, SCALE = 16, 5000, 500000
T0 = time.time()

from dpsh_real_pg19 import tokenize as en_tokenize

# ---------------- 中文语料: minimind jsonl -> jieba 词级 token 流 ----------------

def zh_lines():
    """每行一个 token 列表 (行=文档边界)"""
    if os.path.exists(ZH_CACHE):
        print("[zh] 使用缓存", ZH_CACHE)
        with open(ZH_CACHE, encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)
        return
    import jieba
    jieba.setLogLevel(60)
    src = r"F:\OpenASH2605\minimind_data\pretrain_t2t_mini.jsonl"
    n = 0
    with open(src, encoding="utf-8") as f, open(ZH_CACHE, "w", encoding="utf-8") as out:
        for line in f:
            try:
                text = json.loads(line).get("text", "")
            except Exception:
                continue
            toks = [t.strip().lower() for t in jieba.lcut(text)]
            toks = [t for t in toks if t and not t.isspace()]
            if len(toks) < 20:
                continue
            out.write(json.dumps(toks, ensure_ascii=False) + "\n")
            n += 1
            if n % 2000 == 0:
                print(f"  jieba 已处理 {n} 行 ({time.time()-T0:.0f}s)")
            if n >= 6000:
                break
    print(f"[zh] 缓存完成 {n} 行")
    with open(ZH_CACHE, encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def build_zh_cell():
    train, dev, test = [], [], []
    quota = [SCALE, 25000, 60000]
    for toks in zh_lines():
        ids = [0] * CTX + toks
        for i in range(3):
            if quota[i] > 0:
                take = min(quota[i], len(ids))
                [train, dev, test][i].extend(ids[:take])
                quota[i] -= take
                break
        if all(q <= 0 for q in quota):
            break
    from collections import Counter
    cnt = Counter(train)
    vocab = ["<unk>"] + [w for w, _ in cnt.most_common(V - 1)]
    wid = {w: i for i, w in enumerate(vocab)}
    enc = lambda ts: np.array([wid.get(w, 0) if isinstance(w, str) else int(w) for w in ts])
    return enc(train), enc(dev), enc(test)

def build_en_cell():
    books = json.load(open(os.path.join(PROJ, "results", "pg19_text_cache.json"), encoding="utf-8"))
    from collections import Counter
    train_t = en_tokenize(books[:40])[:SCALE]
    cnt = Counter(train_t)
    vocab = ["<unk>"] + [w for w, _ in cnt.most_common(V - 1)]
    wid = {w: i for i, w in enumerate(vocab)}
    enc = lambda ts: np.array([wid.get(w, 0) for w in ts])
    tr = np.concatenate([np.zeros(CTX, dtype=np.int64), enc(train_t)])
    dv = np.concatenate([np.zeros(CTX, dtype=np.int64), enc(en_tokenize(books[40:48])[:25000])])
    te = np.concatenate([np.zeros(CTX, dtype=np.int64), enc(en_tokenize(books[48:])[:60000])])
    return tr, dv, te

def stream(data, max_n, ctx=5):
    idx = np.linspace(ctx, len(data) - 2, min(max_n, len(data) - ctx - 1)).astype(int)
    return [(data[i - ctx:i], int(data[i])) for i in idx]

# ---------------- NNLM (同 phase3 加强版) ----------------

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
        print(f"    epoch {ep+1}: loss {tot/nb:.4f} ({time.time()-T0:.0f}s)")
    return m

@torch.no_grad()
def nnlm_probs(m, ctx_list, T=1.0):
    out = []
    for i in range(0, len(ctx_list), 4096):
        x = torch.tensor(np.array(ctx_list[i:i + 4096], dtype=np.int64), device=DEV)[:, -CTX:]
        out.append(torch.softmax(m(x) / T, -1).cpu().numpy())
    return np.concatenate(out)

def bits_of(P, data_s):
    tgt = np.array([t for _, t in data_s])
    p = np.clip(P[np.arange(len(tgt)), tgt], 1e-12, 1.0)
    return float(np.mean(np.argmax(P, 1) == tgt)), float(-np.log(p).mean() / math.log(2))

# ---------------- 单语实验 ----------------

def run(lang):
    import dpsh_experiment as dp
    print(f"\n{'#'*70}\n# 语料 {lang} (500k)\n{'#'*70}")
    tr, dv, te = build_zh_cell() if lang == "zh" else build_en_cell()
    dev_s, test_s = stream(dv, 1500), stream(te, 4000)

    m = train_nnlm(tr, lang); m.eval()
    Pnn_d, Pnn_t = nnlm_probs(m, [c for c, _ in dev_s]), nnlm_probs(m, [c for c, _ in test_s])
    acc_n, bits_n = bits_of(Pnn_t, test_s)
    print(f"  NNLM: {acc_n*100:.1f}% / {bits_n:.3f}")

    dp.VOCAB = V; dp.MIN_COUNT = 4
    E = m.emb.weight.detach().cpu().numpy().astype(np.float32)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)
    ix = dp.Index(list(map(int, tr)), E, decay=0.7, K=2500)
    print(f"  索引: 节点 {len(ix.nodes):,} ({time.time()-T0:.0f}s)")

    w_ppm = dp.fit_raw(lambda x: (lambda c: dp.ppm_predict(c, ix, x)), dev_s,
                       dp.L_MAX + 1, np.ones(dp.L_MAX + 1))
    ppm = dp.evaluate(lambda c: dp.ppm_predict(c, ix, w_ppm), test_s)

    init0 = np.r_[1.0, 0.5, np.zeros(dp.L_MAX + 1), 0.0, 0.5, 1.0, 2.0]
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
    fdev, ftest = mk(dev_s), mk(test_s)
    th = fit_chunked(fdev)
    dpsh = dp.evaluate_fast(ftest, th, ix.unigram, "adaptive")

    def post(feats):
        return np.array([dp.score(c, F, cnts, meta, th, ix.unigram, "adaptive")[0]
                         for (c, F, cnts, meta, tgt) in feats])
    P_r_d, P_r_t = post(fdev), post(ftest)
    tgt_d = np.array([t for _, t in dev_s]); tgt_t = np.array([t for _, t in test_s])
    id_d, id_t = np.arange(len(tgt_d)), np.arange(len(tgt_t))

    # 标量 mu + T
    best = (0.5, 1.0, 1e9)
    for T in [0.8, 0.9, 1.0, 1.1, 1.3]:
        Pn = np.clip(nnlm_probs(m, [c for c, _ in dev_s], T), 1e-12, 1.0)
        for mu in np.arange(0, 1.01, 0.05):
            P = mu * P_r_d + (1 - mu) * Pn
            p = np.clip(P[id_d, tgt_d], 1e-12, 1.0)
            b = float(-np.log(p).mean() / math.log(2))
            if b < best[2]:
                best = (mu, T, b)
    mu_s, T_s, _ = best
    Pn_t = np.clip(nnlm_probs(m, [c for c, _ in test_s], T_s), 1e-12, 1.0)
    P_fuse_s = mu_s * P_r_t + (1 - mu_s) * Pn_t
    fuse_scalar = bits_of(P_fuse_s, test_s)

    # 自适应 mu: sigmoid(a + b*log1p(nmax) + c*mlen)
    nmax_d = np.array([meta[1] for (_, _, _, meta, _) in fdev])
    mlen_d = np.array([meta[0] for (_, _, _, meta, _) in fdev])
    nmax_t = np.array([meta[1] for (_, _, _, meta, _) in ftest])
    mlen_t = np.array([meta[0] for (_, _, _, meta, _) in ftest])
    Pn_d_fix = np.clip(nnlm_probs(m, [c for c, _ in dev_s], T_s), 1e-12, 1.0)
    def obj_mu(x):
        mu_d = 1 / (1 + np.exp(-(x[0] + x[1] * np.log1p(nmax_d) + x[2] * mlen_d)))
        P = mu_d[:, None] * P_r_d + (1 - mu_d)[:, None] * Pn_d_fix
        p = np.clip(P[id_d, tgt_d], 1e-12, 1.0)
        return float(-np.log(p).mean() / math.log(2))
    r = minimize(obj_mu, np.array([0.0, 0.0, 0.0]), method="Nelder-Mead",
                 options={"maxiter": 200})
    mu_t = 1 / (1 + np.exp(-(r.x[0] + r.x[1] * np.log1p(nmax_t) + r.x[2] * mlen_t)))
    P_fuse_a = mu_t[:, None] * P_r_t + (1 - mu_t)[:, None] * Pn_t
    fuse_adaptive = bits_of(P_fuse_a, test_s)
    print(f"  标量 mu={mu_s:.2f} T={T_s} | 自适应 mu: mean={mu_t.mean():.2f} "
          f"范围[{mu_t.min():.2f},{mu_t.max():.2f}] a,b,c={np.round(r.x,2)}")

    row = {"PPM": ppm, "DPSH": dpsh, "NNLM": (acc_n, bits_n),
           "FUSION标量mu": fuse_scalar, "FUSION自适应mu": fuse_adaptive,
           "mu_scalar": mu_s, "T": T_s}
    print(f"  [结果] PPM {ppm[1]:.3f} | DPSH {dpsh[1]:.3f} | NNLM {bits_n:.3f} "
          f"| 融合(标量) {fuse_scalar[1]:.3f} | 融合(自适应) {fuse_adaptive[1]:.3f}")
    return row

grid = {"en": run("en"), "zh": run("zh")}
json.dump({k: {n: {"acc": v[0], "bits": v[1]} if isinstance(v, tuple) else v
               for n, v in r.items()} for k, r in grid.items()},
          open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

print("\n" + "=" * 96)
print(f"{'语料':<8}{'PPM':>9}{'DPSH':>9}{'NNLM':>9}{'融合(标量)':>11}{'融合(自适应)':>12}   C1   C2")
print("-" * 96)
for lang, r in grid.items():
    c1 = r["DPSH"][1] < r["PPM"][1]
    c2 = r["FUSION标量mu"][1] < min(r["DPSH"][1], r["NNLM"][1])
    print(f"{lang:<8}{r['PPM'][1]:9.3f}{r['DPSH'][1]:9.3f}{r['NNLM'][1]:9.3f}"
          f"{r['FUSION标量mu'][1]:11.3f}{r['FUSION自适应mu'][1]:12.3f}   "
          f"{'✓' if c1 else '✗'}    {'✓' if c2 else '✗'}")
print("=" * 96)
for lang, r in grid.items():
    d = (r["FUSION自适应mu"][1] - r["FUSION标量mu"][1]) * 1000
    print(f"{lang}: 自适应mu vs 标量mu = {d:+.1f} mbits")
print("已保存:", OUT)
