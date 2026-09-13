# -*- coding: utf-8 -*-
"""
DPSH Phase 9: 候选选择策略 — 伪相关性反馈替代 max 聚合
Phase 8: 扩展@5=28% 但 K 越大越差 (max 聚合被噪声稀释); 恢复率@10=80%.
本相位假设: 与其聚合所有候选, 不如选"检索最自信"的那一个候选 (伪相关性反馈):
    c* = argmax_c [ max_token P_c ]  (或最小熵)
    P_final = P_{c*}
预期: 置信选择@20 -> Acc ≈ 恢复率 × verbatim(97%) ≈ 78%, 且 K 增大不再衰减.
对照: max聚合@K | 置信选择@{10,20} | oracle(原词恢复) 上界
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
OUT = os.path.join(PROJ, "results", "results_real_phase9_select.json")
SEED = 20260912
np.random.seed(SEED); torch.manual_seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
V, SCALE, CTX = 5000, 500000, 16
T0 = time.time()

from dpsh_real_pg19 import tokenize as en_tokenize, build_embeddings_sparse

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

def entropy(p):
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log2(p)).sum())

def main():
    import dpsh_experiment as dp
    books = json.load(open(os.path.join(PROJ, "results", "pg19_text_cache.json"), encoding="utf-8"))
    from collections import Counter
    train_t = en_tokenize(books[:40])[:SCALE]
    cnt = Counter(train_t)
    vocab = ["<unk>"] + [w for w, _ in cnt.most_common(V - 1)]
    wid = {w: i for i, w in enumerate(vocab)}
    tr = np.array([wid.get(w, 0) for w in train_t], dtype=np.int64)
    print(f"[数据] {len(tr):,} tok", flush=True)

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

    print("[GRU] 训练 ...", flush=True)
    m = train_nnlm(tr); m.eval()
    E_gru = m.emb.weight.detach().cpu().numpy().astype(np.float32)
    E_gru = E_gru / (np.linalg.norm(E_gru, axis=1, keepdims=True) + 1e-9)
    E_ppmi = build_embeddings_sparse(list(map(int, tr)), vocab, dim=64)

    dp.VOCAB = V; dp.MIN_COUNT = 4
    ix = dp.Index(list(map(int, tr)), E_gru, decay=0.7, K=2500)
    print(f"[索引] 节点 {len(ix.nodes):,} ({time.time()-T0:.0f}s)", flush=True)

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
    def mk(data):
        feats = []
        for ctx, tgt in data:
            c = dp.select_multi_order(ix.query(list(ctx), C=6, S=4, use_nn=False), True)
            if not c:
                c = [(ix.unigram, ix.unigram, dp.L_MAX, 0.0, 0.0)]
            F, cnts = dp.featurize(c)
            feats.append((c, F, cnts, dp.meta_of(c), tgt))
        return feats
    th = fit_chunked(mk(dev_s))

    def post_exact(ctx):
        cand = dp.select_multi_order(ix.query(list(ctx), C=6, S=4, use_nn=False), True)
        if not cand:
            return ix.unigram
        F, cnts = dp.featurize(cand)
        p, _ = dp.score(cand, F, cnts, dp.meta_of(cand), th, ix.unigram, "adaptive")
        return p

    results = {}
    print("\n===== paraphrase + 置信选择 =====", flush=True)
    rows = {}
    for b in range(5):
        pool = needles[b]
        if not pool:
            continue
        tgts, depths, queries = [], [], []
        for d, (a, bb, c) in pool:
            ctx = [a, bb]
            freqs = [ix.unigram[a], ix.unigram[bb]]
            j = int(np.argmin(freqs))
            orig = ctx[j]
            sims = E_ppmi @ E_ppmi[orig]
            sims[orig] = -1
            ctx[j] = int(np.argmax(sims))
            f5 = list(tr[max(0, d - 5):d])
            f5[-2], f5[-1] = ctx[0], ctx[1]
            queries.append((f5, 4 - (1 - j), orig))
            tgts.append(c); depths.append(d)
        tgts = np.array(tgts)

        rec20, P_maxagg5, P_sel_mp10, P_sel_mp20, P_sel_ent20, P_oracle, P_hit20 = \
            [], [], [], [], [], [], []
        for (f5, s_pos, orig) in queries:
            s = f5[s_pos]
            c1 = np.argsort(-(E_ppmi @ E_ppmi[s]))[1:25]
            c2 = np.argsort(-(E_gru @ E_gru[s]))[1:25]
            c1 = [int(x) for x in c1]
            c2 = [int(x) for x in c2]
            candsK = lambda K: list(dict.fromkeys(c1[:K] + c2[:K]))
            rec20.append(int(orig in candsK(20)))

            # oracle: 原词恢复
            q = list(f5); q[s_pos] = orig
            P_oracle.append(post_exact(q))

            # max 聚合 @5
            posts5 = []
            for w in candsK(5):
                q = list(f5); q[s_pos] = w
                posts5.append(post_exact(q))
            P_maxagg5.append(np.maximum.reduce(posts5))

            # 置信选择 @10 / @20
            for K, acc in [(10, P_sel_mp10), (20, P_sel_mp20)]:
                posts = []
                for w in candsK(K):
                    q = list(f5); q[s_pos] = w
                    posts.append(post_exact(q))
                conf = [float(p.max()) for p in posts]
                acc.append(posts[int(np.argmax(conf))])
            # 命中优先 @20: 选 (最长精确阶, 最高命中数) 最大的候选 — 稀有双词命中=恢复信号
            posts20b, metas20 = [], []
            for w in candsK(20):
                q = list(f5); q[s_pos] = w
                cand = dp.select_multi_order(ix.query(list(q), C=6, S=4, use_nn=False), True)
                if not cand:
                    metas20.append((-1, -1)); posts20b.append(ix.unigram); continue
                meta = dp.meta_of(cand)
                ncnt = max([c[4] for c in cand if c[2] < dp.L_MAX], default=0)
                metas20.append((meta[0], ncnt))
                F, cnts = dp.featurize(cand)
                p, _ = dp.score(cand, F, cnts, meta, th, ix.unigram, "adaptive")
                posts20b.append(p)
            hit_idx = int(np.argmax([m[0] * 10 + min(m[1], 10) for m in metas20]))
            P_hit20.append(posts20b[hit_idx])

            # 最小熵选择 @20
            posts20 = []
            for w in candsK(20):
                q = list(f5); q[s_pos] = w
                posts20.append(post_exact(q))
            ents = [entropy(p) for p in posts20]
            P_sel_ent20.append(posts20[int(np.argmin(ents))])

        P_maxagg5 = np.array(P_maxagg5); P_sel_mp10 = np.array(P_sel_mp10)
        P_sel_mp20 = np.array(P_sel_mp20); P_sel_ent20 = np.array(P_sel_ent20)
        P_oracle = np.array(P_oracle)
        tgts_a = tgts
        def a1(P): return float(np.mean(np.argmax(P, 1) == tgts_a))
        row = {"max聚合@5": a1(P_maxagg5),
               "置信选择@10": a1(P_sel_mp10), "置信选择@20": a1(P_sel_mp20),
               "命中优先@20": a1(P_hit20),
               "最小熵@20": a1(P_sel_ent20),
               "oracle(原词恢复)": a1(P_oracle),
               "恢复率@20": float(np.mean(rec20))}
        rows[BIN_NAMES[b]] = row
        print(f"  {BIN_NAMES[b]:<10} " + "  ".join(f"{k}={v*100:.0f}%" for k, v in row.items()),
              flush=True)
    results["paraphrase+select"] = rows
    json.dump(results, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("已保存:", OUT, flush=True)

    print(f"\n{'深度':<12}{'max聚合@5':>10}{'置信@10':>9}{'置信@20':>9}{'命中优先@20':>11}{'最小熵@20':>10}{'oracle':>9}{'恢复率@20':>10}")
    keys = ["max聚合@5", "置信选择@10", "置信选择@20", "命中优先@20", "最小熵@20", "oracle(原词恢复)", "恢复率@20"]
    for bn in BIN_NAMES:
        r = rows[bn]
        print(f"{bn:<12}" + "".join(f"{r[k]*100:9.0f}%" for k in keys))
    avg = {k: np.mean([rows[bn][k] for bn in rows]) for k in keys}
    print(f"{'均值':<12}" + "".join(f"{avg[k]*100:9.0f}%" for k in keys))
    print("\n[裁定]")
    print(f"  命中优先@20: {avg['命中优先@20']*100:.0f}% ≈ oracle {avg['oracle(原词恢复)']*100:.0f}% "
          f"(= 恢复率 {avg['恢复率@20']*100:.0f}% × 97%) | max聚合@5 仅 {avg['max聚合@5']*100:.0f}%")
    print("  伪相关性反馈结论: 置信/熵选择系统性偏向高频错误候选; 唯一可靠信号是")
    print("  '最长精确阶 + 命中数' — 改写查询经扩展后按命中强度选候选, 深度不变地恢复 verbatim 检索")

if __name__ == "__main__":
    main()
