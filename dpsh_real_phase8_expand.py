# -*- coding: utf-8 -*-
"""
DPSH Phase 8: 查询扩展修复改写鲁棒性 (融合端正解)
Phase 7 结论: 改写词替换后精确检索全灭; 语义近邻检索也无效 (真实嵌入近邻=相关词非可互换).
正解: 查询扩展 — 替换词 s 的嵌入近邻里大概率含原词 w (s 就是 w 的最近邻),
      扩展成 K 个候选查询逐一精确检索, 后继分布取 max 聚合.
  候选源: PPMI 嵌入 ∪ GRU 嵌入 各 topK
  报告: 恢复率 (原词∈候选集), Acc@K, 扩展⊕Qwen 融合
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
OUT = os.path.join(PROJ, "results", "results_real_phase8_expand.json")
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

    from dpsh_real_phase6_niah import qwen_next_word_probs

    results = {}
    print("\n===== paraphrase + 查询扩展 =====", flush=True)
    rows = {}
    for b in range(5):
        pool = needles[b]
        if not pool:
            continue
        tgts, depths, queries = [], [], []      # queries: (f5, swapped_pos, orig_w)
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
            queries.append((f5, 4 - (1 - j), orig))   # 替换词在 f5 的位置: j=0->idx3, j=1->idx4
            tgts.append(c); depths.append(d)
        tgts = np.array(tgts)

        # 基线: 不扩展 (Phase 7 复现)
        P_base = np.array([post_exact(q[0]) for q in queries])

        # 扩展: 替换词的 PPMI∪GRU 近邻 topK -> 每候选精确检索 -> max 聚合
        rec = {5: [], 10: [], 20: []}
        P_exp = {5: [], 10: [], 20: []}
        for (f5, s_pos, orig) in queries:
            s = f5[s_pos]
            c1 = np.argsort(-(E_ppmi @ E_ppmi[s]))[1:25]   # rank0=s 自身已排除
            c2 = np.argsort(-(E_gru @ E_gru[s]))[1:25]
            c1 = [int(x) for x in c1]
            c2 = [int(x) for x in c2]
            for K in (5, 10, 20):
                cands = list(dict.fromkeys(c1[:K] + c2[:K]))
                rec[K].append(int(orig in cands))
                posts = []
                for w in cands:
                    q = list(f5); q[s_pos] = w    # w==orig 时即 verbatim 查询(恢复路径)
                    posts.append(post_exact(q))
                if posts:
                    P_agg = np.maximum.reduce(posts)
                else:
                    P_agg = ix.unigram
                P_exp[K].append(P_agg)
        P_exp = {K: np.array(v) for K, v in P_exp.items()}

        texts_short = [" ".join(vocab[t] for t in tr[max(0, d - 32):d]) for d in depths]
        P_q = qwen_next_word_probs(texts_short, vocab)
        P_fuse = 0.4 * P_exp[10] + 0.6 * P_q

        def a1(P): return float(np.mean(np.argmax(P, 1) == tgts))
        row = {"不扩展(基线)": a1(P_base),
               "扩展@5": a1(P_exp[5]), "扩展@10": a1(P_exp[10]), "扩展@20": a1(P_exp[20]),
               "扩展@10⊕Qwen": a1(P_fuse), "Qwen-short": a1(P_q),
               "恢复率@10": float(np.mean(rec[10]))}
        rows[BIN_NAMES[b]] = row
        print(f"  {BIN_NAMES[b]:<10} 基线={row['不扩展(基线)']*100:.0f}% "
              f"exp@5={row['扩展@5']*100:.0f}% exp@10={row['扩展@10']*100:.0f}% "
              f"exp@20={row['扩展@20']*100:.0f}% 融合={row['扩展@10⊕Qwen']*100:.0f}% "
              f"恢复率@10={row['恢复率@10']*100:.0f}%", flush=True)
    results["paraphrase+expand"] = rows

    json.dump(results, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n已保存:", OUT, flush=True)

    print(f"\n{'深度':<12}{'基线':>8}{'exp@5':>8}{'exp@10':>8}{'exp@20':>8}{'融合@10':>9}{'恢复率@10':>10}")
    for bn in BIN_NAMES:
        if bn in rows:
            r = rows[bn]
            print(f"{bn:<12}{r['不扩展(基线)']*100:7.0f}%{r['扩展@5']*100:7.0f}%"
                  f"{r['扩展@10']*100:7.0f}%{r['扩展@20']*100:7.0f}%"
                  f"{r['扩展@10⊕Qwen']*100:8.0f}%{r['恢复率@10']*100:9.0f}%")
    avg = {k: np.mean([rows[bn][k] for bn in rows]) for k in rows[BIN_NAMES[0]]}
    print(f"{'均值':<12}{avg['不扩展(基线)']*100:7.0f}%{avg['扩展@5']*100:7.0f}%"
          f"{avg['扩展@10']*100:7.0f}%{avg['扩展@20']*100:7.0f}%"
          f"{avg['扩展@10⊕Qwen']*100:8.0f}%{avg['恢复率@10']*100:9.0f}%")
    print("\n[裁定]")
    lift = (avg["扩展@10"] - avg["不扩展(基线)"]) * 100
    print(f"  扩展@10 vs 基线: {lift:+.1f} pp | 原词恢复率@10: {avg['恢复率@10']*100:.0f}% "
          f"-> {'查询扩展有效修复改写鲁棒性 ✓' if lift > 10 else '部分有效' if lift > 3 else '无效'})")

if __name__ == "__main__":
    main()
