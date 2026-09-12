# -*- coding: utf-8 -*-
"""
DPSH 真实链路验证: PG-19 真实书本上的 next-token 预测 (bits/token)
路线图来源: README "再往后是接真实链路: RWKV7+ROSA 流水线、PG-19 PPL、改写版 NIAH、LongBench"

设计:
  - 语料: PG-19 真实英文小说 (流式拉取, 本地缓存), 书本级切分 train/dev/test
  - 词级 token (小写), 词表封顶 -> <unk>
  - 嵌入: 稀疏 PPMI + svds(64 维)  (真实词表下 dense SVD 不可行)
  - 方法: Unigram / ROSA-hard / ROSA+WB / PPM(L-BFGS 拟合) /
          DPSH w/o近邻 / DPSH-full(自适应超边)   (均按 README 口径, dev 上自动拟合)
判定: test books 上 DPSH bits < PPM bits => 机制在真实文本上有增量价值
"""
import os, sys, io, json, math, time, re
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import numpy as np
from collections import Counter

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
CACHE = os.path.join(PROJ, "results", "pg19_text_cache.json")

# ---------------- 1. 真实语料: PG-19 (fallback wikitext-103) ----------------

def load_books():
    if os.path.exists(CACHE):
        print(f"[语料] 使用缓存 {CACHE}")
        return json.load(open(CACHE, encoding="utf-8"))
    books = []
    try:
        from datasets import load_dataset
        print("[语料] 流式拉取 deepmind/pg19 ...")
        ds = load_dataset("deepmind/pg19", split="train", streaming=True)
        for row in ds:
            t = row.get("full_text") or row.get("text") or ""
            if len(t) > 20000:
                books.append(t)
            if len(books) >= 60:
                break
        src = "pg19"
    except Exception as e:
        print(f"[语料] pg19 失败({e}), fallback wikitext-103")
        from datasets import load_dataset
        ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1",
                          split="train", streaming=True)
        buf, books = [], []
        for row in ds:
            buf.append(row["text"])
            if len(buf) >= 8000:
                books.append("\n".join(buf)); buf = []
            if len(books) >= 80:
                break
        src = "wikitext-103"
    json.dump(books, open(CACHE, "w", encoding="utf-8"))
    print(f"[语料] {src}: {len(books)} 本书已缓存")
    return books


def tokenize(texts):
    toks = []
    for t in texts:
        toks += re.findall(r"[a-z]+", t.lower())
    return toks

# ---------------- 2. 稀疏 PPMI + svds 嵌入 (真实词表) ----------------

def build_embeddings_sparse(corpus, vocab, dim=64, window=5):
    import scipy.sparse as sp
    from scipy.sparse.linalg import svds
    n = len(vocab)
    c = np.asarray(corpus)
    rows, cols, vals = [], [], []
    for d in range(1, window + 1):
        a, b = c[:-d], c[d:]
        for side in (0, 1):
            r = a if side == 0 else b
            cc = b if side == 0 else a
            mask = r != cc
            rows.append(r[mask]); cols.append(cc[mask])
            vals.append(np.full(mask.sum(), 1.0 / d, dtype=np.float32))
    rows = np.concatenate(rows); cols = np.concatenate(cols); vals = np.concatenate(vals)
    co = sp.coo_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float32)
    co.sum_duplicates()
    tot = co.sum()
    row = np.bincount(co.row, weights=co.data, minlength=n)
    col = np.bincount(co.col, weights=co.data, minlength=n)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((co.data * tot + 1e-9) / (row[co.row] * col[co.col] + 1e-9))
    ppmi_data = np.maximum(pmi, 0.0).astype(np.float32)
    ppmi = sp.csr_matrix((ppmi_data, (co.row, co.col)), shape=(n, n))
    U, S, _ = svds(ppmi, k=dim)
    order = np.argsort(-S)
    U, S = U[:, order], S[order]
    E = (U * S[None, :] ** 0.5).astype(np.float32)
    E[E < -1e4] = 0; E[E > 1e4] = 0
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

# ---------------- 3. 主流程 ----------------

def main():
    import dpsh_experiment as dp

    t0 = time.time()
    books = load_books()
    n_train, n_dev = 40, 8
    train_t = tokenize(books[:n_train])
    dev_t = tokenize(books[n_train:n_train + n_dev])
    test_t = tokenize(books[n_train + n_dev:])
    print(f"[切分] 书本级: train={n_train}本({len(train_t):,}tok) "
          f"dev={n_dev}本({len(dev_t):,}tok) test={len(books)-n_train-n_dev}本({len(test_t):,}tok)")

    # 词表: train 高频 5000 + <unk>=0
    V = 5000
    cnt = Counter(train_t)
    vocab = ["<unk>"] + [w for w, _ in cnt.most_common(V - 1)]
    wid = {w: i for i, w in enumerate(vocab)}
    unk = 0
    enc = lambda ts: [wid.get(w, unk) for w in ts]
    train = enc(train_t)[:150000]          # 索引规模封顶 (与合成实验同量级)
    dev = enc(dev_t)[:25000]
    test = enc(test_t)[:60000]
    dp.VOCAB = V
    oov_test = np.mean([1 if w not in wid else 0 for w in test_t])
    print(f"[词表] {V} (OOV率 test={oov_test*100:.1f}% -> <unk>)")

    print("[嵌入] 稀疏 PPMI + svds(64) ...")
    E = build_embeddings_sparse(train, vocab, dim=64)
    print(f"  嵌入完成 ({time.time()-t0:.0f}s)")

    # 索引 (MIN_COUNT=3 控制节点数; K 收敛到节点数约束内)
    dp.MIN_COUNT = 3
    print("[索引] 后缀计数 + 节点 + 聚类超边 ...")
    ix = dp.Index(train, E, decay=0.7, K=1200)
    print(f"  节点 {len(ix.nodes):,} | 聚类 K={ix.K} ({time.time()-t0:.0f}s)")

    # 评测样本: 滑窗所有位置 (真实分布, 非构造任务)
    def stream(data, max_n):
        idx = np.linspace(6, len(data) - 2, min(max_n, len(data) - 6)).astype(int)
        return [(data[i - 5:i], data[i]) for i in idx]
    dev_s = stream(dev, 1500)
    test_s = stream(test, 4000)
    print(f"[评测] dev={len(dev_s)}, test={len(test_s)} 位置")

    results = {}
    results["Unigram"] = dp.evaluate(lambda c: ix.unigram, test_s)
    results["ROSA-hard"] = dp.evaluate(lambda c: dp.rosa_predict(c, ix, "none"), test_s)
    results["ROSA+WB"] = dp.evaluate(lambda c: dp.rosa_predict(c, ix, "wb"), test_s)
    print(f"[fit] PPM (L-BFGS on dev) ...")
    w_ppm = dp.fit_raw(lambda x: (lambda c: dp.ppm_predict(c, ix, x)), dev_s, dp.L_MAX + 1,
                       np.ones(dp.L_MAX + 1))
    results["PPM(变阶n-gram)"] = dp.evaluate(lambda c: dp.ppm_predict(c, ix, w_ppm), test_s)
    print(f"  PPM done ({time.time()-t0:.0f}s)")

    N_THETA = 2 + (dp.L_MAX + 1) + 3 + 1
    init0 = np.r_[1.0, 0.5, np.zeros(dp.L_MAX + 1), 0.0, 0.5, 1.0, 2.0]

    # 分块拟合适配真实词表内存 (覆盖 dp.fit 的整批 make_batch)
    from scipy.optimize import minimize
    def fit_chunked(feats, mode, chunk=250, iters=40):
        B = [dp.make_batch(feats[i:i + chunk]) for i in range(0, len(feats), chunk)]
        ns = np.array([len(b["tgt"]) for b in B])
        def obj(x):
            tot = sum(dp.batch_forward(b, x, ix.unigram, mode)[1] * n
                      for b, n in zip(B, ns))
            return tot / ns.sum()
        return minimize(obj, init0, method="L-BFGS-B",
                        options={"maxiter": iters, "maxfun": iters * 5}).x

    def mk(data, use_nn, multi_order):
        feats = []
        for ctx, tgt in data:
            c = dp.select_multi_order(ix.query(ctx, C=6, S=4, use_nn=use_nn), multi_order)
            if not c:
                c = [(ix.unigram, ix.unigram, dp.L_MAX, 0.0, 0.0)]
            F, cnts = dp.featurize(c)
            feats.append((c, F, cnts, dp.meta_of(c), tgt))
        return feats

    for name, kw in {
        "DPSH w/o近邻": dict(use_nn=False, multi_order=True, mode="adaptive"),
        "DPSH-full(自适应超边)": dict(use_nn=True, multi_order=True, mode="adaptive"),
    }.items():
        print(f"[fit] {name} (聚类重挂 + dev 拟合) ...")
        fdev = mk(dev_s, kw["use_nn"], kw["multi_order"])
        th = fit_chunked(fdev, kw["mode"])
        ftest = mk(test_s, kw["use_nn"], kw["multi_order"])
        results[name] = dp.evaluate_fast(ftest, th, ix.unigram, kw["mode"])
        print(f"  done ({time.time()-t0:.0f}s)")

    # ---------------- 结果 ----------------
    print("\n" + "=" * 72)
    print(f"PG-19 真实书本 next-token 预测 (词级, 词表 {V}, OOV={oov_test*100:.1f}%)")
    print(f"{'方法':<22}{'acc':>10}{'bits/token':>14}")
    print("-" * 72)
    for n in ["Unigram", "ROSA-hard", "ROSA+WB", "PPM(变阶n-gram)",
              "DPSH w/o近邻", "DPSH-full(自适应超边)"]:
        a, b = results[n]
        print(f"{n:<22}{a*100:9.1f}%{b:14.3f}")
    print("=" * 72)

    out = os.path.join(PROJ, "results", "results_real_pg19.json")
    json.dump({n: {"acc": results[n][0], "bits": results[n][1]} for n in results},
              open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("已保存:", out)

    b_ppm = results["PPM(变阶n-gram)"][1]
    b_dp = results["DPSH-full(自适应超边)"][1]
    b_nonn = results["DPSH w/o近邻"][1]
    print("\n[裁定]")
    print(f"  DPSH-full vs PPM : {b_dp:.3f} vs {b_ppm:.3f} bits "
          f"({'' if b_dp < b_ppm else '未'}优于 PPM, {(1-b_dp/b_ppm)*100:+.1f}%)")
    print(f"  语义近邻的增量   : {b_nonn:.3f} -> {b_dp:.3f} "
          f"({'' if b_dp < b_nonn else '未'}有效, {(1-b_dp/b_nonn)*100:+.1f}%)")

if __name__ == "__main__":
    main()
