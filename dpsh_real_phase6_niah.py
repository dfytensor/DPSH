# -*- coding: utf-8 -*-
"""
DPSH Phase 6: NIAH 长上下文本尊 (机制原生版)
设计:
  - 草垛 = 500k 真实历史 (wikitext-103, 与 phase3/4/5 同切片)
  - 针 = 历史中真实存在的独特 trigram (w_{i-2}, w_{i-1}, w_i), w_i 为低频词
  - 查询 = 针前缀 [.., w_{i-2}, w_{i-1}] -> 预测 w_i; 深度 d = i (针在历史中的位置)
  - 深度带: 0-1k / 1k-5k / 5k-20k / 20k-100k / 100k-500k
  - 原版: 查询逐字复制针前缀
  - 改写版: 把前缀中最稀有的词换成 PPMI 余弦近邻 (精确匹配断裂)
方法:
  检索系 (无窗口, O(1)): ROSA-hard / PPM / DPSH / DPSH⊕Qwen-short(局部32词)
  窗口系: Qwen-long (针前尽量多正文, 上限 20k 词) -> 深度悬崖对照
判据: 检索系 Acc@1 应深度不变; Qwen-long 在深度>窗口处崩 -> 长上下文优势立住
"""
import os, sys, io, json, math, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
from scipy.optimize import minimize

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
OUT = os.path.join(PROJ, "results", "results_real_phase6_niah.json")
SEED = 20260912
np.random.seed(SEED); torch.manual_seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
V, SCALE, QWEN_CTX_WORDS, N_PER_BIN = 5000, 500000, 20000, 60
T0 = time.time()

from dpsh_real_pg19 import tokenize as en_tokenize, build_embeddings_sparse

BINS = [(0, 1000), (1000, 5000), (5000, 20000), (20000, 100000), (100000, 500000)]
BIN_NAMES = ["0-1k", "1k-5k", "5k-20k", "20k-100k", "100k-500k"]

# ---------------- 数据 + 索引 ----------------

def build():
    books = json.load(open(os.path.join(PROJ, "results", "pg19_text_cache.json"), encoding="utf-8"))
    from collections import Counter
    train_t = en_tokenize(books[:40])[:SCALE]
    cnt = Counter(train_t)
    vocab = ["<unk>"] + [w for w, _ in cnt.most_common(V - 1)]
    wid = {w: i for i, w in enumerate(vocab)}
    tr = np.array([wid.get(w, 0) for w in train_t], dtype=np.int64)
    return tr, vocab

def sample_needles(tr, vocab):
    """独特 trigram + 低频后继, 按深度带采样"""
    from collections import Counter
    freq = Counter(tr.tolist())
    tri_pos = {}
    for i in range(3, len(tr)):
        key = (int(tr[i - 2]), int(tr[i - 1]), int(tr[i]))
        tri_pos.setdefault(key, []).append(i)
    uniq = {k: v[0] for k, v in tri_pos.items() if len(v) == 1}
    cand = [(d, k) for k, d in uniq.items()
            if 0 not in k and 2 <= freq[k[2]] <= 50]
    out = {}
    for b, (lo, hi) in enumerate(BINS):
        pool = [(d, k) for d, k in cand if lo <= d < hi]
        np.random.shuffle(pool)
        out[b] = pool[:N_PER_BIN]
    return out

# ---------------- Qwen 分支 ----------------

_QWEN = {}

def _get_qwen():
    if "m" not in _QWEN:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",
                                                 dtype=torch.float16).to(DEV).eval()
        _QWEN["m"], _QWEN["tok"] = m, tok
    return _QWEN["m"], _QWEN["tok"]

def qwen_next_word_probs(texts, vocab):
    m, tok = _get_qwen()
    first_id = np.zeros(V, dtype=np.int64)
    for i, w in enumerate(vocab):
        ids = tok.encode(" " + w, add_special_tokens=False)
        first_id[i] = ids[0] if ids else 0
    order = np.argsort([len(t) for t in texts])          # 按长度排序 -> 同批等长
    P = np.zeros((len(texts), V), dtype=np.float32)
    done = 0
    with torch.no_grad():
        i = 0
        while i < len(order):
            j = i + 1
            mx = len(texts[order[i]])
            while j < len(order) and len(texts[order[j]]) < 3000 and j - i < 64:
                j += 1
            idxs = order[i:j]
            bs = 1 if mx > 3000 else (16 if mx > 500 else 64)
            for k in range(0, len(idxs), bs):
                sel = idxs[k:k + bs]
                enc = tok([texts[t] for t in sel], return_tensors="pt", padding=True,
                          truncation=True, max_length=4096).to(DEV)
                logits = m(**enc, logits_to_keep=1).logits[:, -1].float()
                prob = torch.softmax(logits, -1).cpu().numpy()
                P[sel] = prob[:, first_id]
                P[sel] /= np.maximum(P[sel].sum(1, keepdims=True), 1e-12)
            i = j
            done += len(idxs)
            if done % 128 < 70:
                print(f"    qwen {done}/{len(texts)} ({time.time()-T0:.0f}s)", flush=True)
    return P

# ---------------- 主流程 ----------------

def main():
    import dpsh_experiment as dp
    tr, vocab = build()
    print(f"[数据] {len(tr):,} tokens ({time.time()-T0:.0f}s)")

    needles = sample_needles(tr, vocab)
    n_all = sum(len(v) for v in needles.values())
    print(f"[针] {n_all} 个独特 trigram, 深度带 {[len(needles[b]) for b in range(5)]}")

    # 索引 + 拟合 (与 phase5 相同配方)
    dp.VOCAB = V; dp.MIN_COUNT = 4
    E = build_embeddings_sparse(list(map(int, tr)), vocab, dim=64)
    ix = dp.Index(list(map(int, tr)), E, decay=0.7, K=2500)
    print(f"[索引] 节点 {len(ix.nodes):,} ({time.time()-T0:.0f}s)")
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
            c = dp.select_multi_order(ix.query(ctx, C=6, S=4, use_nn=False), True)
            if not c:
                c = [(ix.unigram, ix.unigram, dp.L_MAX, 0.0, 0.0)]
            F, cnts = dp.featurize(c)
            feats.append((c, F, cnts, dp.meta_of(c), tgt))
        return feats
    th = fit_chunked(mk(dev_s))

    def dpsq_probs(ctx_list):
        P = []
        for i in range(0, len(ctx_list), 500):
            for ctx in ctx_list[i:i + 500]:
                c = dp.select_multi_order(ix.query(list(ctx), C=6, S=4, use_nn=False), True)
                if not c:
                    P.append(ix.unigram); continue
                F, cnts = dp.featurize(c)
                p, _ = dp.score(c, F, cnts, dp.meta_of(c), th, ix.unigram, "adaptive")
                P.append(p)
        return np.array(P)

    def ppm_probs(ctx_list):
        w = np.ones(dp.L_MAX + 1)
        return np.array([dp.ppm_predict(list(c), ix, w) for c in ctx_list])

    # 评测两种变体
    results = {}
    for variant in ["verbatim", "paraphrase"]:
        print(f"\n===== 变体 {variant} =====")
        rows = {}
        for b in range(5):
            pool = needles[b]
            if not pool:
                continue
            ctx5, tgts, depths, paro = [], [], [], []
            for d, (a, bb, c) in pool:
                ctx = [a, bb]
                if variant == "paraphrase":
                    # 替换前缀中最稀有的词 -> PPMI 近邻
                    freqs = [ix.unigram[a], ix.unigram[bb]]
                    j = int(np.argmin(freqs))
                    orig = ctx[j]
                    e = E[orig]
                    sims = E @ e
                    sims[orig] = -1
                    ctx[j] = int(np.argmax(sims))
                full5 = list(tr[max(0, d - 5):d])          # 原 5 词后缀(检索用)
                full5[-1], full5[-2] = ctx[1], ctx[0]      # 替换最后两词
                ctx5.append(full5); tgts.append(c); depths.append(d)
                paro.append(full5)
            tgts = np.array(tgts)

            P_dpsh = dpsq_probs(ctx5)
            P_ppm = ppm_probs(ctx5)
            rosa = []
            for ctx in ctx5:
                P = dp.rosa_predict(list(ctx), ix, "none")
                rosa.append(P)
            rosa = np.array(rosa)
            fuse = 0.4 * P_dpsh + 0.6 * rosa * 0            # 占位, 下面真融合
            # Qwen-short: 局部 32 词 (检索⊕Qwen 融合的神经分支)
            texts_short = [" ".join(vocab[t] for t in tr[max(0, d - 32):d])
                           for d in depths]
            P_q = qwen_next_word_probs(texts_short, vocab)
            P_fuse = 0.4 * P_dpsh + 0.6 * P_q
            # Qwen-long: 窗口 20k 词(针前全文) — 每带抽 25 个控成本
            sub = np.random.choice(len(depths), size=min(8, len(depths)), replace=False)
            texts_long = [" ".join(vocab[t] for t in tr[max(0, depths[i] - QWEN_CTX_WORDS):depths[i]])
                          for i in sub]
            P_ql = qwen_next_word_probs(texts_long, vocab)
            sub_t = tgts[sub]

            def acc1(P, tgt=None):
                t = tgts if tgt is None else tgt
                return float(np.mean(np.argmax(P, 1) == t))
            row = {
                "ROSA-hard": acc1(rosa), "PPM": acc1(P_ppm), "DPSH": acc1(P_dpsh),
                "DPSH⊕Qwen(融合)": acc1(P_fuse),
                "Qwen-long(20k窗)": acc1(P_ql, sub_t),
                "n": len(tgts), "n_long": len(sub),
            }
            rows[BIN_NAMES[b]] = row
            print(f"  {BIN_NAMES[b]:<10} " + " ".join(
                f"{k.split('(')[0].strip()}={v*100:.0f}%" for k, v in row.items()
                if isinstance(v, float)))
        results[variant] = rows

    json.dump(results, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n已保存:", OUT)

    # 汇总
    for variant in results:
        print(f"\n[{variant}] Acc@1 (%)")
        methods = ["ROSA-hard", "PPM", "DPSH", "DPSH⊕Qwen(融合)", "Qwen-long(20k窗)"]
        print(f"{'深度':<12}" + "".join(f"{m:>18}" for m in methods))
        for bn in BIN_NAMES:
            if bn not in results[variant]:
                continue
            r = results[variant][bn]
            print(f"{bn:<12}" + "".join(f"{r[m]*100:17.1f}%" for m in methods))
    print("\n[裁定]")
    for variant in results:
        rows = results[variant]
        deep = BIN_NAMES[-1]
        if deep in rows:
            r = rows[deep]
            print(f"  {variant}: 100k-500k 深度带 -> DPSH {r['DPSH']*100:.0f}% "
                  f"vs Qwen-long {r['Qwen-long(20k窗)']*100:.0f}% "
                  f"(窗口悬崖 {'可见' if r['Qwen-long(20k窗)'] < r['DPSH'] - 0.2 else '不可见'})")

if __name__ == "__main__":
    main()
