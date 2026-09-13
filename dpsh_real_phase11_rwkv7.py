# -*- coding: utf-8 -*-
"""
DPSH Phase 11: 第一档(推理时外挂 logit 融合) @ RWKV-7 "Goose" — 引用测试
协议与 Phase 5 完全一致 (en 500k 同切片, 检索后验 P_r 相同配方):
  - DPSH(检索) 单独
  - RWKV-7 0.4B 单独 (32 词窗口, 首 BPE 聚合到词级 5000)
  - 融合: mu*P_r + (1-mu)*P_rwkv(T), dev 网格拟合
对照: Phase 5 Qwen 融合 5.966 bits / GRU 融合 6.855 bits
实现: transformers-rwkv7 (纯 PyTorch, 零 fla/triton), 模型 F:\\rwkv\\models\\rwkv7-0.4b-hf
"""
import os, sys, io, json, math, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import torch
from scipy.optimize import minimize

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
sys.path.insert(0, r"F:\rwkv\transformers-rwkv7")
OUT = os.path.join(PROJ, "results", "results_real_phase11_rwkv7.json")
SEED = 20260912
np.random.seed(SEED); torch.manual_seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CTX, V, SCALE, NCTX_R = 16, 5000, 500000, 32
T0 = time.time()

from dpsh_real_pg19 import tokenize as en_tokenize, build_embeddings_sparse

# ---------------- 数据 (phase5 同切片) ----------------

def build_en_cell():
    books = json.load(open(os.path.join(PROJ, "results", "pg19_text_cache.json"), encoding="utf-8"))
    from collections import Counter
    train_t = en_tokenize(books[:40])[:SCALE]
    cnt = Counter(train_t)
    vocab = ["<unk>"] + [w for w, _ in cnt.most_common(V - 1)]
    wid = {w: i for i, w in enumerate(vocab)}
    enc = lambda ts: np.array([wid.get(w, 0) for w in ts], dtype=np.int64)
    tr = enc(train_t)
    dv = enc(en_tokenize(books[40:48])[:25000])
    te = enc(en_tokenize(books[48:])[:60000])
    return tr, dv, te, vocab

def stream(data, max_n, ctx=5):
    idx = np.linspace(ctx, len(data) - 2, min(max_n, len(data) - ctx - 1)).astype(int)
    return [(data[i - ctx:i], int(data[i])) for i in idx]

# ---------------- RWKV-7 分支 ----------------

def get_rwkv():
    if "m" not in _R:
        import transformers_rwkv7  # 纯 torch RWKV-7, 无需 fla/triton
        from transformers_rwkv7 import Rwkv7ForCausalLM, Rwkv7Tokenizer
        d = r"F:\rwkv\models\rwkv7-0.4b-hf"
        tok = Rwkv7Tokenizer(vocab_file=r"F:\rwkv\models\rwkv7-0.1b-hf\rwkv_vocab_v20230424.txt")
        if tok.pad_token is None:
            tok.pad_token = "\n\n"          # RWKV World 词表 id 0
        if tok.pad_token_id is None or tok.pad_token_id < 0:
            tok.pad_token_id = 0
        m = Rwkv7ForCausalLM.from_pretrained(d, dtype=torch.float16).to(DEV).eval()
        _R["m"], _R["tok"] = m, tok
        print(f"    RWKV-7 加载: {sum(p.numel() for p in m.parameters())/1e6:.0f}M 参数 "
              f"({time.time()-T0:.0f}s)", flush=True)
    return _R["m"], _R["tok"]

@torch.no_grad()
def rwkv_probs(texts, vocab):
    """(N, V) 概率: 每条文本的下一 token 分布按 词首BPE 聚合"""
    m, tok = get_rwkv()
    first_id = np.zeros(V, dtype=np.int64)
    for i, w in enumerate(vocab):
        ids = tok.encode(" " + w, add_special_tokens=False)
        first_id[i] = ids[0] if ids else 0
    P = np.zeros((len(texts), V), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            enc = tok(texts[i:i + 64], return_tensors="pt", padding=True,
                      truncation=True, max_length=256).to(DEV)
            logits = m(**enc).logits[:, -1].float()
            prob = torch.softmax(logits, -1).cpu().numpy()
            P[i:i + 64] = prob[:, first_id]
            P[i:i + 64] /= np.maximum(P[i:i + 64].sum(1, keepdims=True), 1e-12)
    return P

def bits_of(P, data_s):
    tgt = np.array([t for _, t in data_s])
    p = np.clip(P[np.arange(len(tgt)), tgt], 1e-12, 1.0)
    return float(np.mean(np.argmax(P, 1) == tgt)), float(-np.log(p).mean() / math.log(2))

_R = {}

def main():
    import dpsh_experiment as dp
    tr, dv, te, vocab = build_en_cell()
    dev_s, test_s = stream(dv, 1500), stream(te, 4000)
    print(f"[数据] en 500k | dev {len(dev_s)} | test {len(test_s)} ({time.time()-T0:.0f}s)", flush=True)

    print("[RWKV-7] 分支分布 ...", flush=True)
    texts_d = [" ".join(vocab[t] for t in dv[max(0, p - NCTX_R):p]) for p in
               np.linspace(6, len(dv) - 2, len(dev_s)).astype(int)]
    texts_t = [" ".join(vocab[t] for t in te[max(0, p - NCTX_R):p]) for p in
               np.linspace(6, len(te) - 2, len(test_s)).astype(int)]
    P_rkv_d = rwkv_probs(texts_d, vocab)
    P_rkv_t = rwkv_probs(texts_t, vocab)
    acc_r, bits_r = bits_of(P_rkv_t, test_s)
    print(f"  RWKV-7 单独: {acc_r*100:.1f}% / {bits_r:.3f} bits ({time.time()-T0:.0f}s)", flush=True)

    # 检索分支 (phase5 同配方: PPMI+SVD 嵌入索引)
    dp.VOCAB = V; dp.MIN_COUNT = 4
    print("[索引] PPMI+SVD ...", flush=True)
    E = build_embeddings_sparse(list(map(int, tr)), vocab, dim=64)
    ix = dp.Index(list(map(int, tr)), E, decay=0.7, K=2500)
    print(f"  节点 {len(ix.nodes):,} ({time.time()-T0:.0f}s)", flush=True)

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
            c = dp.select_multi_order(ix.query(list(ctx), C=6, S=4, use_nn=False), True)
            if not c:
                c = [(ix.unigram, ix.unigram, dp.L_MAX, 0.0, 0.0)]
            F, cnts = dp.featurize(c)
            feats.append((c, F, cnts, dp.meta_of(c), tgt))
        return feats
    fdev, ftest = mk(dev_s), mk(test_s)
    th = fit_chunked(fdev)
    dpsh = dp.evaluate_fast(ftest, th, ix.unigram, "adaptive")
    print(f"[DPSH] {dpsh[0]*100:.1f}% / {dpsh[1]:.3f} ({time.time()-T0:.0f}s)", flush=True)

    def post(feats):
        return np.array([dp.score(c, F, cnts, meta, th, ix.unigram, "adaptive")[0]
                         for (c, F, cnts, meta, tgt) in feats])
    P_r_d, P_r_t = post(fdev), post(ftest)

    # 融合 (dev 网格 mu, T)
    tgt_d = np.array([t for _, t in dev_s]); tgt_t = np.array([t for _, t in test_s])
    id_d, id_t = np.arange(len(tgt_d)), np.arange(len(tgt_t))
    clip = lambda P: np.clip(P, 1e-12, 1.0)
    print("[融合] dev 网格 mu,T ...", flush=True)
    best = (0.5, 1.0, 1e9)
    for T in [0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.5]:
        Pn_raw = P_rkv_d ** (1 / T)
        Pn = clip(Pn_raw / np.maximum(Pn_raw.sum(1, keepdims=True), 1e-12))
        for mu in np.arange(0, 1.01, 0.05):
            P = mu * P_r_d + (1 - mu) * Pn
            p = np.clip(P[id_d, tgt_d], 1e-12, 1.0)
            b = float(-np.log(p).mean() / math.log(2))
            if b < best[2]:
                best = (mu, T, b)
    mu, T, b_dev = best
    Pn_t = np.clip(P_rkv_t ** (1 / T) / np.maximum((P_rkv_t ** (1 / T)).sum(1, keepdims=True), 1e-12), 1e-12, 1.0)
    P_fuse = mu * P_r_t + (1 - mu) * Pn_t
    fuse = bits_of(P_fuse, test_s)

    results = {
        "DPSH(检索)": {"acc": dpsh[0], "bits": dpsh[1]},
        "RWKV-7 0.4B": {"acc": acc_r, "bits": bits_r},
        "融合:检索⊕RWKV-7": {"acc": fuse[0], "bits": fuse[1]},
        "mu": float(mu), "T": float(T),
        "ref_phase5_Qwen融合": {"acc": 0.290, "bits": 5.966},
    }
    json.dump(results, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print("\n" + "=" * 66)
    print("Phase 11 — 第一档 logit 融合 @ RWKV-7 (en 500k)")
    print(f"{'方法':<22}{'acc':>10}{'bits/token':>14}")
    print("-" * 66)
    print(f"{'DPSH(检索)':<22}{dpsh[0]*100:9.1f}%{dpsh[1]:14.3f}")
    print(f"{'RWKV-7 0.4B':<22}{acc_r*100:9.1f}%{bits_r:14.3f}")
    print(f"{'融合:检索⊕RWKV-7':<22}{fuse[0]*100:9.1f}%{fuse[1]:14.3f}")
    print("-" * 66)
    print(f"{'ref: 检索⊕Qwen(P5)':<22}{'29.0%':>10}{'5.966':>14}")
    print("=" * 66)
    print(f"mu={mu:.2f}, T={T} | dev bits={b_dev:.3f}")
    gain = (bits_r - fuse[1]) * 1000
    print(f"[裁定] 检索在 RWKV-7 之上增量: {gain:+.0f} mbits -> "
          f"{'第一档在 RWKV-7 上成立 ✓' if fuse[1] < min(bits_r, dpsh[1]) else '未超过单分支'}")
    print("已保存:", OUT)

if __name__ == "__main__":
    main()
