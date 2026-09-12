# -*- coding: utf-8 -*-
"""
DPSH Phase 5: 真预训练 LM 当神经分支 (Qwen2.5-0.5B) — 最终配方验证
问题: 检索后验在一个真正预训练 LM 之上还有没有增量? (README: "真实系统应与 LM logits 融合")
方案:
  - en 500k (与 phase4 同切片), PPMI+SVD 索引, DPSH w/o近邻 -> P_r
  - Qwen 分支: 每位置取 Qwen last-position 分布, 按词首BPE id 聚合到 5000 词表 -> P_q
  - GRU 分支: 重训 (同 phase3/4 配方) -> P_g
  - dev 网格拟合: 检索⊕Qwen; 检索⊕GRU⊕Qwen (三路)
对照: DPSH / Qwen / GRU 单分支, phase4 的 GRU 融合 (6.855)
"""
import os, sys, io, json, math, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
OUT = os.path.join(PROJ, "results", "results_real_phase5.json")
SEED = 20260912
torch.manual_seed(SEED); np.random.seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
CTX, V, SCALE, NCTX_Q = 16, 5000, 500000, 32
T0 = time.time()

from dpsh_real_pg19 import tokenize as en_tokenize, build_embeddings_sparse

# ---------------- 数据 (phase4 同切片: start=0) ----------------

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

# ---------------- GRU 分支 (phase3/4 同配方) ----------------

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
        print(f"    epoch {ep+1}: loss {tot/nb:.4f} ({time.time()-T0:.0f}s)")
    return m

@torch.no_grad()
def gru_probs(m, data_s, T=1.0):
    out = []
    for i in range(0, len(data_s), 4096):
        x = torch.tensor(np.array([c for c, _ in data_s[i:i + 4096]], dtype=np.int64), device=DEV)[:, -CTX:]
        out.append(torch.softmax(m(x) / T, -1).cpu().numpy())
    return np.concatenate(out)

# ---------------- Qwen 分支: 首BPE聚合 ----------------

def qwen_probs(data_s, data_ids, vocab):
    """每位置: 文本(后32词) -> Qwen last logits -> 按 词首BPEid 聚合 -> (N,V) 概率"""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    tok.padding_side = "left"          # 关键: batch 推理取 [-1] 位置必须是真实最后 token
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",
                                             dtype=torch.float16).to(DEV).eval()
    # 词 -> 首BPE id (" "+w, en 惯例)
    first_id = np.full(V, -1, dtype=np.int64)
    for i, w in enumerate(vocab[1:], 1):
        ids = tok.encode(" " + w, add_special_tokens=False)
        if ids:
            first_id[i] = ids[0]
    first_id[0] = tok.encode(" <unk>", add_special_tokens=False)[0] if tok.encode(" <unk>", add_special_tokens=False) else 0
    words = [vocab[i] for i in data_ids]
    pos = [int(i) for i in np.linspace(5, len(data_ids) - 2, len(data_s)).astype(int)]
    texts = [" ".join(words[max(0, p - NCTX_Q):p]) for p in pos]
    P = np.zeros((len(texts), V), dtype=np.float32)
    bs = 128
    with torch.no_grad():
        for i in range(0, len(texts), bs):
            enc = tok(texts[i:i + bs], return_tensors="pt", padding=True,
                      truncation=True, max_length=256).to(DEV)
            logits = m(**enc).logits[:, -1].float()
            prob = torch.softmax(logits, -1).cpu().numpy()
            mask = first_id >= 0
            P[i:i + bs][:, mask] = prob[:, first_id[mask]]
            P[i:i + bs] /= np.maximum(P[i:i + bs].sum(1, keepdims=True), 1e-12)
            if (i // bs) % 5 == 0:
                print(f"    qwen {i}/{len(texts)} ({time.time()-T0:.0f}s)")
    del m; torch.cuda.empty_cache()
    return P

def bits_of(P, data_s):
    tgt = np.array([t for _, t in data_s])
    p = np.clip(P[np.arange(len(tgt)), tgt], 1e-12, 1.0)
    return float(np.mean(np.argmax(P, 1) == tgt)), float(-np.log(p).mean() / math.log(2))

# ---------------- 主流程 ----------------

def main():
    import dpsh_experiment as dp
    tr, dv, te, vocab = build_en_cell()
    dev_s, test_s = stream(dv, 1500), stream(te, 4000)
    print(f"[数据] en 500k, dev {len(dev_s)}, test {len(test_s)} ({time.time()-T0:.0f}s)")

    # GRU 分支
    print("[GRU] 训练 ...")
    m = train_nnlm(tr); m.eval()
    P_g_d = gru_probs(m, dev_s); P_g_t = gru_probs(m, test_s)
    acc_g, bits_g = bits_of(P_g_t, test_s)
    print(f"  GRU: {acc_g*100:.1f}% / {bits_g:.3f} ({time.time()-T0:.0f}s)")

    # Qwen 分支
    print("[Qwen] 预计算分支分布 ...")
    P_q_d = qwen_probs(dev_s, dv, vocab)
    P_q_t = qwen_probs(test_s, te, vocab)
    acc_q, bits_q = bits_of(P_q_t, test_s)
    print(f"  Qwen: {acc_q*100:.1f}% / {bits_q:.3f} ({time.time()-T0:.0f}s)")

    # 检索分支 (PPMI 嵌入索引, phase1 配方)
    dp.VOCAB = V; dp.MIN_COUNT = 4
    print("[索引] PPMI+SVD ...")
    E = build_embeddings_sparse(list(map(int, tr)), vocab, dim=64)
    ix = dp.Index(list(map(int, tr)), E, decay=0.7, K=2500)
    print(f"  节点 {len(ix.nodes):,} ({time.time()-T0:.0f}s)")

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
    print(f"[DPSH] {dpsh[0]*100:.1f}% / {dpsh[1]:.3f} ({time.time()-T0:.0f}s)")

    def post(feats):
        return np.array([dp.score(c, F, cnts, meta, th, ix.unigram, "adaptive")[0]
                         for (c, F, cnts, meta, tgt) in feats])
    P_r_d, P_r_t = post(fdev), post(ftest)

    # 融合拟合
    tgt_d = np.array([t for _, t in dev_s]); tgt_t = np.array([t for _, t in test_s])
    id_d, id_t = np.arange(len(tgt_d)), np.arange(len(tgt_t))
    clip = lambda P: np.clip(P, 1e-12, 1.0)

    def fit_mix(bases_d):
        """网格拟合各分支权重 (步长0.05, 权重和=1) + T 对 Qwen/GRU"""
        best = (None, 1e9)
        n = len(bases_d)
        if n == 2:
            grids = [(mu, 1 - mu) for mu in np.arange(0, 1.01, 0.05)]
            Ts = [1.0]
        else:
            grids = [(a, b, 1 - a - b) for a in np.arange(0, 1.01, 0.1)
                     for b in np.arange(0, 1.01 - a, 0.1)]
            Ts = [0.9, 1.0]
        for T in Ts:
            bd = [clip(bases_d[0]), clip(bases_d[1] ** (1 / T) if False else bases_d[1])]
            for w in grids:
                P = sum(wi * clip(bd_i) for wi, bd_i in zip(w, bases_d))
                p = np.clip(P[id_d, tgt_d], 1e-12, 1.0)
                b = float(-np.log(p).mean() / math.log(2))
                if b < best[1]:
                    best = (w, b)
        return best

    # 检索 ⊕ Qwen
    (w2, b2) = fit_mix([P_r_d, P_q_d])
    P_f2 = w2[0] * clip(P_r_t) + w2[1] * clip(P_q_t)
    f2 = bits_of(P_f2, test_s)
    # 检索 ⊕ GRU ⊕ Qwen
    (w3, b3) = fit_mix([P_r_d, P_g_d, P_q_d])
    P_f3 = w3[0] * clip(P_r_t) + w3[1] * clip(P_g_t) + w3[2] * clip(P_q_t)
    f3 = bits_of(P_f3, test_s)

    results = {
        "DPSH(检索)": dpsh, "GRU(GRU分支)": (acc_g, bits_g), "Qwen(0.5B)": (acc_q, bits_q),
        "融合:检索⊕Qwen": f2, "融合:检索⊕GRU⊕Qwen": f3,
        "w2": [float(x) for x in w2], "w3": [float(x) for x in w3],
        "ref_phase4_GRU融合": {"acc": 0.201, "bits": 6.855},
    }
    json.dump(results, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)

    print("\n" + "=" * 76)
    print("Phase 5 — 真预训练 LM 分支 (en 500k)")
    print(f"{'方法':<24}{'acc':>10}{'bits/token':>14}")
    print("-" * 76)
    for n in ["DPSH(检索)", "GRU(GRU分支)", "Qwen(0.5B)",
              "融合:检索⊕Qwen", "融合:检索⊕GRU⊕Qwen"]:
        a, b = results[n]
        print(f"{n:<24}{a*100:9.1f}%{b:14.3f}")
    print("-" * 76)
    print(f"{'phase4 融合(检索⊕GRU)':<24}{'20.1%':>10}{'6.855':>14}")
    print("=" * 76)
    print(f"权重: 检索⊕Qwen = {np.round(w2,2)} | 检索⊕GRU⊕Qwen = {np.round(w3,2)}")
    gain_vs_q = (bits_q - f2[1]) * 1000
    print(f"[裁定] 检索在 Qwen 之上增量: {gain_vs_q:+.1f} mbits -> "
          f"{'有真实增量 ✓' if gain_vs_q > 5 else '增量微弱'}")
    print("已保存:", OUT)

if __name__ == "__main__":
    main()
