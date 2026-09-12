"""
DPSH (Differentiable Probabilistic Suffix Hypergraph) vs ROSA — 原型与合成基准

ROSA : 离散、无参数、最长精确后缀匹配 -> 硬复制后继 token。
DPSH : 检索保持离散高效（后缀索引 + 聚类超边），聚合全程连续可微
       （可学习多阶混合 + 语义近邻 + 超边归并 + 证据门控），
       输出完整 next-token 分布而非硬 token。

判别性任务
  T1 Verbatim  : 待预测上下文在历史中精确出现过   -> ROSA 强项，DPSH 必须不退化
  T2 Paraphrase: 上下文被同义词改写，精确匹配断裂 -> ROSA 必崩，检验语义泛化
  T3 Mixed     : 通用 next-token 预测

公平性：所有含自由参数的方法（PPM / DPSH 变体）在同一 dev split 上用 L-BFGS
        最小化 NLL 自动拟合，不手工调参。ROSA / ROSA+WB 无参。
"""

import json
import math
import os
import numpy as np
from collections import Counter, defaultdict
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(HERE, "results")

SEED = 20260911
VARIANTS_FULL = False   # True: 追加聚类准则消融（下一步实验，尚未跑完）
rng = np.random.default_rng(SEED)

# ---------------------------------------------------------------------------
# 1. 合成语料：filler 模板 + 同义词组 + 事实句
# ---------------------------------------------------------------------------

N_FILLER = 60
N_GROUP = 120
GROUP_SIZE = 3
N_VALUE = 120
N_TPL = 10          # filler 模板数
PARA_LEN = 6

FILLER = list(range(N_FILLER))
KEY = np.array([[N_FILLER + g * GROUP_SIZE + j for j in range(GROUP_SIZE)]
                for g in range(N_GROUP)])
Q0 = N_FILLER + N_GROUP * GROUP_SIZE          # 问句骨架标记（filler 中永不出现）
Q1 = Q0 + 1
VALUE0 = Q1 + 1
VOCAB = VALUE0 + N_VALUE
VALUE = {g: VALUE0 + g for g in range(N_GROUP)}

# 预生成 filler 模板：每槽位 = 固定 filler 词 或 key 槽
TPL = []
for t in range(N_TPL):
    slots = []
    for s in range(PARA_LEN):
        if rng.random() < 0.45:
            slots.append(("K", None))
        else:
            slots.append(("F", int(FILLER[(t * 7 + s * 13) % N_FILLER])))
    TPL.append(slots)


def gen_filler_para():
    """主题一致性：一段话内的 key 槽都取自同一同义组（自然语言普遍性质）。

    这样同组 key 共享几乎相同的上下文分布 -> PPMI 嵌入才能学到同义关系。
    """
    topic = int(rng.integers(N_GROUP))
    out = []
    for kind, v in TPL[rng.integers(N_TPL)]:
        if kind == "F":
            out.append(v)
        else:
            out.append(int(KEY[topic, rng.integers(GROUP_SIZE)]))
    return out


def build_corpus(n_hist=60000, seen_keys_per_group=2, fact_ratio=0.32):
    """事实句形如 [Q0, Q1, K, V]：每组只有 seen_keys_per_group 个 key 作主语。

    fact_ratio 控制事实句密度 -> 稀疏场景下单个上下文节点的证据量变小，
    用于检验"超边共享统计"究竟在什么条件下有价值。
    """
    hist, fact = [], {}
    for g in range(N_GROUP):
        chosen = rng.choice(GROUP_SIZE, size=seen_keys_per_group, replace=False)
        fact[g] = [int(KEY[g, j]) for j in chosen]
    while len(hist) < n_hist:
        if rng.random() < fact_ratio:
            g = int(rng.integers(N_GROUP))
            hist += [Q0, Q1, int(rng.choice(fact[g])), VALUE[g]]
        else:
            hist += gen_filler_para()
    return hist[:n_hist], fact


def build_eval_sets(fact, n_per_task):
    pad = [Q0, Q1]
    t1, t2 = [], []
    for g in range(N_GROUP):
        seen = set(fact[g])
        unseen = [int(k) for k in KEY[g] if int(k) not in seen]
        for _ in range(max(1, n_per_task // N_GROUP)):
            if seen:
                t1.append((pad + [int(rng.choice(sorted(seen)))], VALUE[g]))
            if unseen:
                t2.append((pad + [int(rng.choice(unseen))], VALUE[g]))
    t3 = []
    while len(t3) < n_per_task:
        para = gen_filler_para()
        i = int(rng.integers(1, len(para)))
        t3.append((para[:i], para[i]))
    return t1, t2, t3


# ---------------------------------------------------------------------------
# 2. 分布语义嵌入（PPMI + SVD）
# ---------------------------------------------------------------------------

def build_embeddings(corpus, dim=64, window=5):
    co = np.zeros((VOCAB, VOCAB), dtype=np.float32)
    c = np.asarray(corpus)
    for d in range(1, window + 1):
        a, b = c[:-d], c[d:]
        w = np.float32(1.0 / d)
        np.add.at(co, (a, b), w)
        np.add.at(co, (b, a), w)
    tot = co.sum()
    row = co.sum(1, keepdims=True)
    col = co.sum(0, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        pmi = np.log((co * tot + 1e-9) / (row * col + 1e-9))
    ppmi = np.maximum(pmi, 0.0).astype(np.float32)
    U, S, _ = np.linalg.svd(ppmi, full_matrices=False)
    E = (U[:, :dim] * S[:dim] ** 0.5).astype(np.float32)
    return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)


# ---------------------------------------------------------------------------
# 3. 历史索引：后缀计数 + 上下文节点 + 超边（聚类归并）
# ---------------------------------------------------------------------------

L_MAX = 5
MIN_COUNT = 2
ALPHA = 0.15


class Index:
    def __init__(self, corpus, E, decay=0.7, K=800):
        self.E = E
        self.decay = decay
        self.succ = defaultdict(Counter)
        for l in range(1, L_MAX + 1):
            for i in range(l, len(corpus)):
                self.succ[tuple(corpus[i - l:i])][corpus[i]] += 1
        ug = np.full(VOCAB, 0.1)
        for t, c in Counter(corpus).items():
            ug[t] += c
        self.unigram = ug / ug.sum()
        self._dc = {}                       # 后继分布缓存
        self.nodes = [c for c, cnt in self.succ.items()
                      if sum(cnt.values()) >= MIN_COUNT]
        self.node_emb = self._emb(self.nodes)
        self.node_cnt = np.array([sum(self.succ[c].values()) for c in self.nodes],
                                 dtype=np.float64)
        self.node_dist = np.stack([self.dist(c) for c in self.nodes])
        self._cluster(K)

    def dist(self, ctx):
        """平滑后的后继分布。"""
        r = self._dc.get(ctx)
        if r is None:
            cnt = self.succ.get(ctx)
            d = np.full(VOCAB, ALPHA * self.unigram, dtype=np.float32)
            n = 0
            if cnt:
                for t, v in cnt.items():
                    d[t] += v
                n = sum(cnt.values())
            r = (d / (n + ALPHA)).astype(np.float32)
            self._dc[ctx] = r
        return r

    def _emb(self, contexts):
        """顺序敏感的后缀嵌入：右对齐拼接（最近 token 固定在最后一维）。

        加权求和会退化成 bag-of-words，使 (x,y,Q0,Q1) 与 (Q0,Q1,k) 无法区分，
        污染近邻检索；右对齐拼接保留位置信息，是后缀语义的正确表示。
        """
        d = self.E.shape[1]
        out = np.zeros((len(contexts), d * L_MAX), dtype=np.float32)
        w = self.decay ** np.arange(L_MAX - 1, -1, -1)      # 越近权重越大
        for i, c in enumerate(contexts):
            c = list(c)[-L_MAX:]
            for j, tok in enumerate(c):                      # 右对齐
                p = L_MAX - len(c) + j
                out[i, p * d:(p + 1) * d] = self.E[tok] * w[p]
        return out / (np.linalg.norm(out, axis=1, keepdims=True) + 1e-9)

    def _cluster(self, K, method="ctx"):
        """method: ctx=按上下文嵌入聚类 | succ=按后继分布聚类 | both=两者拼接。

        检索一律走上下文嵌入质心；聚类准则决定"哪些上下文共享同一条超边"。
        CDAWG 的 endpos 等价类提示：正确的并合应基于后继分布而非表面上下文。
        """
        from sklearn.cluster import MiniBatchKMeans
        K = min(K, max(2, len(self.nodes) // 4))
        if method == "ctx":
            X = self.node_emb
        elif method == "succ":
            X = self.node_dist
        else:
            X = np.hstack([self.node_emb, self.node_dist]).astype(np.float32)
        km = MiniBatchKMeans(n_clusters=K, random_state=0, batch_size=2048,
                             n_init=3).fit(X)
        self.labels = km.labels_
        self.K = K
        # 检索质心始终取上下文嵌入的簇内均值
        cent = np.zeros((K, self.node_emb.shape[1]), dtype=np.float32)
        for k in range(K):
            m = self.labels == k
            cent[k] = self.node_emb[m].mean(0) if m.any() else 0.0
        self.cent = cent / (np.linalg.norm(cent, axis=1, keepdims=True) + 1e-9)
        self._rebuild_hyperedges()

    def _rebuild_hyperedges(self):
        K = self.K
        self.he_dist = np.zeros((K, VOCAB), dtype=np.float64)
        self.he_cnt = np.zeros(K)
        _o = np.argsort(self.labels)
        _b = np.searchsorted(self.labels[_o], np.arange(K + 1))
        self.members = [_o[_b[k]:_b[k + 1]] for k in range(K)]
        for k in range(K):
            m = self.members[k]
            w = self.node_cnt[m]
            if w.sum() > 0:
                self.he_cnt[k] = w.sum()
                self.he_dist[k] = (self.node_dist[m] * w[:, None]).sum(0) / w.sum()

    def query(self, ctx, C=6, S=4, use_nn=True):
        """候选 = (d_node, d_hyperedge, order_idx, cos, node_count)
        order_idx = L_MAX 表示近邻候选；两种分布都返回，由 score 按证据自适应混合。"""
        out = []
        for l in range(1, L_MAX + 1):
            key = tuple(ctx[-l:])
            if key in self.succ:
                d = self.dist(key)
                out.append((d, d, l - 1, 1.0, float(sum(self.succ[key].values()))))
        if use_nn:
            q = self._emb([ctx])[0]
            sims = self.cent @ q
            top = np.argpartition(-sims, min(C, self.K - 1))[:C]
            for k in top:
                mem = self.members[k]
                if len(mem) == 0:
                    continue
                ms = self.node_emb[mem] @ q
                for j in mem[np.argsort(-ms)[:S]]:
                    cos = float(self.node_emb[j] @ q)
                    out.append((self.node_dist[j], self.he_dist[k], L_MAX,
                                max(cos, 0.0), float(self.node_cnt[j])))
        return out


# ---------------------------------------------------------------------------
# 4. 候选特征与打分（候选集与参数无关 -> 可预计算，优化极快）
# ---------------------------------------------------------------------------

NF = L_MAX + 3      # [阶one-hot(L_MAX) | 近邻标志 | cos | log1p(count)]


def featurize(cands):
    """只构造特征矩阵；分布矩阵保留为引用，打分时才组装（避免 OOM）。"""
    m = len(cands)
    F = np.zeros((m, NF), dtype=np.float32)
    cnts = np.zeros(m, dtype=np.float32)
    for i, (_, _, oi, cos, cnt) in enumerate(cands):
        if oi < L_MAX:
            F[i, oi] = 1.0
        else:
            F[i, L_MAX] = 1.0
        F[i, L_MAX + 1] = cos
        F[i, L_MAX + 2] = math.log1p(cnt)
        cnts[i] = cnt
    return F, cnts


def make_batch(feats):
    """把一批样本拼成扁平矩阵，供整批向量化前向使用（分段 softmax）。"""
    Dn = np.concatenate([np.stack([c[0] for c in cs]) for cs, _, _, _, _ in feats])
    Dh = np.concatenate([np.stack([c[1] for c in cs]) for cs, _, _, _, _ in feats])
    F = np.vstack([f for _, f, _, _, _ in feats])
    cnts = np.concatenate([cn for _, _, cn, _, _ in feats])
    lens = np.array([len(cs) for cs, _, _, _, _ in feats])
    offs = np.concatenate([[0], np.cumsum(lens)[:-1]])
    mlen = np.array([m[0] for _, _, _, m, _ in feats])
    nmax = np.array([m[1] for _, _, _, m, _ in feats])
    tgt = np.array([t for _, _, _, _, t in feats])
    return dict(Dn=Dn, Dh=Dh, F=F, cnts=cnts, lens=lens, offs=offs,
                mlen=mlen, nmax=nmax, tgt=tgt)


def batch_forward(B, theta, unigram, mode):
    """theta = [beta, gamma, b(0..L_MAX), lam_a, lam_b, lam_c, tau]"""
    D = merge(B["Dn"], B["Dh"], B["cnts"], mode, theta[2 + L_MAX + 4])
    b = theta[2:2 + L_MAX + 1]
    s = (B["F"][:, :L_MAX + 1] @ b
         + theta[0] * B["F"][:, L_MAX + 1] + theta[1] * B["F"][:, L_MAX + 2])
    offs, lens = B["offs"], B["lens"]
    mx = np.maximum.reduceat(s, offs)
    e = np.exp(s - np.repeat(mx, lens))
    w = e / np.repeat(np.add.reduceat(e, offs), lens)
    P = np.add.reduceat(w[:, None] * D, offs, axis=0)          # (S, V)
    lp = theta[2 + L_MAX + 1: 2 + L_MAX + 4]
    lam = 1.0 / (1.0 + np.exp(-(lp[0] + lp[1] * np.log1p(B["nmax"])
                                + lp[2] * B["mlen"])))
    P = lam[:, None] * P + (1 - lam)[:, None] * unigram
    P = P / P.sum(1, keepdims=True)
    tgt = B["tgt"]
    acc = float(np.mean(np.argmax(P, 1) == tgt))
    bits = float(np.mean(-np.log(np.clip(P[np.arange(len(tgt)), tgt], EPS, 1.0)))
                 / math.log(2))
    return acc, bits


def evaluate_fast(feats, theta, unigram, mode, chunk=400):
    tot_a = tot_b = tot_n = 0.0
    for i in range(0, len(feats), chunk):
        B = make_batch(feats[i:i + chunk])
        a, b = batch_forward(B, theta, unigram, mode)
        n = len(B["tgt"])
        tot_a += a * n
        tot_b += b * n
        tot_n += n
    return tot_a / tot_n, tot_b / tot_n


def select_multi_order(cands, multi_order):
    if multi_order:
        return cands
    exact = [c for c in cands if c[2] < L_MAX]
    nn = [c for c in cands if c[2] == L_MAX]
    keep = [max(exact, key=lambda c: c[2])] if exact else []
    return keep + nn[:1]


def merge(Dn, Dh, cnts, mode, tau):
    """证据自适应超边归并：证据越弱越依赖超边的共享统计。"""
    if mode == "node":
        return Dn
    if mode == "hyperedge":
        return Dh
    rho = 1.0 / (1.0 + np.exp(-(np.log1p(cnts) - tau) / 0.5))   # 证据足 -> 用自身
    return rho[:, None] * Dn + (1 - rho)[:, None] * Dh


def score(cands, F, cnts, meta, theta, unigram, mode="adaptive"):
    """theta = [beta, gamma, b(0..L_MAX), lam_a, lam_b, lam_c, tau]"""
    Dn = np.stack([c[0] for c in cands])
    Dh = np.stack([c[1] for c in cands])
    beta, gamma = theta[0], theta[1]
    b = theta[2:2 + L_MAX + 1]
    lam_par = theta[2 + L_MAX + 1: 2 + L_MAX + 4]
    tau = theta[2 + L_MAX + 4]
    D = merge(Dn, Dh, cnts, mode, tau)
    s = F[:, :L_MAX + 1] @ b + beta * F[:, L_MAX + 1] + gamma * F[:, L_MAX + 2]
    s = s - s.max()
    w = np.exp(s)
    w /= w.sum()
    P = w @ D
    mlen, nmax = meta
    lam = 1.0 / (1.0 + math.exp(-(lam_par[0] + lam_par[1] * math.log1p(nmax)
                                  + lam_par[2] * mlen)))
    P = lam * P + (1 - lam) * unigram
    return P / P.sum(), lam


def meta_of(cands):
    exact = [c for c in cands if c[2] < L_MAX]
    mlen = max([c[2] + 1 for c in exact], default=0) / L_MAX
    nmax = max([c[4] for c in exact], default=0.0)
    return mlen, nmax


# ---------------------------------------------------------------------------
# 5. 基线模型
# ---------------------------------------------------------------------------

EPS = 1e-12


def rosa_predict(ctx, ix, backoff="wb"):
    """最长精确后缀匹配 -> 硬复制后继 token。"""
    for l in range(L_MAX, 0, -1):
        key = tuple(ctx[-l:])
        if key in ix.succ:
            cnt = ix.succ[key]
            tok = max(cnt.items(), key=lambda kv: kv[1])[0]
            P = np.full(VOCAB, EPS)
            P[tok] = 1.0
            return P / P.sum()
    if backoff == "wb":
        return ix.unigram.copy()
    return np.full(VOCAB, 1.0 / VOCAB)      # abstain -> 均匀


def ppm_predict(ctx, ix, w):
    """变阶插值 n-gram（经典 PPM 式基线）。"""
    ws = 1.0 / (1.0 + np.exp(-np.asarray(w)))
    ws = ws / ws.sum()
    out = np.zeros(VOCAB)
    tot = 0.0
    for l in range(1, L_MAX + 1):
        key = tuple(ctx[-l:])
        if key in ix.succ:
            out += ws[l - 1] * ix.dist(key)
            tot += ws[l - 1]
    out += ws[L_MAX] * ix.unigram
    return out / max(tot + ws[L_MAX], EPS)


# ---------------------------------------------------------------------------
# 6. 评测
# ---------------------------------------------------------------------------

def evaluate(pred_fn, data):
    acc = nll = 0.0
    for ctx, tgt in data:
        P = np.clip(pred_fn(ctx), EPS, 1.0)
        P /= P.sum()
        acc += int(np.argmax(P) == tgt)
        nll += -math.log(P[tgt])
    n = len(data)
    return acc / n, nll / n / math.log(2)


def evaluate_pre(pred_fn, feats):
    acc = nll = 0.0
    for (cands, F, cnts, meta, tgt) in feats:
        P = np.clip(pred_fn(cands, F, cnts, meta), EPS, 1.0)
        P /= P.sum()
        acc += int(np.argmax(P) == tgt)
        nll += -math.log(P[tgt])
    n = len(feats)
    return acc / n, nll / n / math.log(2)


def fit(feats_dev, n_params, init, unigram, mode, iters=40):
    """在预计算候选上拟合（DPSH 变体），整批向量化前向。"""
    B = make_batch(feats_dev)
    def obj(x):
        return batch_forward(B, x, unigram, mode)[1]
    return minimize(obj, init, method="L-BFGS-B",
                    options={"maxiter": iters, "maxfun": iters * 5}).x


def fit_raw(builder, dev, n_params, init, iters=18):
    """在原始样本上拟合（PPM 等无预计算候选的方法）。"""
    def obj(x):
        f = builder(x)
        return evaluate(f, dev)[1]
    return minimize(obj, init, method="L-BFGS-B",
                    options={"maxiter": iters, "maxfun": iters * 5}).x


# ---------------------------------------------------------------------------
# 7. 主流程
# ---------------------------------------------------------------------------

def run_scenario(tag, fact_ratio, n_per_task=3000):
    print(f"\n===== 场景 {tag}：事实句比例 {fact_ratio} =====")
    hist, fact = build_corpus(fact_ratio=fact_ratio)
    train = hist[5000:]
    E = build_embeddings(train)
    ix = Index(train, E)

    # 证据密度诊断：事实句上下文节点的平均观测次数
    fcounts = [sum(ix.succ[(Q0, Q1, k)].values())
               for g in range(N_GROUP) for k in fact[g] if (Q0, Q1, k) in ix.succ]
    print(f"  事实句上下文节点平均证据量: {np.mean(fcounts):.1f} 次")

    t1, t2, t3 = build_eval_sets(fact, n_per_task=n_per_task)
    dev = t1[:200] + t2[:200] + t3[:200]
    test = {"T1_verbatim": t1[200:], "T2_paraphrase": t2[200:], "T3_mixed": t3[200:]}

    results = {}
    N_THETA = 2 + (L_MAX + 1) + 3 + 1          # +tau（证据自适应归并阈值）
    init0 = np.r_[1.0, 0.5, np.zeros(L_MAX + 1), 0.0, 0.5, 1.0, 2.0]
    assert len(init0) == N_THETA

    results["Unigram"] = {k: evaluate(lambda c: ix.unigram, v) for k, v in test.items()}
    results["ROSA-hard"] = {k: evaluate(lambda c: rosa_predict(c, ix, "none"), v)
                            for k, v in test.items()}
    results["ROSA+WB"] = {k: evaluate(lambda c: rosa_predict(c, ix, "wb"), v)
                          for k, v in test.items()}
    w_ppm = fit_raw(lambda x: (lambda c: ppm_predict(c, ix, x)), dev, L_MAX + 1,
                    np.ones(L_MAX + 1))
    results["PPM(变阶n-gram)"] = {k: evaluate(lambda c: ppm_predict(c, ix, w_ppm), v)
                                  for k, v in test.items()}

    # baseline：与 results_dense.json / results_sparse.json 对应的已验证变体
    variants = {
        "DPSH w/o近邻":         dict(use_nn=False, multi_order=True,
                                    mode="adaptive", cluster="ctx"),
        "DPSH w/o多阶":         dict(use_nn=True, multi_order=False,
                                    mode="adaptive", cluster="ctx"),
        "DPSH 硬超边":          dict(use_nn=True, multi_order=True,
                                    mode="hyperedge", cluster="ctx"),
        "DPSH w/o超边":         dict(use_nn=True, multi_order=True,
                                    mode="node", cluster="ctx"),
        "DPSH-full(自适应超边)": dict(use_nn=True, multi_order=True,
                                    mode="adaptive", cluster="ctx"),
    }
    if VARIANTS_FULL:      # 下一阶段：聚类准则消融（已实现，尚未跑完）
        variants.update({
            "DPSH 自适应(后继聚类)": dict(use_nn=True, multi_order=True,
                                       mode="adaptive", cluster="succ"),
            "DPSH 自适应(混合聚类)": dict(use_nn=True, multi_order=True,
                                       mode="adaptive", cluster="both"),
        })
    for name, kw in variants.items():
        mode = kw["mode"]
        ix._cluster(ix.K, kw["cluster"])

        def mk(data, kw=kw):
            feats = []
            for ctx, tgt in data:
                c = select_multi_order(
                    ix.query(ctx, C=6, S=4, use_nn=kw["use_nn"]), kw["multi_order"])
                if not c:
                    feats.append(([(ix.unigram, ix.unigram, L_MAX, 0.0, 0.0)],
                                  np.zeros((1, NF), dtype=np.float32),
                                  np.zeros(1, dtype=np.float32), (0.0, 0.0), tgt))
                    continue
                F, cnts = featurize(c)
                feats.append((c, F, cnts, meta_of(c), tgt))
            return feats

        fdev = mk(dev)
        ftest = {k: mk(v) for k, v in test.items()}
        th = fit(fdev, N_THETA, init0, ix.unigram, mode)
        results[name] = {k: evaluate_fast(v, th, ix.unigram, mode)
                         for k, v in ftest.items()}

    # 退化验证：关近邻 + 单阶 + 超低温 -> 应收敛到 ROSA 的硬复制
    def degen(ctx):
        c = select_multi_order(ix.query(ctx, C=6, S=4, use_nn=False), False)
        if not c:
            return np.full(VOCAB, 1.0 / VOCAB)
        F, cnts = featurize(c)
        th = np.zeros(N_THETA)
        th[2 + L_MAX - 1] = 80.0        # 只信最长精确阶
        th[2 + L_MAX + 1] = 10.0        # lambda -> 1（完全信检索）
        P, _ = score(c, F, cnts, meta_of(c), th, ix.unigram, "node")
        return P
    results["DPSH→ROSA退化"] = {k: evaluate(degen, v) for k, v in test.items()}
    return results


ORDER = ["Unigram", "ROSA-hard", "ROSA+WB", "PPM(变阶n-gram)",
         "DPSH w/o近邻", "DPSH w/o多阶", "DPSH 硬超边", "DPSH w/o超边",
         "DPSH-full(自适应超边)", "DPSH→ROSA退化"]
if VARIANTS_FULL:
    ORDER += ["DPSH 自适应(后继聚类)", "DPSH 自适应(混合聚类)"]
TASKS = ["T1_verbatim", "T2_paraphrase", "T3_mixed"]


def print_table(tag, res):
    print("\n" + "=" * 88)
    print(f"场景：{tag}")
    print(f"{'方法':<22}" + "".join(f"{t.split('_')[1]:>22}" for t in TASKS))
    print(f"{'':<22}" + "".join(f"{'acc / bits':>22}" for _ in TASKS))
    print("-" * 88)
    for name in ORDER:
        row = f"{name:<22}"
        for t in TASKS:
            a, b = res[name][t]
            row += f"{a*100:11.1f}% /{b:7.2f}"
        print(row)
    print("=" * 88)


def make_plot(all_res, scen):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(len(scen), 3, figsize=(19, 5.2 * len(scen)))
    axes = np.atleast_2d(axes)
    for r, (tag, _) in enumerate(scen):
        for c, (t, lab) in enumerate(zip(TASKS, ["T1 精确召回", "T2 改写召回",
                                                 "T3 混合预测"])):
            ax = axes[r][c]
            def _acc(v):
                return v["acc"] if isinstance(v, dict) else v[0]
            accs = np.array([_acc(all_res[tag][n][t]) * 100 for n in ORDER])
            cols = ["#d62728" if ("ROSA" in n) else
                    ("#2ca02c" if ("自适应" in n or n.startswith("DPSH-full")) else
                     ("#7f7f7f" if n == "Unigram" else "#1f77b4")) for n in ORDER]
            y = np.arange(len(ORDER))
            ax.barh(y, accs, color=cols)
            ax.set_yticks(y)
            ax.set_yticklabels(ORDER, fontsize=8)
            ax.set_xlabel("Top-1 准确率 (%)")
            ax.set_title(f"[{tag}] {lab}", fontsize=11)
            ax.set_xlim(0, 118)
            for i, v in enumerate(accs):
                ax.text(v + 1.5, i, f"{v:.1f}", va="center", fontsize=7.5)
            ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "dpsh_results.png"), dpi=150)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="dense",
                    choices=["dense", "sparse", "plot"])
    ap.add_argument("--full", action="store_true",
                    help="追加聚类准则消融（下一步实验，尚未跑完）")
    args = ap.parse_args()
    global VARIANTS_FULL
    VARIANTS_FULL = args.full
    if args.full and "DPSH 自适应(后继聚类)" not in ORDER:
        ORDER.extend(["DPSH 自适应(后继聚类)", "DPSH 自适应(混合聚类)"])
    scen_map = {"dense": ("密集", 0.32), "sparse": ("稀疏", 0.05)}

    if args.scenario == "plot":
        all_res = {}
        for key, (tag, _) in scen_map.items():
            all_res[tag] = json.load(open(os.path.join(RESULTS_DIR, f"results_{key}.json"), encoding="utf-8"))
        make_plot(all_res, list(scen_map.values()))
        print("图已更新:", os.path.join(RESULTS_DIR, "dpsh_results.png"))
        return

    tag, fr = scen_map[args.scenario]
    res = run_scenario(tag, fr)
    print_table(tag, res)
    with open(os.path.join(RESULTS_DIR, f"results_{args.scenario}.json"), "w", encoding="utf-8") as f:
        json.dump({n: {t: {"acc": res[n][t][0], "bits": res[n][t][1]}
                       for t in TASKS} for n in ORDER}, f, indent=2, ensure_ascii=False)
    print(f"已保存 results_{args.scenario}.json")


if __name__ == "__main__":
    main()
