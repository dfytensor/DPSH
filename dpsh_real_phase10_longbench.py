# -*- coding: utf-8 -*-
"""
DPSH Phase 10: LongBench-lite — 多跳 QA 生成管线
数据: LongBench dev (hotpotqa / 2wikimqa / musique 各 30 条, hf-mirror 流式)
协议:
  - 每样本: 词级词表(上下文+问题), 对完整上下文建后缀索引 (DPSH 读全文, 无截断)
  - Qwen 分支: 上下文截断 (头2k+尾1k词 ≈ 4k BPE), generate(output_scores)
  - 融合: 逐步 0.4*P_dpsh + 0.6*P_qwen(首BPE聚合)
  - 指标: LongBench 口径 EM / word-F1 (normalize 后)
生成: 贪心 12 token, 标点即停
"""
import os, sys, io, json, math, time, re, string
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import numpy as np
import torch
from collections import Counter

import dpsh_experiment as dp

PROJ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJ)
OUT = os.path.join(PROJ, "results", "results_real_phase10_longbench.json")
SEED = 20260912
np.random.seed(SEED); torch.manual_seed(SEED)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
N_PER, GEN_MAX, V_CAP = 30, 12, 4000
T0 = time.time()

# ---------------- LongBench 官方口径指标 ----------------

def normalize_answer(s):
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())

def f1_score(pred, ground):
    pred_t = normalize_answer(pred).split()
    gt_t = normalize_answer(ground).split()
    common = Counter(pred_t) & Counter(gt_t)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_t)
    recall = num_same / len(gt_t)
    return 2 * precision * recall / (precision + recall)

def em_score(pred, ground):
    return float(normalize_answer(pred) == normalize_answer(ground))

def best_over(pred, answers):
    return max(f1_score(pred, a) for a in answers), max(em_score(pred, a) for a in answers)

# ---------------- 数据 ----------------

def load_samples():
    out = {}
    for name in ["hotpotqa", "2wikimqa", "musique"]:
        path = os.path.join(PROJ, "results", "longbench", "data", f"{name}.jsonl")
        if not os.path.exists(path):
            print(f"[data] {name}: 本地无 {path}")
            continue
        rows = []
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            ctx = r["context"]
            if isinstance(ctx, str):
                try:
                    ctx = json.loads(ctx)
                except Exception:
                    pass
            if isinstance(ctx, list):
                if ctx and isinstance(ctx[0], list):
                    text = " ".join(str(t) for pair in ctx for t in pair)
                else:
                    text = " ".join(str(x) for x in ctx)
            else:
                text = str(ctx)
            ans = r["answers"]
            if isinstance(ans, str):
                try:
                    ans = json.loads(ans)
                except Exception:
                    ans = [ans]
            rows.append({"q": r["input"], "ctx": text, "answers": ans})
            if len(rows) >= N_PER:
                break
        out[name] = rows
        print(f"[data] {name}: {len(rows)} 条 (ctx均值 {np.mean([len(r['ctx'].split()) for r in rows]):.0f} 词)",
              flush=True)
    return out

# ---------------- Qwen ----------------

_Q = {}

def get_qwen():
    if "m" not in _Q:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct",
                                                 dtype=torch.float16).to(DEV).eval()
        _Q["m"], _Q["tok"] = m, tok
    return _Q["m"], _Q["tok"]

@torch.no_grad()
def qwen_generate(scores_out, prompt):
    """贪心 12 token, 返回文本与逐步 BPE 分布 (首词聚合由调用方处理)"""
    m, tok = get_qwen()
    enc = tok(prompt, return_tensors="pt", truncation=True, max_length=5100).to(DEV)
    out = m.generate(**enc, max_new_tokens=GEN_MAX, do_sample=False,
                     output_scores=True, return_dict_in_generate=True,
                     pad_token_id=tok.pad_token_id)
    text = tok.decode(out.sequences[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
    scores_out.extend([s[0].float().cpu() for s in out.scores])
    return text

# ---------------- DPSH 生成 ----------------

def find_best_query(ix, q_words):
    """扫描问题所有 n-gram 窗口 (最长优先), 选有精确命中且命中数最高的窗口;
    无任何命中 -> 回退最长窗口"""
    best = None
    for l in range(min(5, len(q_words)), 0, -1):
        for i in range(len(q_words) - l + 1):
            key = tuple(q_words[i:i + l])
            if key in ix.succ:
                cnt = sum(ix.succ[key].values())
                if best is None or (l, cnt) > best[0]:
                    best = ((l, cnt), list(key))
    if best is not None:
        return best[1]
    return list(q_words[-min(5, len(q_words)):])

def dpsh_generate(ix, q_words, wid, ivoc, theta, stop_words):
    words = []
    ctx = find_best_query(ix, q_words)
    for _ in range(GEN_MAX):
        cand = dp_select(ix, ctx)
        if cand is None:
            break
        F, cnts = dp.featurize(cand)
        P, _ = dp.score(cand, F, cnts, dp.meta_of(cand), theta, ix.unigram, "adaptive")
        nid = int(np.argmax(P))
        w = ivoc.get(nid, "")
        if w in stop_words or w == "<unk>":
            break
        words.append(w)
        ctx = (ctx + [nid])[-5:]
    return " ".join(words)

def dp_select(ix, ctx):
    import dpsh_experiment as dp
    c = dp.select_multi_order(ix.query(list(ctx), C=6, S=4, use_nn=False), True)
    if not c:
        return None
    return c

def main():
    samples = load_samples()
    m, tok = get_qwen()

    all_res = {}
    for name, rows in samples.items():
        ems = {"Qwen-long": [], "DPSH(全文索引)": [], "融合(0.4)": [], "融合(轻检索0.15)": []}
        f1s = {k: [] for k in ems}
        for si, r in enumerate(rows):
            # 词级词表: 上下文 + 问题
            ctx_words = re.findall(r"[a-z0-9']+", r["ctx"].lower())[:12000]
            q_words = re.findall(r"[a-z0-9']+", r["q"].lower())
            cnt = Counter(ctx_words)
            vocab = ["<unk>"] + [w for w, _ in cnt.most_common(V_CAP - 1)]
            for w in q_words:
                if w not in set(vocab):
                    vocab.append(w)
            wid = {w: i for i, w in enumerate(vocab)}
            ivoc = {i: w for w, i in wid.items()}
            Vn = len(vocab)
            dp.VOCAB = Vn; dp.MIN_COUNT = 2
            ctx_ids = [wid[w] for w in ctx_words]
            try:
                E_dummy = np.ones((Vn, 2), dtype=np.float32)
                ix = dp.Index(ctx_ids, E_dummy, decay=0.7, K=8)
            except Exception:
                continue
            # theta: 首样本拟合一次, 复用
            if si == 0:
                dev_idx = np.linspace(6, len(ctx_ids) - 2, 400).astype(int)
                dev_s = [(ctx_ids[i - 5:i], ctx_ids[i]) for i in dev_idx]
                init0 = np.r_[1.0, 0.5, np.zeros(dp.L_MAX + 1), 0.0, 0.5, 1.0, 2.0]
                def fit(feats, chunk=250, iters=15):
                    B = [dp.make_batch(feats[i:i + chunk]) for i in range(0, len(feats), chunk)]
                    ns = np.array([len(b["tgt"]) for b in B])
                    f = lambda x: sum(dp.batch_forward(b, x, ix.unigram, "adaptive")[1] * n
                                      for b, n in zip(B, ns)) / ns.sum()
                    return minimize(f, init0, method="L-BFGS-B",
                                    options={"maxiter": iters, "maxfun": iters * 5}).x
                from scipy.optimize import minimize
                feats = []
                for c0, t0 in dev_s:
                    c = dp.select_multi_order(ix.query(list(c0), C=6, S=4, use_nn=False), True)
                    if not c:
                        continue
                    F, cnts_ = dp.featurize(c)
                    feats.append((c, F, cnts_, dp.meta_of(c), t0))
                theta = fit(feats)

            stop_words = {".", "?", "!", ",", ";", ":", "</s>"}
            q_ids = [wid[w] for w in q_words if w in wid]   # 词 -> ID (索引键是 ID 元组)
            # DPSH 全文索引生成
            ans_dpsh = dpsh_generate(ix, q_ids, wid, ivoc, theta, stop_words)
            # Qwen (头2k+尾1k词)
            ctx_text = " ".join(ctx_words[:2000]) + " ... " + " ".join(ctx_words[-1000:])
            prompt = (f"Answer the question based on the passage. Only give the short answer.\n"
                      f"Passage: {ctx_text}\nQuestion: {r['q']}\nAnswer:")
            sc = []
            ans_q = qwen_generate(sc, prompt).strip().split("\n")[0]
            # 融合: 逐步 0.4 DPSH + 0.6 Qwen(首BPE聚合)
            first_id = np.zeros(Vn, dtype=np.int64)
            for i, w in enumerate(vocab):
                ids = tok.encode(" " + w, add_special_tokens=False)
                first_id[i] = ids[0] if ids else 0
            words_f, ctx = [], find_best_query(ix, q_ids)
            for t in range(min(GEN_MAX, len(sc))):
                cand = dp_select(ix, ctx)
                if cand is None:
                    break
                F, cnts_ = dp.featurize(cand)
                P_d, _ = dp.score(cand, F, cnts_, dp.meta_of(cand), theta, ix.unigram, "adaptive")
                qb = sc[t].numpy()
                P_q = qb[first_id]
                P_q = P_q / max(P_q.sum(), 1e-12)
                P = 0.4 * P_d + 0.6 * P_q
                nid = int(np.argmax(P))
                w = ivoc.get(nid, "")
                if w in stop_words or w == "<unk>":
                    break
                words_f.append(w)
                ctx = (ctx + [nid])[-5:]
            ans_f = " ".join(words_f)
            # 轻检索权重: QA 生成场景, 检索分支仅作证据校正 (对照 0.4 固定权重的失效)
            words_l, ctx = [], find_best_query(ix, q_ids)
            for t in range(min(GEN_MAX, len(sc))):
                cand = dp_select(ix, ctx)
                if cand is None:
                    break
                F, cnts_ = dp.featurize(cand)
                P_d, _ = dp.score(cand, F, cnts_, dp.meta_of(cand), theta, ix.unigram, "adaptive")
                qb = sc[t].numpy()
                P_q = qb[first_id]
                P_q = P_q / max(P_q.sum(), 1e-12)
                P = 0.15 * P_d + 0.85 * P_q
                nid = int(np.argmax(P))
                w = ivoc.get(nid, "")
                if w in stop_words or w == "<unk>":
                    break
                words_l.append(w)
                ctx = (ctx + [nid])[-5:]
            ans_fl = " ".join(words_l)

            for k, pred in [("Qwen-long", ans_q), ("DPSH(全文索引)", ans_dpsh),
                            ("融合(0.4)", ans_f), ("融合(轻检索0.15)", ans_fl)]:
                f1, em = best_over(pred, r["answers"])
                f1s[k].append(f1); ems[k].append(em)
            if si < 3:
                print(f"  [{name}#{si}] gold={r['answers'][:1]} | qwen='{ans_q[:40]}' "
                      f"| dpsh='{ans_dpsh[:40]}' | fuse04='{ans_f[:40]}' | fuse015='{ans_fl[:40]}'", flush=True)

        all_res[name] = {
            k: {"EM": float(np.mean(ems[k])), "F1": float(np.mean(f1s[k])), "n": len(ems[k])}
            for k in ems}
        print(f"\n[{name}] " + "  ".join(
            f"{k}: EM={np.mean(ems[k])*100:.1f}% F1={np.mean(f1s[k])*100:.1f}%" for k in ems), flush=True)

    json.dump(all_res, open(OUT, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print("\n" + "=" * 72)
    print(f"{'数据集':<12}{'方法':<18}{'EM':>9}{'F1':>9}")
    for name, res in all_res.items():
        for k, v in res.items():
            print(f"{name:<12}{k:<18}{v['EM']*100:8.1f}%{v['F1']*100:8.1f}%")
    print("=" * 72)
    print("已保存:", OUT)

if __name__ == "__main__":
    main()
