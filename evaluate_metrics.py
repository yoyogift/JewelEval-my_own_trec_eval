#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import math
from collections import defaultdict

def load_relevance(path):
    qrels = defaultdict(dict)
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            qid, _, docid, rel = parts
            qrels[qid][docid] = int(rel)
    return dict(qrels)

def load_run(path):
    runs = defaultdict(list)
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 6 or parts[1] != 'Q0':
                continue
            qid = parts[0]
            rank  = int(parts[-3])
            score = float(parts[-2])
            docid = ' '.join(parts[2:-3])
            runs[qid].append((docid, score, rank))
    out = {}
    for q, lst in runs.items():
        lst.sort(key=lambda x: x[2])
        out[q] = [(d,s) for d,s,_ in lst]
    return out

def compute_all(qrels, runs, Ks):
    # 初始化
    recall_at = {K: [] for K in Ks}
    p_at_10 = []
    rr_list = []
    ap_list = []
    ndcg_at_10 = []
    num_rel, num_rel_ret, num_ret = 0, 0, 0

    for qid, retrieved in runs.items():
        rel_docs = {d for d,r in qrels.get(qid,{}).items() if r>0}
        total_rel = len(rel_docs)
        if total_rel == 0:
            continue

        docs = [d for d,_ in retrieved]
        num_ret += len(docs)
        num_rel += total_rel
        hit_rel = 0

        # Recall@K, Precision@10, nDCG@10, RR, AP
        precision_sums = 0.0
        num_hits = 0
        dcg, idcg = 0.0, 0.0

        # 预计算 ideal DCG for @10
        ideal_rels = sorted(qrels[qid].values(), reverse=True)[:10]
        for i, r in enumerate(ideal_rels, start=1):
            idcg += (2**r - 1)/math.log2(i+1)

        for idx, d in enumerate(docs, start=1):
            is_rel = 1 if d in rel_docs else 0

            # num_rel_ret
            if is_rel:
                num_rel_ret += 1

            # Recall@K
            for K in Ks:
                if idx <= K and is_rel:
                    hit_rel += 1
                    recall_at[K].append(hit_rel/total_rel)

            # Precision@10
            if idx <= 10:
                if is_rel:
                    num_hits += 1
                p = num_hits/idx
            if idx == 10:
                p_at_10.append(p)

            # RR
            if is_rel and rr_list is not None:
                if idx == 1 or (len(rr_list) < len(ap_list)+1):
                    rr_list.append(1/idx)

            # AP
            if is_rel:
                precision_sums += num_hits/idx

            # DCG@10
            if idx <= 10:
                rel = qrels[qid].get(d,0)
                dcg += (2**rel - 1)/math.log2(idx+1)

        # 完成一个 query
        ap_list.append(precision_sums/total_rel)
        ndcg_at_10.append(dcg/(idcg if idcg>0 else 1))

    # 聚合
    num_q = len(runs)
    results = {
        'num_q': num_q,
        'num_ret': num_ret,
        'num_rel': num_rel,
        'num_rel_ret': num_rel_ret,
        'map': sum(ap_list)/len(ap_list) if ap_list else 0.0,
        'P_10': sum(p_at_10)/len(p_at_10) if p_at_10 else 0.0,
        'MRR': sum(rr_list)/len(rr_list) if rr_list else 0.0,
        'nDCG@10': sum(ndcg_at_10)/len(ndcg_at_10) if ndcg_at_10 else 0.0
    }
    # 各 recall@K
    for K in Ks:
        results[f'recall@{K}'] = (sum(recall_at[K])/len(recall_at[K])) if recall_at[K] else 0.0

    return results

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--relevance", required=True, help="relatedness file")
    p.add_argument("--run",       required=True, help="run file")
    args = p.parse_args()

    Ks = [5,10,15,20,30,100,200,500,1000]
    qrels = load_relevance(args.relevance)
    runs  = load_run(args.run)
    res   = compute_all(qrels, runs, Ks)

    # 打印
    print(f"runid all run1")
    print(f"num_q    all {res['num_q']}")
    print(f"num_ret  all {res['num_ret']}")
    print(f"num_rel  all {res['num_rel']}")
    print(f"num_rel_ret all {res['num_rel_ret']}")
    print(f"map      all {res['map']:.4f}")
    print(f"P_10     all {res['P_10']:.4f}")
    for K in Ks:
        print(f"recall_{K:<3d} all {res[f'recall@{K}']:.4f}")
    print(f"MRR      all {res['MRR']:.4f}")
    print(f"ndcg_cut_10 all {res['nDCG@10']:.4f}")
