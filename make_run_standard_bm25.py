#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import jieba
from collections import defaultdict
from rank_bm25 import BM25Okapi

# 停用词（如有需要可扩充）
STOPWORDS = set(['\n',' ','\t','，','。','（','）','：','“','”'])

def tokenize(text):
    return [w for w in jieba.lcut(text) if w.strip() and w not in STOPWORDS]

def load_corpus(path):
    doc_ids, docs = [], []
    with open(path, encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            doc_ids.append(obj['name'])
            docs.append(tokenize(obj['content']))
    return doc_ids, docs

def load_queries(dev_txt, queries_json):
    dev_ids = set()
    with open(dev_txt, encoding='utf-8') as f:
        for L in f:
            qid,_ = L.strip().split('\t',1)
            dev_ids.add(str(qid))
    qs = []
    with open(queries_json, encoding='utf-8') as f:
        for obj in json.load(f):
            qid = str(obj.get('query_id'))
            if qid in dev_ids:
                qs.append((qid, obj['问题']))
    return qs

def load_qrels(path):
    """
    读取 qrels 文件 (<qid> 0 <docid> <rel>)，只保留 rel>0
    返回 dict: qid -> set(docid)
    """
    qrels = defaultdict(set)
    with open(path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4:
                continue
            qid, _, docid, rel = parts
            if int(rel) > 0:
                qrels[qid].add(docid)
    return qrels

def micro_recall_at_K(qrels, runs, K):
    """
    micro‑recall@K = (∑_q |retrieved∩relevant|) / (∑_q |relevant|)
    """
    total_rel = sum(len(rset) for rset in qrels.values())
    total_ret = 0
    for qid, rel_docs in qrels.items():
        retrieved = [d for d,_ in runs.get(qid, [])[:K]]
        total_ret += sum(1 for d in retrieved if d in rel_docs)
    return total_ret / total_rel if total_rel else 0.0

def main():
    # 配置
    CORPUS       = 'STARD/data/corpus.jsonl'
    DEV_TXT      = 'STARD/data/example/dev.query.txt'
    QUERIES_JSON = 'STARD/data/queries.json'
    QRELS        = 'relevance.jewelstar'      # 你的 qrels 文件
    OUTPUT       = 'bm25.run.jewelstar'
    TOPK         = 1000

    # 1) 载入语料
    doc_ids, docs_tokens = load_corpus(CORPUS)

    # 2) 初始化 BM25
    bm25 = BM25Okapi(docs_tokens)

    # 3) 载入查询
    queries = load_queries(DEV_TXT, QUERIES_JSON)

    # 4) 检索、写 run & 保留 runs
    runs = {}
    with open(OUTPUT, 'w', encoding='utf-8') as out:
        for qid, text in queries:
            q_tokens = tokenize(text)
            scores   = bm25.get_scores(q_tokens)
            ranked   = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:TOPK]

            # 收集到 runs
            runs[qid] = [(doc_ids[idx], score) for idx, score in ranked]

            # 写文件
            for rank, (doc_idx, score) in enumerate(ranked, start=1):
                out.write(f"{qid} Q0 {doc_ids[doc_idx]} {rank} {score:.6f} BM25\n")

    print(f"✅ Generated {OUTPUT}")

    # 5) 当场评测 micro‑recall@K
    qrels = load_qrels(QRELS)
    print("\n=== On-the-fly evaluation (micro‑recall@K) ===")
    for K in [5,10,15,20,30,100,200,500,1000]:
        r = micro_recall_at_K(qrels, runs, K)
        print(f"recall@{K:<4d} all {r:.4f}")

if __name__ == '__main__':
    main()
