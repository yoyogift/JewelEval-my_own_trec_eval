#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import jieba
import math
from collections import Counter, defaultdict

# 停用词集
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
    dev = set()
    with open(dev_txt, encoding='utf-8') as f:
        for L in f:
            qid,_ = L.strip().split('\t',1)
            dev.add(str(qid))
    qs = []
    with open(queries_json, encoding='utf-8') as f:
        for obj in json.load(f):
            qid = str(obj.get('query_id'))
            if qid in dev:
                qs.append((qid, obj['问题']))
    return qs

class QueryLikelihood:
    def __init__(self, docs_tokens, mu=2000):
        self.mu = mu
        self.model = [(Counter(d), len(d)) for d in docs_tokens]
        self.bg_cf = Counter()
        self.bg_len = 0
        for tf, ln in self.model:
            self.bg_cf.update(tf)
            self.bg_len += ln
        self.V = len(self.bg_cf)

    def score(self, q_tokens, tf, dl):
        score = 0.0
        for w in q_tokens:
            cf = self.bg_cf.get(w, 0)
            p_bg = (cf + 1) / (self.bg_len + self.V)
            p = (tf.get(w, 0) + self.mu * p_bg) / (dl + self.mu)
            score += math.log(p)
        return score

    def query(self, q_tokens, topk=None):
        scores = [(idx, self.score(q_tokens, tf, dl))
                  for idx,(tf, dl) in enumerate(self.model)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores if topk is None else scores[:topk]

# ———— 以下两个是评测所需 ————
def load_qrels(path):
    """
    读取 relevance 文件 (<qid> 0 <docid> <rel>)，只保留 rel>0
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
    计算 micro‑recall@K：∑_q |retrieved∩relevant|  / ∑_q |relevant|
    """
    total_rel = sum(len(rset) for rset in qrels.values())
    total_ret = 0
    for qid, rel_docs in qrels.items():
        retrieved = [d for d,_ in runs.get(qid, [])[:K]]
        total_ret += sum(1 for d in retrieved if d in rel_docs)
    return total_ret / total_rel if total_rel else 0.0
# ———————————————————————

def main():
    CORPUS       = 'STARD/data/corpus.jsonl'
    DEV_TXT      = 'STARD/data/example/dev.query.txt'
    QUERIES_JSON = 'STARD/data/queries.json'
    OUTPUT       = 'ql.run.jewelstar'
    TOPK         = 1000
    MU           = 2000

    # 1) 载入语料
    doc_ids, docs_tokens = load_corpus(CORPUS)

    # 2) 构建 QL 模型
    ql = QueryLikelihood(docs_tokens, mu=MU)

    # 3) 读取查询
    queries = load_queries(DEV_TXT, QUERIES_JSON)

    # 4) 检索、写 run，同时保留 runs
    runs = {}
    with open(OUTPUT, 'w', encoding='utf-8') as out:
        for qid, text in queries:
            q_tokens = tokenize(text)
            hits = ql.query(q_tokens, topk=TOPK)
            runs[qid] = [(doc_ids[idx], score) for idx, score in hits]
            for rank, (idx, score) in enumerate(hits, start=1):
                out.write(f"{qid} Q0 {doc_ids[idx]} {rank} {score:.6f} QL\n")

    print(f"✅ Generated {OUTPUT}")

    # 5) 当场评测 (micro recall@K)
    qrels = load_qrels('relevance.jewelstar')
    print("\n=== On-the-fly evaluation ===")
    for K in [5,10,15,20,30,100,200,500,1000]:
        r = micro_recall_at_K(qrels, runs, K)
        print(f"recall@{K:<4d} all {r:.4f}")

if __name__ == '__main__':
    main()
