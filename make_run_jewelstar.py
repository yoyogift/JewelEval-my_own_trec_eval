#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import jieba
import math
from collections import Counter, defaultdict

# -------------- BM25 实现 --------------
class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = docs
        self.N = len(docs)
        self.doc_len = [len(d) for d in docs]
        self.avg = sum(self.doc_len) / self.N
        self.tf = []
        df = Counter()
        for d in docs:
            freqs = Counter(d)
            self.tf.append(freqs)
            for w in freqs:
                df[w] += 1
        self.idf = {
            w: math.log(1 + (self.N - df_w + 0.5) / (df_w + 0.5))
            for w, df_w in df.items()
        }
        self.k1, self.b = k1, b

    def score(self, q_tokens, idx):
        freqs = self.tf[idx]
        dl = self.doc_len[idx]
        s = 0.0
        for w in q_tokens:
            if w not in freqs:
                continue
            f = freqs[w]
            num = self.idf[w] * f * (self.k1 + 1)
            den = f + self.k1 * (1 - self.b + self.b * dl / self.avg)
            s += num / den
        return s

    def query(self, q_tokens, topk=1000):
        scores = [(i, self.score(q_tokens, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:topk]

# ———— 评测函数 ————
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
    micro‑recall@K = (∑_q |retrieved∩relevant|) / (∑_q |relevant|)
    """
    total_rel = sum(len(rset) for rset in qrels.values())
    total_ret = 0
    for qid, rel_docs in qrels.items():
        retrieved = [d for d,_ in runs.get(qid, [])[:K]]
        total_ret += sum(1 for d in retrieved if d in rel_docs)
    return total_ret / total_rel if total_rel else 0.0

# ———— 主流程 ————
def main():
    CORPUS       = 'STARD/data/corpus.jsonl'
    DEV_TXT      = 'STARD/data/example/dev.query.txt'
    QUERIES_JSON = 'STARD/data/queries.json'
    QRELS        = 'relevance.jewelstar'
    OUTPUT       = 'bm25p.run.jewelstar'
    TOPK         = 1000
    K_VALUES     = [5,10,15,20,30,100,200,500,1000]

    # 1) 读入语料并分词
    docs, doc_ids = [], []
    with open(CORPUS, encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            doc_ids.append(obj['name'])
            docs.append(jieba.lcut(obj['content']))

    # 2) 构建 BM25
    bm25 = BM25(docs)

    # 3) 读 dev 查询列表
    dev_ids = set()
    with open(DEV_TXT, encoding='utf-8') as f:
        for L in f:
            qid, _ = L.strip().split('\t', 1)
            dev_ids.add(str(qid))

    queries = []
    with open(QUERIES_JSON, encoding='utf-8') as f:
        for obj in json.load(f):
            qid = str(obj.get('query_id'))
            if qid in dev_ids:
                queries.append((qid, obj['问题']))

    # 4) 检索、写 run & 收集 runs
    runs = {}
    with open(OUTPUT, 'w', encoding='utf-8') as out:
        for qid, text in queries:
            q_tokens = jieba.lcut(text)
            hits = bm25.query(q_tokens, topk=TOPK)
            runs[qid] = [(doc_ids[idx], score) for idx, score in hits]
            for rank, (idx, score) in enumerate(hits, start=1):
                out.write(f"{qid} Q0 {doc_ids[idx]} {rank} {score:.6f} BM25\n")

    print(f"✅ Generated {OUTPUT}")

    # 5) 当场评测 micro‑recall@K
    qrels = load_qrels(QRELS)
    print("\n=== On-the-fly evaluation (micro‑recall@K) ===")
    for K in K_VALUES:
        r = micro_recall_at_K(qrels, runs, K)
        print(f"recall@{K:<4d} all {r:.4f}")

if __name__ == '__main__':
    main()
