#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import jieba
from collections import Counter
import math

# -------------- BM25 实现 --------------
class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.docs = docs
        self.N = len(docs)
        self.doc_len = [len(d) for d in docs]
        self.avg = sum(self.doc_len)/self.N
        # 统计 tf 和 df
        self.tf = []
        df = Counter()
        for d in docs:
            freqs = Counter(d)
            self.tf.append(freqs)
            for w in freqs:
                df[w] += 1
        # idf
        self.idf = {w: math.log(1 + (self.N - df_w + 0.5)/(df_w+0.5))
                    for w, df_w in df.items()}
        self.k1, self.b = k1, b

    def score(self, q, idx):
        s = 0.0
        freqs = self.tf[idx]
        dl = self.doc_len[idx]
        for w in q:
            if w not in freqs: continue
            f = freqs[w]
            num = self.idf[w] * f*(self.k1+1)
            den = f + self.k1*(1 - self.b + self.b*dl/self.avg)
            s += num/den
        return s

    def query(self, q_tokens, topk=1000):
        scores = [(i, self.score(q_tokens, i)) for i in range(self.N)]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:topk]

# -------------- 主流程 --------------
def main():
    # 1. 读 corpus，做 jieba 分词
    docs, doc_ids = [], []
    with open('STARD/data/corpus.jsonl', encoding='utf-8') as f:
        for line in f:
            obj = json.loads(line)
            doc_ids.append(obj['name'])
            tokens = jieba.lcut(obj['content'])
            docs.append(tokens)

    # 2. 构建 BM25
    bm25 = BM25(docs)

    # 3. 读 dev queries
    dev = set()
    with open('STARD/data/example/dev.query.txt', encoding='utf-8') as f:
        for line in f:
            qid, _ = line.strip().split('\t',1)
            dev.add(qid)

    queries = []
    with open('STARD/data/queries.json', encoding='utf-8') as f:
        for obj in json.load(f):
            qid = str(obj.get('query_id'))
            if qid in dev:
                queries.append((qid, obj['问题']))

    # 4. 对每个 query 做检索，写 run 文件
    with open('bm25p.run.jewelstar', 'w', encoding='utf-8') as out:
        for qid, text in queries:
            q_tokens = jieba.lcut(text)
            hits = bm25.query(q_tokens, topk=len(docs))
            for rank, (idx, sc) in enumerate(hits, start=1):
                out.write(f"{qid} Q0 {doc_ids[idx]} {rank} {sc:.6f} BM25\n")

    print("Generated bm25p.run.jewelstar")

if __name__ == '__main__':
    main()
