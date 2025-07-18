# new_trec_eval_jewelstar_version
Puisque trec_eval m’a planté comme un GPS en zone sans réseau, j’ai décidé de lui faire un remake maison : ma propre version “artisanale” de trec_eval pour piloter mes expériences. Résultat : aussi chiadé qu’une recette de grand‑mère, mais ça carbure !🦐

## 📦 目录结构
```
.
├── STARD
│ └── data
│ ├── example
│ │ └── dev.query.txt # 开发集需要评测的 query_id 列表
│ ├── corpus.jsonl # 文档库
│ └── queries.json # 查询和 match_name 标注
├── make_relevance_jewelstar.py # 生成 relevance.jewelstar 的脚本
├── make_run_jewelstar.py # 生成 bm25p.run.jewelstar 的脚本
├── evaluate_metrics.py # 评测脚本，输出 recall/MRR/nDCG/P@10/MAP 等
└── README.md # 本说明文件
```


## ⚙️ 环境与依赖

- Python ≥3.7
- pip install jieba


## 1. 生成相关性标注文件 relevance.jewelstar

```
python make_relevance_jewelstar.py
```

该脚本会读取 STARD/data/example/dev.query.txt 与 STARD/data/queries.json，输出去重、排序后的：

/<qid/> 0 /<docid/> 1'

格式文件 relevance.jewelstar。

## 2.生成检索结果文件 bm25p.run.jewelstar

```
python make_run_jewelstar.py
```

该脚本会：

- 载入 STARD/data/corpus.jsonl，对每篇文档用 jieba.lcut 分词
- 载入 STARD/data/example/dev.query.txt 与 STARD/data/queries.json 中的 query
- 用纯 Python BM25 对每个 query 排序整个语料，输出：

```
<qid> Q0 <docid> <rank> <score> BM25
```

到 bm25p.run.jewelstar。
## 3. 评测并输出指标

```
python evaluate_metrics.py \
  --relevance relevance.jewelstar \
  --run      bm25p.run.jewelstar
```

默认会计算并打印：

    runid、num_q、num_ret、num_rel、num_rel_ret
    
    map、P_10
    
    recall_5、recall_10、recall_15、recall_20、recall_30、recall_100、recall_200、recall_500、recall_1000
    
    MRR (reciprocal rank)
    
    ndcg_cut_10

示例输出：

```
runid all run1
num_q    all 308
num_ret  all 17047184
num_rel  all 512
num_rel_ret all 512
map      all 0.2602
P_10     all 0.0594
recall_5   all 1.0971
recall_10  all 1.8420
recall_15  all 2.5290
recall_20  all 3.1448
recall_30  all 3.6781
recall_100 all 3.8389
recall_200 all 4.2316
recall_500 all 4.5487
recall_1000 all 4.9578
MRR      all 0.3280
ndcg_cut_10 all 0.3084
```

## 4. 验证与对比

若你仍可使用官方 trec_eval，可将 relevance.jewelstar 重命名为 qrels.trec，bm25p.run.jewelstar命名为bm25p.run并用：

trec_eval -m all_qrels qrels.trec bm25p.run

核对关键指标是否完全一致。
