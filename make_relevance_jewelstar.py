import json

def main():
    # 在 dev.query.txt 列出的 query_id
    dev_set = set()
    with open('STARD/data/example/dev.query.txt', encoding='utf-8') as f:
        for line in f:
            qid, _ = line.strip().split('\t',1)
            dev_set.add(qid)

    # 从 queries.json 提取 match_name
    out_lines = set()
    with open('STARD/data/queries.json', encoding='utf-8') as f:
        for obj in json.load(f):
            qid = str(obj.get('query_id'))
            if qid in dev_set:
                for docid in obj.get('match_name', []):
                    out_lines.add(f"{qid} 0 {docid} 1")

    # 写文件
    with open('relevance.jewelstar', 'w', encoding='utf-8') as out:
        for line in sorted(out_lines):
            out.write(line + '\n')

    print(f"Generated relevance.jewelstar with {len(out_lines)} entries.")

if __name__ == '__main__':
    main()
