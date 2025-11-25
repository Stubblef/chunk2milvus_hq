import argparse
import json
import csv
import sys
import gc
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path
from tqdm import tqdm

from chunk2milvus_hq import MilvusClient


client = MilvusClient(database="test")

collection = client.get_collection("qwen2048")

num_entities = collection.num_entities
print(f"Collection 总共有 {num_entities} 条数据")


pk_field = "pk"

all_pks_set = set() 
all_pks = []


# collection.query(expr="",output_fields=[pk_field],limit=50)
pk_field = "pk"
batch_size = 1000  # 每批取 1000，可以按需求调整

all_pks = []
last_pk = None

# 输出文件（JSONL 格式）
output_file = "all_pks.jsonl"

# 创建进度条
pbar = tqdm(total=num_entities, desc="导出数据", unit="条", unit_scale=True)

with open(output_file, "w", encoding="utf-8") as f:
    while True:
        if last_pk is None:
            expr = ""   # 第一批，没有条件
        else:
            expr = f'{pk_field} > "{last_pk}"'

        results = collection.query(
            expr=expr,
            output_fields=[pk_field, "doc_id", "text", "dense_vector", "metadata"],
            limit=batch_size,
            order_by=pk_field  # 按 PK 排序非常重要！
        )

        if not results:
            break

        # 取出本批所有 pk
        batch_pks = [r[pk_field] for r in results]
        all_pks.extend(batch_pks)
        
        
        # ✨写入文件，每个 PK 一行
        for r in results:
            pk = r[pk_field]
            doc_id = r["doc_id"]
            text = r["text"]
            dense_vector = r["dense_vector"]
            metadata = r["metadata"]
            f.write(json.dumps({"pk": pk, "doc_id": doc_id, "text": text, "dense_vector": dense_vector, "metadata": metadata}, ensure_ascii=False) + "\n")

        # 更新游标
        last_pk = batch_pks[-1]
        
        # 更新进度条
        pbar.update(len(batch_pks))
        pbar.set_postfix({"当前PK": last_pk})

# 关闭进度条
pbar.close()

print(f"共获取到 PK 数量: {len(all_pks)}")
