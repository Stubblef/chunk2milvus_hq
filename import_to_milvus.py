#!/usr/bin/env python3
"""将 JSONL 文件导入到 Milvus collection"""

import argparse
import json
from chunk2milvus_hq import MilvusClient
from typing import List, Dict, Any, Optional, Set, Tuple
from pymilvus import FieldSchema, DataType, CollectionSchema


"""
fields = infer_schema_from_data(
                sample_data,
                vector_dim=args.vector_dim,
                pk_field=args.pk_field
            )
            
            # 创建 collection
            collection = client.create_collection(
                collection_name=args.collection,
                fields=fields,
                description=f"从文件 {args.file} 导入的数据",
                enable_dynamic_field=False
            )
            print(f"Collection '{args.collection}' 创建成功")
            
            # 如果有向量字段，创建索引
            if any(f.name == "dense_vector" for f in fields):
                print("  正在创建向量索引...")
                try:
                    index_params = {
                        "index_type": "HNSW",
                        "metric_type": "COSINE",
                        "params": {"M": 32, "efConstruction": 300}
                    }
                    client.create_index(args.collection, "dense_vector", index_params)
                    print("  向量索引创建成功")
                except Exception as e:
                    print(f"  警告: 索引创建失败（可能已存在）: {e}")
"""



def main():
    parser = argparse.ArgumentParser(description="将 JSONL 文件导入到 Milvus collection")
    parser.add_argument("--input", "-i", default="all_pks.jsonl", help="输入的 JSONL 文件路径")
    parser.add_argument("--collection", "-c", required=True, help="目标 collection 名称")
    parser.add_argument("--database", "-d", default="default", help="目标数据库名称（可选）")
    parser.add_argument("--batch-size", "-b", type=int, default=1000, help="批量插入大小，默认 1000")
    
    args = parser.parse_args()
    
    # 初始化客户端
    client = MilvusClient(uri="http://8.130.130.118:19530",token="root:wjaKnxKUxtA0Qgmc",database=args.database)
    # client.list_collections()
    
    fields = [
            FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=1024),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535, enable_analyzer=True),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=2048),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]    
    
    if not client.has_collection(args.collection):
        print(f"Collection '{args.collection}' 不存在，正在创建...")
        collection = client.create_collection(
                collection_name=args.collection,
                fields=fields,
                enable_dynamic_field=False
            )
        print(f"Collection '{args.collection}' 创建成功")
        
        # 如果有向量字段，创建索引
        if any(f.name == "dense_vector" for f in fields):
            print("  正在创建向量索引...")
            try:
                index_params = {
                    "index_type": "HNSW",
                    "metric_type": "COSINE",
                    "params": {"M": 32, "efConstruction": 300}
                }
                client.create_index(args.collection, "dense_vector", index_params)
                print("  向量索引创建成功")
            except Exception as e:
                print(f"  警告: 索引创建失败（可能已存在）: {e}")
    else:
        collection = client.get_collection(args.collection)
        print(f"Collection '{args.collection}' 已存在")
    
    # 读取并批量插入
    batch = []
    total = 0
    from tqdm import tqdm
    with open(args.input, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            data = json.loads(line.strip())
            batch.append(data)
            
            if len(batch) >= args.batch_size:
                collection.insert(batch)
                total += len(batch)
                print(f"已插入 {total} 条数据")
                batch = []
        
        # 插入剩余数据
        if batch:
            collection.insert(batch)
            total += len(batch)
            print(f"已插入 {total} 条数据")
    
    # 刷新确保数据持久化
    collection.flush()
    print(f"导入完成，共插入 {total} 条数据")


if __name__ == "__main__":
    main()

