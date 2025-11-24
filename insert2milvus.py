#!/usr/bin/env python3
"""
从文件（CSV/JSON）插入数据到 Milvus 的脚本

支持功能：
1. 从 CSV 或 JSON 文件读取数据
2. 自动创建 collection（如果不存在）
3. 批量插入数据
4. 支持自动向量化（如果需要）
"""

import argparse
import json
import csv
import sys
import gc
from typing import List, Dict, Any, Optional
from pathlib import Path

from chunk2milvus_hq import MilvusClient, EmbeddingService
from pymilvus import FieldSchema, DataType, CollectionSchema


def parse_vector(vector_str: str) -> List[float]:
    """解析向量字符串为浮点数列表"""
    if not vector_str or vector_str.strip() == '':
        return []
    try:
        # 尝试解析为 JSON 数组
        if vector_str.strip().startswith('['):
            return json.loads(vector_str)
        # 尝试按逗号分割
        return [float(x.strip()) for x in vector_str.split(',') if x.strip()]
    except Exception as e:
        print(f"警告: 无法解析向量: {vector_str[:50]}... 错误: {e}")
        return []


def parse_metadata(metadata_str: str) -> Dict[str, Any]:
    """解析元数据字符串为字典"""
    if not metadata_str or metadata_str.strip() == '':
        return {}
    try:
        if isinstance(metadata_str, dict):
            return metadata_str
        if isinstance(metadata_str, str):
            return json.loads(metadata_str)
        return {}
    except Exception as e:
        print(f"警告: 无法解析元数据: {metadata_str[:50]}... 错误: {e}")
        return {}


def read_csv_file(file_path: str, batch_size: int = 1000) -> List[Dict[str, Any]]:
    """读取 CSV 文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # 处理向量字段
            if 'dense_vector' in row and row['dense_vector']:
                row['dense_vector'] = parse_vector(row['dense_vector'])
            
            # 处理元数据字段
            if 'metadata' in row and row['metadata']:
                row['metadata'] = parse_metadata(row['metadata'])
            
            data.append(row)
            
            # 如果达到批次大小，返回数据（用于流式处理）
            if len(data) >= batch_size:
                yield data
                data = []
    
    # 返回剩余数据
    if data:
        yield data


def read_json_file(file_path: str, batch_size: int = 1000) -> List[Dict[str, Any]]:
    """读取 JSON 文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 如果是列表，分批返回
    if isinstance(data, list):
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            # 处理向量和元数据字段
            for item in batch:
                if 'dense_vector' in item and isinstance(item['dense_vector'], str):
                    item['dense_vector'] = parse_vector(item['dense_vector'])
                if 'metadata' in item and isinstance(item['metadata'], str):
                    item['metadata'] = parse_metadata(item['metadata'])
            yield batch
    else:
        # 单个对象
        if 'dense_vector' in data and isinstance(data['dense_vector'], str):
            data['dense_vector'] = parse_vector(data['dense_vector'])
        if 'metadata' in data and isinstance(data['metadata'], str):
            data['metadata'] = parse_metadata(data['metadata'])
        yield [data]


def infer_schema_from_data(
    sample_data: List[Dict[str, Any]],
    vector_dim: Optional[int] = None,
    pk_field: str = "pk"
) -> List[FieldSchema]:
    """
    从数据推断 schema
    
    Args:
        sample_data: 样本数据
        vector_dim: 向量维度（如果为 None 则从数据推断）
        pk_field: 主键字段名
    """
    if not sample_data:
        raise ValueError("无法从空数据推断 schema")
    
    fields = []
    first_row = sample_data[0]
    
    # 计算所有数据中每个字段的最大长度（用于 VARCHAR 字段）
    def get_max_length(field_name: str, all_data: List[Dict[str, Any]]) -> int:
        """获取字段在所有数据中的最大长度"""
        max_len = 0
        for row in all_data:
            value = row.get(field_name)
            if value is not None:
                length = len(str(value))
                max_len = max(max_len, length)
        return max_len
    
    for field_name, field_value in first_row.items():
        if field_name == pk_field:
            # 主键字段
            if isinstance(field_value, str):
                max_length = get_max_length(field_name, sample_data)
                # 添加 20% 的缓冲，但不超过 65535
                max_length = min(int(max_length * 1.2) + 100, 65535)
                fields.append(FieldSchema(
                    name=field_name,
                    dtype=DataType.VARCHAR,
                    is_primary=True,
                    auto_id=False,
                    max_length=max_length
                ))
            else:
                fields.append(FieldSchema(
                    name=field_name,
                    dtype=DataType.INT64,
                    is_primary=True,
                    auto_id=False
                ))
        elif field_name == "dense_vector":
            # 向量字段
            if isinstance(field_value, list) and len(field_value) > 0:
                dim = len(field_value) if vector_dim is None else vector_dim
                fields.append(FieldSchema(
                    name=field_name,
                    dtype=DataType.FLOAT_VECTOR,
                    dim=dim
                ))
            else:
                raise ValueError("无法推断向量维度，请指定 --vector-dim")
        elif field_name == "metadata":
            # 元数据字段（JSON）
            fields.append(FieldSchema(
                name=field_name,
                dtype=DataType.JSON
            ))
        elif field_name == "text":
            # 文本字段 - 直接使用最大值 65535
            fields.append(FieldSchema(
                name=field_name,
                dtype=DataType.VARCHAR,
                max_length=65535,
                enable_analyzer=True
            ))
        else:
            # 其他字段，根据类型推断
            if isinstance(field_value, str):
                max_length = get_max_length(field_name, sample_data)
                # 添加 20% 的缓冲，但不超过 65535
                max_length = min(int(max_length * 1.2) + 100, 65535)
                fields.append(FieldSchema(
                    name=field_name,
                    dtype=DataType.VARCHAR,
                    max_length=max_length
                ))
            elif isinstance(field_value, int):
                fields.append(FieldSchema(
                    name=field_name,
                    dtype=DataType.INT64
                ))
            elif isinstance(field_value, float):
                fields.append(FieldSchema(
                    name=field_name,
                    dtype=DataType.DOUBLE
                ))
            elif isinstance(field_value, bool):
                fields.append(FieldSchema(
                    name=field_name,
                    dtype=DataType.BOOL
                ))
            else:
                # 默认使用 VARCHAR
                fields.append(FieldSchema(
                    name=field_name,
                    dtype=DataType.VARCHAR,
                    max_length=256
                ))
    
    return fields


def insert_data_batch(
    client: MilvusClient,
    collection_name: str,
    batch_data: List[Dict[str, Any]],
    auto_embed: bool = False
) -> int:
    """
    插入一批数据到 collection
    
    Returns:
        插入的记录数
    """
    collection = client.get_collection(collection_name)
    
    # 获取 schema 字段信息
    schema_fields = {field.name: field for field in collection.schema.fields}
    
    # 找到主键字段名
    pk_field_name = None
    for field_name, field in schema_fields.items():
        if field.is_primary:
            pk_field_name = field_name
            break
    
    if pk_field_name is None:
        raise ValueError("No primary key field found in schema")
    
    # 准备插入数据（按字段组织）
    insert_data = {}
    num_rows = len(batch_data)
    
    # 提取各字段数据
    for field_name in schema_fields.keys():
        field_values = []
        for row in batch_data:
            value = row.get(field_name)
            if value is None:
                # 根据字段类型设置默认值
                field = schema_fields[field_name]
                if field.dtype == DataType.JSON:
                    value = {}
                elif field.dtype == DataType.VARCHAR:
                    value = ""
                elif field.dtype == DataType.INT64:
                    value = 0
                elif field.dtype == DataType.DOUBLE:
                    value = 0.0
                elif field.dtype == DataType.BOOL:
                    value = False
                elif field.dtype == DataType.FLOAT_VECTOR:
                    value = []
            field_values.append(value)
        insert_data[field_name] = field_values
    
    # 如果需要自动向量化且没有向量数据
    if "dense_vector" in schema_fields and auto_embed:
        if not any(insert_data.get("dense_vector", [])):
            # 检查是否有 text 字段
            if "text" in schema_fields and insert_data.get("text"):
                if client.embedding_service is None:
                    raise ValueError(
                        "embedding_service is required for auto embedding. "
                        "Please provide embedding service configuration."
                    )
                print(f"  正在向量化 {num_rows} 个文本块...")
                vectors = client.embedding_service.embeddings(
                    insert_data["text"],
                    batch_size=10,
                    show_progress=True
                )
                insert_data["dense_vector"] = vectors
    
    # 转换为按行组织的数据
    data_rows = []
    for i in range(num_rows):
        row = {}
        for field_name, field_values in insert_data.items():
            row[field_name] = field_values[i]
        data_rows.append(row)
    
    # 插入数据
    insert_result = collection.insert(data_rows)
    collection.flush()
    
    return num_rows


def main():
    parser = argparse.ArgumentParser(
        description="从文件插入数据到 Milvus",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 从 CSV 文件插入数据
  python insert2milvus.py -f data.csv -c my_collection

  # 从 JSON 文件插入数据，自动创建 collection
  python insert2milvus.py -f data.json -c my_collection --create

  # 指定数据库和向量维度
  python insert2milvus.py -f data.csv -c my_collection -d test --vector-dim 3072

  # 批量插入，每批 500 条
  python insert2milvus.py -f data.csv -c my_collection --batch-size 500
        """
    )
    
    parser.add_argument(
        "-f", "--file",
        required=True,
        help="输入文件路径（CSV 或 JSON）"
    )
    
    parser.add_argument(
        "-c", "--collection",
        required=True,
        help="Collection 名称"
    )
    
    parser.add_argument(
        "-a", "--alias",
        default="default",
        help="Milvus 连接别名（默认: default）"
    )
    
    parser.add_argument(
        "-d", "--database",
        default=None,
        help="数据库名称（默认: 从环境变量 MILVUS_DATABASE 读取，或使用 'default'）"
    )
    
    parser.add_argument(
        "--create",
        action="store_true",
        help="如果 collection 不存在则创建（自动推断 schema）"
    )
    
    parser.add_argument(
        "--vector-dim",
        type=int,
        default=None,
        help="向量维度（如果数据中没有向量字段或需要自动向量化）"
    )
    
    parser.add_argument(
        "--pk-field",
        default="pk",
        help="主键字段名（默认: pk）"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="批量插入大小（默认: 1000）"
    )
    
    parser.add_argument(
        "--auto-embed",
        action="store_true",
        help="自动向量化（需要提供 embedding_service）"
    )
    
    parser.add_argument(
        "--uri",
        default=None,
        help="Milvus URI（默认: 从环境变量 MILVUS_URI 读取）"
    )
    
    parser.add_argument(
        "--token",
        default=None,
        help="Milvus Token（默认: 从环境变量 MILVUS_TOKEN 读取）"
    )
    
    args = parser.parse_args()
    
    # 检查文件是否存在
    if not Path(args.file).exists():
        print(f"错误: 文件 '{args.file}' 不存在", file=sys.stderr)
        sys.exit(1)
    
    # 确定文件类型
    file_ext = Path(args.file).suffix.lower()
    if file_ext not in ['.csv', '.json']:
        print(f"错误: 不支持的文件格式 '{file_ext}'，仅支持 .csv 和 .json", file=sys.stderr)
        sys.exit(1)
    
    try:
        # 创建 Milvus 客户端
        database_info = f", database: {args.database}" if args.database else ""
        print(f"正在连接到 Milvus (alias: {args.alias}{database_info})...")
        
        # 如果需要自动向量化，创建 embedding service
        embedding_service = None
        if args.auto_embed:
            embedding_service = EmbeddingService()
        
        client = MilvusClient(
            uri=args.uri,
            token=args.token,
            alias=args.alias,
            database=args.database,
            embedding_service=embedding_service
        )
        
        # 检查连接
        if not client.check_connection():
            print("错误: 无法连接到 Milvus", file=sys.stderr)
            sys.exit(1)
        
        conn_info = client.get_connection_info()
        print(f"连接成功: {conn_info.get('address', 'N/A')}")
        
        # 检查 collection 是否存在
        collection_exists = client.has_collection(args.collection)
        
        if not collection_exists:
            if not args.create:
                print(f"错误: Collection '{args.collection}' 不存在，请使用 --create 参数自动创建", file=sys.stderr)
                sys.exit(1)
            
            # 自动创建 collection
            print(f"\nCollection '{args.collection}' 不存在，正在创建...")
            
            # 读取样本数据以推断 schema（只需要少量样本即可）
            print("  正在读取样本数据以推断 schema...")
            sample_data = []
            if file_ext == '.csv':
                for batch in read_csv_file(args.file, batch_size=100):
                    sample_data.extend(batch)
                    if len(sample_data) >= 100:  # 只需要前100条来推断字段类型
                        break
            else:
                for batch in read_json_file(args.file, batch_size=100):
                    sample_data.extend(batch)
                    if len(sample_data) >= 100:
                        break
            
            if not sample_data:
                print("错误: 无法从文件读取数据", file=sys.stderr)
                sys.exit(1)
            
            # 推断 schema
            print("  正在推断 schema...")
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
        
        # 读取并插入数据
        print(f"\n正在从文件 '{args.file}' 读取数据并插入到 collection '{args.collection}'...")
        total_inserted = 0
        batch_num = 0
        
        # 读取文件（流式处理）
        if file_ext == '.csv':
            data_reader = read_csv_file(args.file, batch_size=args.batch_size)
        else:
            data_reader = read_json_file(args.file, batch_size=args.batch_size)
        
        for batch_data in data_reader:
            if not batch_data:
                continue
            
            batch_num += 1
            print(f"  正在插入第 {batch_num} 批数据 ({len(batch_data)} 条)...")
            
            try:
                inserted = insert_data_batch(
                    client=client,
                    collection_name=args.collection,
                    batch_data=batch_data,
                    auto_embed=args.auto_embed
                )
                total_inserted += inserted
                print(f"    已插入 {inserted} 条数据（累计: {total_inserted}）")
                
                # 每10批执行一次垃圾回收
                if batch_num % 10 == 0:
                    gc.collect()
            except Exception as e:
                print(f"    错误: 插入失败: {e}", file=sys.stderr)
                import traceback
                traceback.print_exc()
                # 继续处理下一批
                continue
        
        print(f"\n插入完成！共插入 {total_inserted} 条数据")
        
    except KeyboardInterrupt:
        print("\n\n操作已取消")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

