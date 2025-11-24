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
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Set, Tuple
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # 简单的进度条替代
    class tqdm:
        def __init__(self, iterable=None, total=None, desc=None, unit=None):
            self.iterable = iterable
            self.total = total
            self.desc = desc or ""
            self.unit = unit or "it"
            self.n = 0
        
        def __iter__(self):
            return iter(self.iterable) if self.iterable else self
        
        def __next__(self):
            if self.iterable:
                return next(self.iterable)
            raise StopIteration
        
        def update(self, n=1):
            self.n += n
            if self.total:
                percent = (self.n / self.total) * 100
                print(f"\r{self.desc}: {self.n}/{self.total} ({percent:.1f}%)", end='', flush=True)
        
        def close(self):
            print()  # 换行
        
        def __enter__(self):
            return self
        
        def __exit__(self, *args):
            self.close()

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


def get_existing_pks(
    client: MilvusClient,
    collection_name: str,
    pk_field_name: str,
    pk_values: List[Any],
    batch_size: int = 500
) -> Set[Any]:
    """
    查询已存在的主键值（分批查询，避免表达式过长和内存溢出）
    
    Args:
        batch_size: 每批查询的主键数量（默认500，避免表达式过长）
    
    Returns:
        已存在的主键集合
    """
    if not pk_values:
        return set()
    
    collection = client.get_collection(collection_name)
    collection.load()
    
    # 如果 collection 为空，直接返回空集合
    num_entities = collection.num_entities
    if num_entities == 0:
        return set()
    
    # 获取主键字段类型
    pk_field = next((f for f in collection.schema.fields if f.name == pk_field_name), None)
    if not pk_field:
        return set()
    
    existing_pks = set()
    
    # 分批查询，避免表达式过长和内存占用过大
    for i in range(0, len(pk_values), batch_size):
        batch_pks = pk_values[i:i + batch_size]
        
        try:
            # 构建查询表达式
            if pk_field.dtype == DataType.VARCHAR:
                pk_list_str = ", ".join([f'"{pk}"' for pk in batch_pks])
                expr = f"{pk_field_name} in [{pk_list_str}]"
            elif pk_field.dtype == DataType.INT64:
                pk_list_str = ", ".join([str(pk) for pk in batch_pks])
                expr = f"{pk_field_name} in [{pk_list_str}]"
            else:
                pk_list_str = ", ".join([f'"{str(pk)}"' for pk in batch_pks])
                expr = f"{pk_field_name} in [{pk_list_str}]"
            
            # 查询已存在的主键
            # 注意：如果 collection 为空，query 应该返回空列表
            # 但如果 expr 为空字符串，可能会返回所有数据，所以必须确保 expr 不为空
            if not expr or expr.strip() == "":
                continue  # 跳过无效表达式
            
            results = collection.query(
                expr=expr,
                output_fields=[pk_field_name],
                limit=len(batch_pks)
            )
            
            # 提取主键值并添加到集合
            # 确保结果不为空且包含主键字段
            if results:
                batch_existing = {row[pk_field_name] for row in results if pk_field_name in row}
                existing_pks.update(batch_existing)
            
            # 释放中间变量
            del results, batch_existing, batch_pks
        except Exception as e:
            # 如果查询失败，记录警告但继续处理
            print(f"  警告: 查询已存在主键失败（批次 {i//batch_size + 1}）: {e}")
            continue
    
    return existing_pks


def insert_data_batch(
    client: MilvusClient,
    collection_name: str,
    batch_data: List[Dict[str, Any]],
    auto_embed: bool = False,
    skip_existing: bool = False,
    logger: Optional[logging.Logger] = None
) -> Tuple[int, List[Any], List[Any]]:
    """
    插入一批数据到 collection
    
    Args:
        skip_existing: 是否跳过已存在的记录（基于主键）
        logger: 日志记录器（可选）
    
    Returns:
        (插入的记录数, 跳过的主键列表, 失败的主键列表)
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
    
    # 如果需要跳过已存在的记录，先查询已存在的主键
    skipped_pks = []
    existing_pks = None
    if skip_existing:
        pk_values = [row.get(pk_field_name) for row in batch_data if row.get(pk_field_name) is not None]
        if pk_values:
            # 分批查询，避免内存占用过大
            existing_pks = get_existing_pks(client, collection_name, pk_field_name, pk_values, batch_size=500)
            
            # 记录跳过的主键
            skipped_pks = [row.get(pk_field_name) for row in batch_data if row.get(pk_field_name) in existing_pks]
            
            # 过滤掉已存在的记录
            batch_data = [row for row in batch_data if row.get(pk_field_name) not in existing_pks]
            
            # 释放已存在主键集合
            del existing_pks
            gc.collect()
    
    if not batch_data:
        return (0, skipped_pks, [])  # 所有记录都已存在
    
    num_rows = len(batch_data)
    
    # 直接构建按行组织的数据，避免中间数据复制
    data_rows = []
    for row in batch_data:
        data_row = {}
        for field_name in schema_fields.keys():
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
            data_row[field_name] = value
        data_rows.append(data_row)
    
    # 释放原始批次数据
    del batch_data
    gc.collect()
    
    # 如果需要自动向量化且没有向量数据
    if "dense_vector" in schema_fields and auto_embed:
        # 检查是否需要向量化
        need_embedding = False
        text_values = []
        for row in data_rows:
            if not row.get("dense_vector"):
                text_val = row.get("text", "")
                if text_val:
                    need_embedding = True
                    text_values.append(text_val)
                else:
                    text_values.append("")
            else:
                text_values.append("")
        
        if need_embedding and client.embedding_service:
            # 只向量化有文本的行
            texts_to_embed = [text for text in text_values if text]
            if texts_to_embed:
                # 向量化时不显示进度条（避免与主进度条冲突）
                vectors = client.embedding_service.embeddings(
                    texts_to_embed,
                    batch_size=10,
                    show_progress=False
                )
                # 将向量分配回对应的行
                vector_idx = 0
                for i, row in enumerate(data_rows):
                    if text_values[i]:
                        row["dense_vector"] = vectors[vector_idx]
                        vector_idx += 1
                # 释放中间变量
                del vectors, texts_to_embed, text_values
                gc.collect()
    
    # 保存主键列表（用于错误处理）
    inserted_pks = [row.get(pk_field_name) for row in data_rows if row.get(pk_field_name) is not None]
    failed_pks = []
    
    try:
        # 插入数据
        insert_result = collection.insert(data_rows)
        collection.flush()
    except Exception as e:
        # 插入失败，记录所有主键
        failed_pks = inserted_pks.copy()
        if logger:
            logger.error(f"批量插入失败: {e}")
            for pk in failed_pks:
                logger.error(f"FAILED: {pk}")
        raise
    
    # 释放数据
    del data_rows
    gc.collect()
    
    return (num_rows, skipped_pks, failed_pks)


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

  # 跳过已存在的记录（基于主键）
  python insert2milvus.py -f data.csv -c my_collection --skip-existing
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
        "--skip-existing",
        action="store_true",
        help="跳过已存在的记录（基于主键判断）"
    )
    
    parser.add_argument(
        "--log-file",
        default=None,
        help="日志文件路径（记录跳过和失败的主键，默认: {collection_name}_insert.log）"
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
        
        # 设置日志记录器
        log_file = args.log_file or f"{args.collection}_insert.log"
        logger = logging.getLogger('insert2milvus')
        logger.setLevel(logging.INFO)
        
        # 清除已有的处理器
        logger.handlers.clear()
        
        # 文件处理器：记录跳过和失败的主键
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        # 控制台处理器：只记录重要信息
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        logger.addHandler(console_handler)
        
        logger.info(f"开始插入数据到 collection: {args.collection}")
        logger.info(f"输入文件: {args.file}")
        if args.skip_existing:
            logger.info("启用跳过已存在记录功能")
        
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
        
        # 先统计总行数（用于进度条）- 使用流式统计，避免加载整个文件到内存
        print("  正在统计文件总行数...")
        total_rows = 0
        if file_ext == '.csv':
            # CSV 文件：流式读取统计
            with open(args.file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                # 使用生成器表达式，避免一次性加载所有数据
                total_rows = sum(1 for _ in reader)
        else:
            # JSON 文件：尝试流式统计（如果文件很大）
            try:
                # 先尝试快速统计（只读取结构）
                with open(args.file, 'r', encoding='utf-8') as f:
                    # 对于大文件，使用流式解析（如果可能）
                    # 这里简化处理，对于 JSON 数组，需要完整加载
                    # 但我们可以先检查文件大小，如果太大就使用动态统计
                    import os
                    file_size = os.path.getsize(args.file)
                    if file_size > 100 * 1024 * 1024:  # 大于 100MB
                        print("  警告: JSON 文件较大，将使用动态统计（可能不准确）")
                        total_rows = None  # 使用 None 表示未知
                    else:
                        data = json.load(f)
                        total_rows = len(data) if isinstance(data, list) else 1
                        del data  # 立即释放
                        gc.collect()
            except Exception as e:
                print(f"  警告: 无法统计 JSON 文件行数: {e}，将使用动态统计")
                total_rows = None
        
        if total_rows is None:
            print("  将使用动态进度显示")
        elif total_rows > 0:
            print(f"  文件共有 {total_rows} 条数据")
        else:
            print("  警告: 文件为空或无法读取数据")
            sys.exit(0)
        
        # 读取文件（流式处理）
        if file_ext == '.csv':
            data_reader = read_csv_file(args.file, batch_size=args.batch_size)
        else:
            data_reader = read_json_file(args.file, batch_size=args.batch_size)
        
        total_inserted = 0
        total_skipped = 0
        batch_num = 0
        
        # 使用进度条（如果总行数未知，使用动态模式）
        pbar = tqdm(total=total_rows, desc="插入进度", unit="条") if total_rows else tqdm(desc="插入进度", unit="条")
        
        try:
            with pbar:
                for batch_data in data_reader:
                    if not batch_data:
                        continue
                    
                    batch_num += 1
                    batch_size = len(batch_data)
                    
                    try:
                        inserted, skipped_pks, failed_pks = insert_data_batch(
                            client=client,
                            collection_name=args.collection,
                            batch_data=batch_data,
                            auto_embed=args.auto_embed,
                            skip_existing=args.skip_existing,
                            logger=logger
                        )
                        
                        # 记录跳过的主键
                        if skipped_pks:
                            for pk in skipped_pks:
                                logger.info(f"SKIPPED: {pk}")
                            total_skipped += len(skipped_pks)
                        
                        # 记录失败的主键
                        if failed_pks:
                            for pk in failed_pks:
                                logger.error(f"FAILED: {pk}")
                        
                        total_inserted += inserted
                        
                        # 更新进度条
                        if total_rows:
                            pbar.update(batch_size)
                        else:
                            pbar.update(batch_size)
                            pbar.total = total_inserted + total_skipped
                        
                        pbar.set_postfix({
                            '已插入': total_inserted,
                            '已跳过': total_skipped,
                            '批次': batch_num
                        })
                        
                        # 每批都执行垃圾回收（对于大数据量很重要）
                        gc.collect()
                    except Exception as e:
                        print(f"\n错误: 插入失败（批次 {batch_num}）: {e}", file=sys.stderr)
                        import traceback
                        traceback.print_exc()
                        
                        # 记录整个批次失败的主键
                        collection = client.get_collection(args.collection)
                        schema_fields = {field.name: field for field in collection.schema.fields}
                        pk_field_name = next((name for name, field in schema_fields.items() if field.is_primary), None)
                        if pk_field_name:
                            failed_pks = [row.get(pk_field_name) for row in batch_data if row.get(pk_field_name) is not None]
                            for pk in failed_pks:
                                logger.error(f"FAILED: {pk} (批次 {batch_num} 插入失败: {str(e)})")
                        
                        # 更新进度条（即使失败也更新）
                        if total_rows:
                            pbar.update(batch_size)
                        else:
                            pbar.update(batch_size)
                        # 继续处理下一批
                        continue
        finally:
            # 确保进度条关闭
            if hasattr(pbar, 'close'):
                pbar.close()
        
        print(f"\n插入完成！")
        print(f"  共处理: {total_rows} 条")
        print(f"  已插入: {total_inserted} 条")
        if args.skip_existing:
            print(f"  已跳过: {total_skipped} 条（已存在）")
        
        # 记录完成信息到日志
        logger.info(f"插入完成 - 共处理: {total_rows} 条, 已插入: {total_inserted} 条, 已跳过: {total_skipped} 条")
        print(f"\n日志文件: {log_file}")
        
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

