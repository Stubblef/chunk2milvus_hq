#!/usr/bin/env python3
"""
从 Milvus 导出数据的脚本

支持功能：
1. 指定 alias 和 collection 名称
2. 指定导出数量或导出全部数据
3. 指定导出字段
4. 选择导出格式（CSV 或 JSON）
"""

import argparse
import json
import csv
import sys
import gc
from typing import List, Dict, Any, Optional, Callable
from pathlib import Path

from chunk2milvus_hq import MilvusClient


def query_collection(
    client: MilvusClient,
    collection_name: str,
    limit: Optional[int] = None,
    fields: Optional[List[str]] = None,
    expr: Optional[str] = None,
    batch_callback: Optional[Callable[[List[Dict[str, Any]]], None]] = None
) -> Optional[List[Dict[str, Any]]]:
    """
    从 collection 查询数据
    
    Args:
        client: MilvusClient 实例
        collection_name: collection 名称
        limit: 查询数量限制，None 表示查询全部
        fields: 要查询的字段列表，None 表示查询所有字段
        expr: 过滤表达式（可选）
        batch_callback: 批次回调函数，每批数据查询后立即调用（用于流式处理）
                       如果提供，函数返回 None，否则返回所有结果列表
        
    Returns:
        查询结果列表（如果 batch_callback 为 None），否则返回 None
    """
    collection = client.get_collection(collection_name)
    
    # 加载 collection
    collection.load()
    
    # 如果没有指定字段，获取所有字段（排除向量字段，因为向量字段通常很大）
    if fields is None:
        schema_fields = {field.name: field for field in collection.schema.fields}
        # 默认排除向量字段，如果用户需要可以显式指定
        # fields = [name for name, field in schema_fields.items() 
        #          if field.dtype.name != "FLOAT_VECTOR"]
        fields = [name for name, field in schema_fields.items()]  # 导出所有字段
        if not fields:
            # 如果没有其他字段，至少包含主键
            pk_field = next((name for name, field in schema_fields.items() if field.is_primary), None)
            if pk_field:
                fields = [pk_field]
    
    # 执行查询
    # pymilvus 的 query 方法支持 limit 和 expr 参数
    # 注意：Milvus 的 query 方法限制 limit 最大为 16384
    MAX_QUERY_LIMIT = 16384
    
    if limit is not None and limit > 0:
        # 如果 limit 超过最大值，需要分批查询
        if limit > MAX_QUERY_LIMIT:
            print(f"  注意: limit ({limit}) 超过 Milvus 最大限制 ({MAX_QUERY_LIMIT})，将分批查询...")
            results = []
            remaining = limit
            batch_num = 0
            
            # 获取主键字段用于分批查询
            pk_field = None
            for field in collection.schema.fields:
                if field.is_primary:
                    pk_field = field.name
                    break
            
            if pk_field:
                # 使用主键范围查询来分批获取数据
                # 先获取所有主键（分批获取）
                all_pks_set = set()  # 使用 set 去重
                all_pks = []  # 保持顺序的列表
                pk_batch_size = MAX_QUERY_LIMIT
                pk_offset = 0
                
                while pk_offset < limit:
                    pk_batch_limit = min(pk_batch_size, limit - pk_offset)
                    pk_batch = collection.query(
                        expr=expr if expr else "",
                        output_fields=[pk_field],
                        limit=pk_batch_limit
                    )
                    
                    if not pk_batch:
                        break
                    
                    # 去重并保持顺序
                    for row in pk_batch:
                        pk_value = row[pk_field]
                        if pk_value not in all_pks_set:
                            all_pks_set.add(pk_value)
                            all_pks.append(pk_value)
                    
                    pk_offset = len(all_pks)  # 使用实际去重后的数量
                    
                    if len(pk_batch) < pk_batch_limit or len(all_pks) >= limit:
                        break
                
                # 使用主键列表分批查询数据
                # 使用较小的批次以避免表达式过长（表达式长度限制）
                data_batch_size = min(1000, MAX_QUERY_LIMIT)  # 使用较小的批次
                for i in range(0, len(all_pks), data_batch_size):
                    batch_pks = all_pks[i:i + data_batch_size]
                    batch_num += 1
                    print(f"  正在查询第 {batch_num} 批数据 ({len(batch_pks)} 条)...")
                    
                    # 构建主键过滤表达式
                    if pk_field:
                        # 根据主键类型构建表达式
                        pk_field_type = next((f.dtype for f in collection.schema.fields if f.name == pk_field), None)
                        if pk_field_type and pk_field_type.name in ["INT64", "VARCHAR"]:
                            try:
                                if pk_field_type.name == "VARCHAR":
                                    # VARCHAR 类型使用 in 表达式
                                    pk_list_str = ", ".join([f'"{pk}"' for pk in batch_pks])
                                    batch_expr = f"{pk_field} in [{pk_list_str}]"
                                else:
                                    # INT64 类型
                                    pk_list_str = ", ".join([str(pk) for pk in batch_pks])
                                    batch_expr = f"{pk_field} in [{pk_list_str}]"
                                
                                # 如果原有 expr，需要合并
                                if expr:
                                    batch_expr = f"({expr}) && ({batch_expr})"
                                
                                batch_results = collection.query(
                                    expr=batch_expr,
                                    output_fields=fields,
                                    limit=len(batch_pks)
                                )
                            except Exception as e:
                                # 如果表达式过长或其他错误，回退到简单查询
                                print(f"    警告: 使用主键表达式查询失败，回退到简单查询: {e}")
                                batch_results = collection.query(
                                    expr=expr if expr else "",
                                    output_fields=fields,
                                    limit=min(data_batch_size, remaining)
                                )
                        else:
                            # 如果无法构建表达式，直接查询（可能重复）
                            batch_results = collection.query(
                                expr=expr if expr else "",
                                output_fields=fields,
                                limit=min(data_batch_size, remaining)
                            )
                    else:
                        batch_results = collection.query(
                            expr=expr if expr else "",
                            output_fields=fields,
                            limit=min(data_batch_size, remaining)
                        )
                    
                    # 如果提供了回调函数，立即处理这批数据（流式处理）
                    if batch_callback:
                        batch_callback(batch_results)
                        del batch_results  # 立即释放内存
                        gc.collect()  # 强制垃圾回收
                    else:
                        results.extend(batch_results)
                    remaining -= len(batch_results)
                    
                    if remaining <= 0 or len(batch_results) < data_batch_size:
                        break
            else:
                # 没有主键字段，直接分批查询（可能重复）
                while remaining > 0:
                    batch_num += 1
                    batch_limit = min(MAX_QUERY_LIMIT, remaining)
                    print(f"  正在查询第 {batch_num} 批数据 (最多 {batch_limit} 条)...")
                    
                    batch_results = collection.query(
                        expr=expr if expr else "",
                        output_fields=fields,
                        limit=batch_limit
                    )
                    
                    if not batch_results:
                        break
                    
                    # 如果提供了回调函数，立即处理这批数据（流式处理）
                    if batch_callback:
                        batch_callback(batch_results)
                        del batch_results  # 立即释放内存
                        gc.collect()  # 强制垃圾回收
                    else:
                        results.extend(batch_results)
                    remaining -= len(batch_results)
                    
                    if len(batch_results) < batch_limit:
                        break
        else:
            # limit 在允许范围内，直接查询
            batch_results = collection.query(
                expr=expr if expr else "",
                output_fields=fields,
                limit=limit
            )
            # 如果提供了回调函数，立即处理
            if batch_callback:
                batch_callback(batch_results)
                return None
            else:
                results = batch_results
    else:
        # 导出全部数据，需要分批查询
        num_entities = collection.num_entities
        print(f"Collection 总共有 {num_entities} 条数据")
        
        if num_entities == 0:
            return []
        
        print("  正在分批查询所有数据（可能需要一些时间）...")
        results = [] if batch_callback is None else None
        batch_num = 0
        total_queried = 0
        
        # 获取主键字段用于分批查询
        pk_field = None
        for field in collection.schema.fields:
            if field.is_primary:
                pk_field = field.name
                break
        
        if pk_field:
            # 使用主键分批查询
            # 先分批获取所有主键（流式处理，不全部保存在内存）
            print("  正在获取主键列表...")
            all_pks_set = set()  # 使用 set 去重
            all_pks = []  # 保持顺序的列表
            pk_batch_size = MAX_QUERY_LIMIT
            pk_offset = 0
            
            while pk_offset < num_entities:
                pk_batch_limit = min(pk_batch_size, num_entities - pk_offset)
                pk_batch = collection.query(
                    expr=expr if expr else "",
                    output_fields=[pk_field],
                    limit=pk_batch_limit
                )
                
                if not pk_batch:
                    break
                
                # 去重并保持顺序
                for row in pk_batch:
                    pk_value = row[pk_field]
                    if pk_value not in all_pks_set:
                        all_pks_set.add(pk_value)
                        all_pks.append(pk_value)
                
                pk_offset = len(all_pks)  # 使用实际去重后的数量
                
                print(f"    已获取 {len(all_pks)}/{num_entities} 个主键（已去重）...")
                
                if len(pk_batch) < pk_batch_limit:
                    break
            
            # 使用主键列表分批查询数据
            # 使用较小的批次以避免表达式过长（表达式长度限制）
            data_batch_size = min(1000, MAX_QUERY_LIMIT)  # 使用较小的批次
            for i in range(0, len(all_pks), data_batch_size):
                batch_pks = all_pks[i:i + data_batch_size]
                batch_num += 1
                total_queried += len(batch_pks)
                print(f"  正在查询第 {batch_num} 批数据 ({len(batch_pks)} 条, 进度: {total_queried}/{len(all_pks)})...")
                
                # 构建主键过滤表达式
                pk_field_type = next((f.dtype for f in collection.schema.fields if f.name == pk_field), None)
                if pk_field_type:
                    try:
                        if pk_field_type.name == "VARCHAR":
                            # VARCHAR 类型使用 in 表达式
                            pk_list_str = ", ".join([f'"{pk}"' for pk in batch_pks])
                            batch_expr = f"{pk_field} in [{pk_list_str}]"
                        elif pk_field_type.name == "INT64":
                            # INT64 类型
                            pk_list_str = ", ".join([str(pk) for pk in batch_pks])
                            batch_expr = f"{pk_field} in [{pk_list_str}]"
                        else:
                            # 其他类型，尝试转换为字符串
                            pk_list_str = ", ".join([f'"{str(pk)}"' for pk in batch_pks])
                            batch_expr = f"{pk_field} in [{pk_list_str}]"
                        
                        # 如果原有 expr，需要合并
                        if expr:
                            batch_expr = f"({expr}) && ({batch_expr})"
                        
                        batch_results = collection.query(
                            expr=batch_expr,
                            output_fields=fields,
                            limit=len(batch_pks)
                        )
                    except Exception as e:
                        # 如果表达式过长或其他错误，回退到简单查询
                        print(f"    警告: 使用主键表达式查询失败，回退到简单查询: {e}")
                        batch_results = collection.query(
                            expr=expr if expr else "",
                            output_fields=fields,
                            limit=len(batch_pks)
                        )
                else:
                    batch_results = collection.query(
                        expr=expr if expr else "",
                        output_fields=fields,
                        limit=len(batch_pks)
                    )
                
                # 如果提供了回调函数，立即处理这批数据（流式处理）
                if batch_callback:
                    batch_callback(batch_results)
                    del batch_results  # 立即释放内存
                    # 每10批执行一次垃圾回收，避免频繁GC影响性能
                    if batch_num % 10 == 0:
                        gc.collect()
                else:
                    results.extend(batch_results)
        else:
            # 没有主键字段，使用简单分批查询（可能无法完全避免重复）
            print("  警告: 未找到主键字段，使用简单分批查询（可能无法完全避免重复数据）")
            batch_size = MAX_QUERY_LIMIT
            while total_queried < num_entities:
                batch_num += 1
                batch_limit = min(batch_size, num_entities - total_queried)
                print(f"  正在查询第 {batch_num} 批数据 (最多 {batch_limit} 条, 进度: {total_queried}/{num_entities})...")
                
                batch_results = collection.query(
                    expr=expr if expr else "",
                    output_fields=fields,
                    limit=batch_limit
                )
                
                if not batch_results:
                    break
                
                # 如果提供了回调函数，立即处理这批数据（流式处理）
                if batch_callback:
                    batch_callback(batch_results)
                    del batch_results  # 立即释放内存
                    # 每10批执行一次垃圾回收
                    if batch_num % 10 == 0:
                        gc.collect()
                else:
                    results.extend(batch_results)
                total_queried += len(batch_results)
                
                if len(batch_results) < batch_limit:
                    break
    
    return results


class StreamingJSONWriter:
    """流式 JSON 写入器"""
    def __init__(self, output_file: str, pk_field: Optional[str] = None):
        self.output_file = output_file
        self.file = open(output_file, 'w', encoding='utf-8')
        self.file.write('[\n')
        self.first_item = True
        self.count = 0
        self.pk_field = pk_field
        self.seen_pks = set()  # 用于去重的主键集合
    
    def write(self, data: List[Dict[str, Any]]):
        """写入一批数据"""
        for item in data:
            # 如果指定了主键字段，进行去重
            if self.pk_field and self.pk_field in item:
                pk_value = item[self.pk_field]
                if pk_value in self.seen_pks:
                    continue  # 跳过重复的主键
                self.seen_pks.add(pk_value)
            
            if not self.first_item:
                self.file.write(',\n')
            json.dump(item, self.file, ensure_ascii=False, indent=2)
            self.first_item = False
            self.count += 1
    
    def close(self):
        """关闭文件"""
        self.file.write('\n]')
        self.file.close()
        print(f"数据已导出到 JSON 文件: {self.output_file} (共 {self.count} 条)")


class StreamingCSVWriter:
    """流式 CSV 写入器"""
    def __init__(self, output_file: str, fieldnames: Optional[List[str]] = None, pk_field: Optional[str] = None):
        self.output_file = output_file
        self.file = open(output_file, 'w', encoding='utf-8', newline='')
        self.fieldnames = fieldnames
        self.writer = None
        self.header_written = False
        self.count = 0
        self.pk_field = pk_field
        self.seen_pks = set()  # 用于去重的主键集合
    
    def write(self, data: List[Dict[str, Any]]):
        """写入一批数据"""
        if not data:
            return
        
        # 第一次写入时，确定字段名并写入表头
        if not self.header_written:
            if self.fieldnames is None:
                self.fieldnames = list(data[0].keys())
            self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
            self.writer.writeheader()
            self.header_written = True
        
        # 写入数据
        for row in data:
            # 如果指定了主键字段，进行去重
            if self.pk_field and self.pk_field in row:
                pk_value = row[self.pk_field]
                if pk_value in self.seen_pks:
                    continue  # 跳过重复的主键
                self.seen_pks.add(pk_value)
            
            # 处理复杂数据类型（如列表、字典）转换为字符串
            processed_row = {}
            for key, value in row.items():
                if key in self.fieldnames:
                    if isinstance(value, (list, dict)):
                        processed_row[key] = json.dumps(value, ensure_ascii=False)
                    else:
                        processed_row[key] = value
            self.writer.writerow(processed_row)
            self.count += 1
    
    def close(self):
        """关闭文件"""
        self.file.close()
        print(f"数据已导出到 CSV 文件: {self.output_file} (共 {self.count} 条)")


def export_to_json(data: List[Dict[str, Any]], output_file: str):
    """导出数据到 JSON 文件（兼容旧接口）"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"数据已导出到 JSON 文件: {output_file}")


def export_to_csv(data: List[Dict[str, Any]], output_file: str):
    """导出数据到 CSV 文件（兼容旧接口）"""
    if not data:
        print("警告: 没有数据可导出")
        return
    
    # 获取所有字段名
    fieldnames = list(data[0].keys())
    
    with open(output_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        
        for row in data:
            # 处理复杂数据类型（如列表、字典）转换为字符串
            processed_row = {}
            for key, value in row.items():
                if isinstance(value, (list, dict)):
                    processed_row[key] = json.dumps(value, ensure_ascii=False)
                else:
                    processed_row[key] = value
            writer.writerow(processed_row)
    
    print(f"数据已导出到 CSV 文件: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="从 Milvus 导出数据",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 导出指定 collection 的前 100 条数据到 JSON
  python export_from_milvus.py -c my_collection -n 100 -f json -o output.json

  # 导出指定 collection 的全部数据到 CSV
  python export_from_milvus.py -c my_collection -a all -f csv -o output.csv

  # 导出指定字段的数据
  python export_from_milvus.py -c my_collection -n 100 --fields pk text doc_id -f json

  # 使用指定的 alias 连接
  python export_from_milvus.py -c my_collection -a my_alias -n 100 -f json

  # 指定数据库名称
  python export_from_milvus.py -c my_collection -d test -n 100 -f json

  # 大数据量导出，每100批保存一个文件（避免内存溢出）
  python export_from_milvus.py -c my_collection --batch-save 100 -f json -o output.json
        """
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
        "-n", "--limit",
        type=int,
        default=None,
        help="导出数量限制（默认: 导出全部数据）。使用 -1 或 'all' 表示导出全部"
    )
    
    parser.add_argument(
        "--fields",
        nargs="+",
        default=None,
        help="要导出的字段列表（默认: 导出所有非向量字段）"
    )
    
    parser.add_argument(
        "-f", "--format",
        choices=["json", "csv"],
        default="json",
        help="导出格式（默认: json）"
    )
    
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="输出文件路径（默认: {collection_name}.{format}）"
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
    
    parser.add_argument(
        "--expr",
        default=None,
        help="过滤表达式（可选，例如: 'doc_id == \"doc1\"'）"
    )
    
    parser.add_argument(
        "--batch-save",
        type=int,
        default=0,
        help="每N批数据保存一个文件（0表示不分批，全部保存到一个文件）。用于大数据量导出，避免内存溢出"
    )
    
    args = parser.parse_args()
    
    # 处理 limit 参数
    if args.limit is not None:
        if args.limit == -1:
            limit = None  # 导出全部
        elif args.limit <= 0:
            print("错误: limit 必须大于 0 或等于 -1（表示全部）", file=sys.stderr)
            sys.exit(1)
        else:
            limit = args.limit
    else:
        limit = None  # 默认导出全部
    
    # 确定输出文件路径
    if args.output:
        output_file = args.output
    else:
        output_file = f"{args.collection}.{args.format}"
    
    # 确保输出目录存在（如果指定了目录）
    output_path = Path(output_file)
    if output_path.parent and str(output_path.parent) != '.':
        output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 检查输出文件是否已存在（仅在非分批保存模式下检查）
    if args.batch_save == 0 and Path(output_file).exists():
        response = input(f"文件 {output_file} 已存在，是否覆盖？(y/N): ")
        if response.lower() != 'y':
            print("操作已取消")
            sys.exit(0)
    
    try:
        # 创建 Milvus 客户端
        database_info = f", database: {args.database}" if args.database else ""
        print(f"正在连接到 Milvus (alias: {args.alias}{database_info})...")
        client = MilvusClient(
            uri=args.uri,
            token=args.token,
            alias=args.alias,
            database=args.database
        )
        
        # 检查连接
        if not client.check_connection():
            print("错误: 无法连接到 Milvus", file=sys.stderr)
            sys.exit(1)
        
        conn_info = client.get_connection_info()
        print(f"连接成功: {conn_info.get('address', 'N/A')}")
        
        # 检查 collection 是否存在
        if not client.has_collection(args.collection):
            print(f"错误: Collection '{args.collection}' 不存在", file=sys.stderr)
            sys.exit(1)
        
        # 查询数据
        print(f"\n正在从 collection '{args.collection}' 查询数据...")
        if limit:
            print(f"限制数量: {limit}")
        else:
            print("导出全部数据")
        
        if args.fields:
            print(f"导出字段: {', '.join(args.fields)}")
        
        # 确定是否使用流式写入（大数据量或指定了分批保存）
        use_streaming = (limit is None or limit > 10000) or args.batch_save > 0
        
        if use_streaming:
            # 使用流式写入，避免内存溢出
            print("  使用流式写入模式（避免内存溢出）...")
            
            # 获取字段列表和主键字段（用于CSV写入器和去重）
            collection = client.get_collection(args.collection)
            collection.load()
            schema_fields = {field.name: field for field in collection.schema.fields}
            
            # 获取主键字段名
            pk_field = None
            for field in collection.schema.fields:
                if field.is_primary:
                    pk_field = field.name
                    break
            
            if args.fields:
                fieldnames = args.fields
            else:
                fieldnames = [name for name, field in schema_fields.items()]
            
            # 创建写入器
            if args.batch_save > 0:
                # 分批保存文件
                file_counter = 1
                batch_counter = 0
                current_writer = None
                
                def batch_callback(batch_data: List[Dict[str, Any]]):
                    nonlocal file_counter, batch_counter, current_writer
                    
                    if current_writer is None:
                        # 创建新文件
                        output_path = Path(output_file)
                        base_name = output_path.stem
                        base_ext = output_path.suffix or f".{args.format}"
                        base_dir = output_path.parent
                        
                        # 确保目录存在
                        if base_dir and str(base_dir) != '.':
                            base_dir.mkdir(parents=True, exist_ok=True)
                        
                        new_file = base_dir / f"{base_name}_part{file_counter:04d}{base_ext}"
                        print(f"\n  开始写入文件: {new_file}")
                        
                        if args.format == "json":
                            current_writer = StreamingJSONWriter(str(new_file), pk_field=pk_field)
                        else:
                            current_writer = StreamingCSVWriter(str(new_file), fieldnames, pk_field=pk_field)
                    
                    # 写入数据
                    current_writer.write(batch_data)
                    batch_counter += 1
                    
                    # 如果达到批次数量，关闭当前文件并创建新文件
                    if batch_counter >= args.batch_save:
                        current_writer.close()
                        current_writer = None
                        file_counter += 1
                        batch_counter = 0
                
                # 查询数据（使用回调）
                query_collection(
                    client=client,
                    collection_name=args.collection,
                    limit=limit,
                    fields=args.fields,
                    expr=args.expr,
                    batch_callback=batch_callback
                )
                
                # 关闭最后一个文件
                if current_writer is not None:
                    current_writer.close()
                
                print(f"\n导出完成！共生成 {file_counter} 个文件")
            else:
                # 流式写入到单个文件
                # 获取主键字段名（用于去重）
                collection = client.get_collection(args.collection)
                collection.load()
                pk_field = None
                for field in collection.schema.fields:
                    if field.is_primary:
                        pk_field = field.name
                        break
                
                if args.format == "json":
                    writer = StreamingJSONWriter(output_file, pk_field=pk_field)
                else:
                    writer = StreamingCSVWriter(output_file, fieldnames, pk_field=pk_field)
                
                def batch_callback(batch_data: List[Dict[str, Any]]):
                    writer.write(batch_data)
                
                # 查询数据（使用回调）
                query_collection(
                    client=client,
                    collection_name=args.collection,
                    limit=limit,
                    fields=args.fields,
                    expr=args.expr,
                    batch_callback=batch_callback
                )
                
                # 关闭文件
                writer.close()
                print(f"\n导出完成！")
        else:
            # 小数据量，使用传统方式（全部加载到内存）
            data = query_collection(
                client=client,
                collection_name=args.collection,
                limit=limit,
                fields=args.fields,
                expr=args.expr
            )
            
            if data is None:
                data = []
            
            print(f"\n查询到 {len(data)} 条数据")
            
            if not data:
                print("警告: 没有数据可导出")
                sys.exit(0)
            
            # 基于主键去重
            collection = client.get_collection(args.collection)
            collection.load()
            pk_field = None
            for field in collection.schema.fields:
                if field.is_primary:
                    pk_field = field.name
                    break
            
            if pk_field:
                seen_pks = set()
                deduplicated_data = []
                for row in data:
                    if pk_field in row:
                        pk_value = row[pk_field]
                        if pk_value not in seen_pks:
                            seen_pks.add(pk_value)
                            deduplicated_data.append(row)
                    else:
                        # 如果没有主键字段，保留所有数据
                        deduplicated_data.append(row)
                
                if len(deduplicated_data) < len(data):
                    print(f"  去重: {len(data)} 条 -> {len(deduplicated_data)} 条（基于主键 {pk_field}）")
                data = deduplicated_data
            
            # 导出数据
            print(f"\n正在导出数据到 {args.format.upper()} 文件...")
            if args.format == "json":
                export_to_json(data, output_file)
            else:
                export_to_csv(data, output_file)
            
            print(f"\n导出完成！共导出 {len(data)} 条数据")
        
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

