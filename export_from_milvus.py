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
from typing import List, Dict, Any, Optional
from pathlib import Path

from chunk2milvus_hq import MilvusClient


def query_collection(
    client: MilvusClient,
    collection_name: str,
    limit: Optional[int] = None,
    fields: Optional[List[str]] = None,
    expr: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    从 collection 查询数据
    
    Args:
        client: MilvusClient 实例
        collection_name: collection 名称
        limit: 查询数量限制，None 表示查询全部
        fields: 要查询的字段列表，None 表示查询所有字段
        expr: 过滤表达式（可选）
        
    Returns:
        查询结果列表
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
    if limit is not None and limit > 0:
        results = collection.query(
            expr=expr if expr else "",
            output_fields=fields,
            limit=limit
        )
    else:
        # 导出全部数据
        # 先获取总数
        num_entities = collection.num_entities
        print(f"Collection 总共有 {num_entities} 条数据")
        
        # Milvus 的 query 方法不支持 offset，但可以通过设置一个很大的 limit 来获取所有数据
        # 设置 limit 为总数 + 1000 以确保获取所有数据
        print("  正在查询所有数据（可能需要一些时间）...")
        results = collection.query(
            expr=expr if expr else "",
            output_fields=fields,
            limit=num_entities + 1000  # 设置一个比总数稍大的 limit 以确保获取所有数据
        )
    
    return results


def export_to_json(data: List[Dict[str, Any]], output_file: str):
    """导出数据到 JSON 文件"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"数据已导出到 JSON 文件: {output_file}")


def export_to_csv(data: List[Dict[str, Any]], output_file: str):
    """导出数据到 CSV 文件"""
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
    
    # 检查输出文件是否已存在
    if Path(output_file).exists():
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
        
        data = query_collection(
            client=client,
            collection_name=args.collection,
            limit=limit,
            fields=args.fields,
            expr=args.expr
        )
        
        print(f"\n查询到 {len(data)} 条数据")
        
        if not data:
            print("警告: 没有数据可导出")
            sys.exit(0)
        
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

