#!/usr/bin/env python3
"""
从CSV文件中导出pk字段，并统计pk数量和去重后的pk数量
"""

import argparse
import csv
import sys
from pathlib import Path
from typing import Set, List


def export_pk_from_csv(csv_file: str, output_file: str = None):
    """
    从CSV文件中导出pk字段，并统计数量
    
    Args:
        csv_file: 输入的CSV文件路径
        output_file: 输出的pk文件路径（可选，如果不指定则不导出文件）
    """
    csv_path = Path(csv_file)
    
    if not csv_path.exists():
        print(f"错误: 文件不存在: {csv_file}", file=sys.stderr)
        sys.exit(1)
    
    if not csv_path.is_file():
        print(f"错误: 不是文件: {csv_file}", file=sys.stderr)
        sys.exit(1)
    
    pks: List[str] = []
    unique_pks: Set[str] = set()
    
    print(f"正在读取CSV文件: {csv_file}")
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            # 检测CSV文件是否有BOM
            first_char = f.read(1)
            f.seek(0)
            if first_char != '\ufeff':
                f.seek(0)
            
            reader = csv.DictReader(f)
            
            # 检查是否有pk字段
            if 'pk' not in reader.fieldnames:
                print(f"错误: CSV文件中没有找到'pk'字段", file=sys.stderr)
                print(f"可用的字段: {', '.join(reader.fieldnames)}", file=sys.stderr)
                sys.exit(1)
            
            # 读取所有pk值
            for row_num, row in enumerate(reader, start=2):  # 从第2行开始（第1行是表头）
                pk = row.get('pk', '').strip()
                if pk:  # 只处理非空的pk
                    pks.append(pk)
                    unique_pks.add(pk)
    
    except Exception as e:
        print(f"错误: 读取CSV文件时出错: {e}", file=sys.stderr)
        sys.exit(1)
    
    # 统计信息
    total_count = len(pks)
    unique_count = len(unique_pks)
    duplicate_count = total_count - unique_count
    
    print(f"\n统计结果:")
    print(f"  PK总数: {total_count:,}")
    print(f"  去重后PK数量: {unique_count:,}")
    print(f"  重复PK数量: {duplicate_count:,}")
    
    if unique_count > 0:
        duplicate_rate = (duplicate_count / total_count) * 100
        print(f"  重复率: {duplicate_rate:.2f}%")
    
    # 如果有重复的pk，显示一些示例
    if duplicate_count > 0:
        # 找出重复的pk
        pk_counts = {}
        for pk in pks:
            pk_counts[pk] = pk_counts.get(pk, 0) + 1
        
        duplicate_pks = {pk: count for pk, count in pk_counts.items() if count > 1}
        print(f"\n重复的PK示例（前10个）:")
        for i, (pk, count) in enumerate(list(duplicate_pks.items())[:10], 1):
            print(f"  {i}. {pk[:80]}... (出现{count}次)")
    
    # 导出pk到文件（如果指定了输出文件）
    if output_file:
        output_path = Path(output_file)
        print(f"\n正在导出PK到文件: {output_file}")
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                # 写入表头
                f.write("pk\n")
                # 写入所有唯一的pk
                for pk in sorted(unique_pks):
                    f.write(f"{pk}\n")
            
            print(f"成功导出 {unique_count:,} 个唯一的PK到: {output_file}")
        except Exception as e:
            print(f"错误: 导出文件时出错: {e}", file=sys.stderr)
            sys.exit(1)
    
    return {
        'total_count': total_count,
        'unique_count': unique_count,
        'duplicate_count': duplicate_count,
        'pks': list(unique_pks)
    }


def main():
    parser = argparse.ArgumentParser(
        description='从CSV文件中导出pk字段，并统计pk数量和去重后的pk数量',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 只统计，不导出文件
  python export_pk_from_csv.py output/output.csv
  
  # 统计并导出到文件
  python export_pk_from_csv.py output/output.csv -o pks.txt
  
  # 导出到CSV格式
  python export_pk_from_csv.py output/output.csv -o pks.csv
        """
    )
    
    parser.add_argument(
        'csv_file',
        type=str,
        help='输入的CSV文件路径'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=None,
        help='输出的pk文件路径（可选，如果不指定则不导出文件）'
    )
    
    args = parser.parse_args()
    
    export_pk_from_csv(args.csv_file, args.output)


if __name__ == '__main__':
    main()

