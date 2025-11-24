#!/usr/bin/env python3
"""
脚本用于从CSV文件中剔除指定字段并保存为新的CSV文件

使用方法:
    python remove_fields_from_csv.py input.csv output.csv --fields field1 field2
    或
    python remove_fields_from_csv.py input.csv output.csv -f field1 -f field2
"""

import argparse
import csv
import sys
from pathlib import Path


def remove_fields_from_csv(input_file: str, output_file: str, fields_to_remove: list):
    """
    从CSV文件中剔除指定字段并保存为新文件
    
    Args:
        input_file: 输入CSV文件路径
        output_file: 输出CSV文件路径
        fields_to_remove: 要剔除的字段名列表
    """
    input_path = Path(input_file)
    if not input_path.exists():
        print(f"错误: 输入文件 '{input_file}' 不存在")
        sys.exit(1)
    
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            # 读取CSV文件
            reader = csv.DictReader(infile)
            
            # 获取所有字段名
            all_fields = reader.fieldnames
            if not all_fields:
                print("错误: CSV文件没有字段名")
                sys.exit(1)
            
            # 检查要剔除的字段是否存在
            fields_to_remove_set = set(fields_to_remove)
            existing_fields_to_remove = []
            non_existing_fields = []
            
            for field in fields_to_remove:
                if field in all_fields:
                    existing_fields_to_remove.append(field)
                else:
                    non_existing_fields.append(field)
            
            if non_existing_fields:
                print(f"警告: 以下字段在CSV中不存在: {', '.join(non_existing_fields)}")
            
            if not existing_fields_to_remove:
                print("错误: 没有找到任何要剔除的字段")
                sys.exit(1)
            
            # 计算要保留的字段
            fields_to_keep = [field for field in all_fields if field not in fields_to_remove_set]
            
            if not fields_to_keep:
                print("错误: 剔除所有字段后没有剩余字段")
                sys.exit(1)
            
            print(f"原始字段: {', '.join(all_fields)}")
            print(f"要剔除的字段: {', '.join(existing_fields_to_remove)}")
            print(f"保留的字段: {', '.join(fields_to_keep)}")
            
            # 读取所有数据
            rows = []
            for row in reader:
                # 只保留需要的字段
                filtered_row = {field: row[field] for field in fields_to_keep}
                rows.append(filtered_row)
            
            print(f"共处理 {len(rows)} 行数据")
        
        # 写入新文件
        with open(output_file, 'w', encoding='utf-8', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fields_to_keep)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"成功! 新文件已保存到: {output_file}")
        
    except Exception as e:
        print(f"错误: 处理文件时发生异常: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='从CSV文件中剔除指定字段并保存为新文件',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 剔除单个字段
  python remove_fields_from_csv.py input.csv output.csv -f dense_vector
  
  # 剔除多个字段
  python remove_fields_from_csv.py input.csv output.csv -f dense_vector -f metadata
  
  # 使用--fields参数
  python remove_fields_from_csv.py input.csv output.csv --fields dense_vector metadata
        """
    )
    
    parser.add_argument('input_file', help='输入CSV文件路径')
    parser.add_argument('output_file', help='输出CSV文件路径')
    parser.add_argument(
        '-f', '--fields',
        action='append',
        dest='fields_to_remove',
        required=True,
        help='要剔除的字段名（可多次使用此参数指定多个字段）'
    )
    
    args = parser.parse_args()
    
    remove_fields_from_csv(args.input_file, args.output_file, args.fields_to_remove)


if __name__ == '__main__':
    main()

