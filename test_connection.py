#!/usr/bin/env python3
"""简单的连接测试脚本"""
import sys

print("测试 1: 导入模块...")
try:
    from chunk2milvus_hq import MilvusClient, EmbeddingService
    print("✓ 模块导入成功")
except Exception as e:
    print(f"✗ 模块导入失败: {e}")
    sys.exit(1)

print("\n测试 2: 初始化 EmbeddingService...")
try:
    embedding_service = EmbeddingService()
    print(f"✓ EmbeddingService 初始化成功")
    print(f"  - Model: {embedding_service.model}")
    print(f"  - Dimension: {embedding_service.dimension}")
except Exception as e:
    print(f"✗ EmbeddingService 初始化失败: {e}")
    sys.exit(1)

print("\n测试 3: 初始化 MilvusClient...")
try:
    client = MilvusClient(embedding_service=embedding_service)
    print("✓ MilvusClient 初始化成功")
    print(f"  - URI: {client.uri}")
except Exception as e:
    print(f"✗ MilvusClient 初始化失败: {e}")
    sys.exit(1)

print("\n测试 4: 列出 collections...")
try:
    collections = client.list_collections()
    print(f"✓ 成功列出 collections: {collections}")
except Exception as e:
    print(f"✗ 列出 collections 失败: {e}")
    sys.exit(1)

print("\n所有测试通过！✓")

