"""
MilvusClient 使用示例

环境变量配置（可选）：
在项目根目录创建 .env 文件，包含以下配置：
    MILVUS_URI=http://localhost:19530
    MILVUS_TOKEN=
    EMBEDDING_KEY=your-api-key-here
    EMBEDDING_URL=https://api.openai.com/v1
    EMBEDDING_MODEL=text-embedding-3-large
    EMBEDDING_DIMENSION=3072

如果不设置环境变量，也可以通过参数传入配置。
"""
# 重要：必须先导入 chunk2milvus 以规范化环境变量，再导入 pymilvus
from chunk2milvus_hq import MilvusClient, EmbeddingService
from pymilvus import FieldSchema, DataType


def main():
    # 方式1: 使用环境变量（推荐）
    # 如果设置了环境变量，可以直接使用默认参数
    embedding_service = EmbeddingService()  # 从环境变量读取配置
    client = MilvusClient(embedding_service=embedding_service)  # 从环境变量读取配置
    
    # 方式2: 显式传入参数
    # embedding_service = EmbeddingService(
    #     api_key="your-api-key",
    #     base_url="https://api.openai.com/v1",
    #     model="text-embedding-3-large",
    #     dimension=4096
    # )
    # client = MilvusClient(
    #     uri="http://localhost:19530",
    #     token=None,  # 如果有认证 token，在这里传入
    #     embedding_service=embedding_service
    # )
    
    # 3. 定义字段 schema
    # 注意：dense_vector 的维度需要与 EMBEDDING_DIMENSION 环境变量一致
    # 如果使用环境变量，可以从 embedding_service 获取维度
    vector_dim = embedding_service.dimension if embedding_service else 2048
    
    fields = [
        FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=1024),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535, enable_analyzer=True),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
        FieldSchema(name="metadata", dtype=DataType.JSON)
    ]
    
    collection_name = "test_collection2"
    
    # 4. 创建 collection（如果已存在会报错）
    try:
        collection = client.create_collection(
            collection_name=collection_name,
            fields=fields,
            description="测试 collection",
            enable_dynamic_field=False
        )
        print(f"Collection '{collection_name}' created successfully")
    except ValueError as e:
        print(f"Collection already exists: {e}")
        # 如果已存在，可以获取现有的 collection
        collection = client.get_collection(collection_name)
    
    # 5. 创建索引
    dense_index = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 32, "efConstruction": 300}
    }
    
    try:
        client.create_index(collection_name, "dense_vector", dense_index)
        print("Index created successfully")
    except Exception as e:
        print(f"Index creation failed (may already exist): {e}")
    
    # 6. 插入文本块
    texts = [
        "这是第一段文本",
        "这是第二段文本",
        "这是第三段文本"
    ]
    
    doc_ids = ["doc1", "doc2", "doc3"]
    metadatas = [
        {"source": "file1", "page": 1},
        {"source": "file1", "page": 2},
        {"source": "file2", "page": 1}
    ]
    
    ids = client.insert_texts(
        collection_name=collection_name,
        texts=texts,
        doc_ids=doc_ids,
        metadatas=metadatas,
        auto_embed=True  # 自动进行向量化
    )
    print(f"Inserted {len(ids)} texts with IDs: {ids}")
    
    # 7. 文本搜索
    query_text = "第一段"
    results = client.search_by_text(
        collection_name=collection_name,
        query_text=query_text,
        limit=5,
        output_fields=["text", "doc_id", "metadata"]
    )
    
    print(f"\nSearch results for '{query_text}':")
    for i, hit in enumerate(results):
        print(f"  {i+1}. ID: {hit['id']}, Distance: {hit['distance']:.4f}")
        print(f"     Text: {hit['entity'].get('text', 'N/A')}")
        print(f"     Doc ID: {hit['entity'].get('doc_id', 'N/A')}")
        print()
    
    # 8. 列出所有 collections
    collections = client.list_collections()
    print(f"All collections: {collections}")
    
    # 9. 检查 collection 是否存在
    exists = client.has_collection(collection_name)
    print(f"Collection '{collection_name}' exists: {exists}")
    
    # 10. 删除 collection（可选，取消注释以执行）
    # client.delete_collection(collection_name)
    # print(f"Collection '{collection_name}' deleted")


if __name__ == "__main__":
    main()

