from typing import List, Dict, Any, Optional
import os
from pymilvus import (
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
    connections,
    utility,
)
from .embedding import EmbeddingService


class MilvusClient:
    """Milvus 客户端封装类"""

    def __init__(
        self,
        uri: Optional[str] = None,
        token: Optional[str] = None,
        embedding_service: Optional[EmbeddingService] = None,
        alias: str = "default"
    ):
        """
        初始化 Milvus 客户端
        
        Args:
            uri: Milvus 连接地址，格式如 "http://localhost:19530" 或 "localhost:19530"
                 如果为 None 则从环境变量 MILVUS_URI 读取，默认为 "http://localhost:19530"
            token: 认证 token（可选），如果为 None 则从环境变量 MILVUS_TOKEN 读取
            embedding_service: 向量化服务实例（可选）
            alias: 连接别名，默认为 "default"
        """
        # 从环境变量读取配置
        raw_uri = uri or os.getenv("MILVUS_URI", "http://localhost:19530")
        self.token = token or os.getenv("MILVUS_TOKEN")
        self.alias = alias
        self.embedding_service = embedding_service
        
        # 规范化 URI 格式
        # 如果 URI 不包含协议，自动添加 http://
        if not raw_uri.startswith("http://") and not raw_uri.startswith("https://"):
            # 检查是否包含用户认证信息（user:password@host）
            if "@" in raw_uri:
                # 已有认证信息，添加 http://
                self.uri = f"http://{raw_uri}"
            else:
                # 没有认证信息，添加 http://
                self.uri = f"http://{raw_uri}"
        else:
            self.uri = raw_uri
        
        # 解析 URI 获取 host 和 port
        # 移除协议前缀
        if self.uri.startswith("http://"):
            uri_without_protocol = self.uri[7:]  # 移除 "http://"
        elif self.uri.startswith("https://"):
            uri_without_protocol = self.uri[8:]  # 移除 "https://"
        else:
            uri_without_protocol = self.uri
        
        # 处理认证信息（user:password@host:port）
        if "@" in uri_without_protocol:
            # 分离认证信息和主机信息
            auth_part, host_part = uri_without_protocol.split("@", 1)
            # 这里可以保存认证信息，但 pymilvus 使用 token 认证
            # 所以暂时忽略 URI 中的认证信息
            host_port = host_part
        else:
            host_port = uri_without_protocol
        
        # 解析主机和端口
        if ":" in host_port:
            host, port = host_port.rsplit(":", 1)  # 使用 rsplit 处理 IPv6 地址
            try:
                port = int(port)
            except ValueError:
                raise ValueError(f"Invalid port in URI: {self.uri}")
        else:
            host = host_port
            port = 19530
        
        # 建立连接
        # 优先使用 host:port 方式连接（更可靠，适用于所有 Milvus 部署）
        # connections.connect() 不返回有意义的值（通常返回 None），
        # 它通过副作用建立连接并存储在 connections 对象中
        # 如果连接已存在，先断开再连接
        try:
            result = connections.connect(
                alias=alias,
                host=host,
                port=port,
                token=self.token
            )
            # result 通常是 None，连接信息存储在 connections 对象中
            # 可以通过 connections.get_connection_addr(alias) 验证连接
            # 返回格式: {'address': 'host:port', 'user': 'username'}
            # user 字段：用于用户名/密码认证时的用户名，使用 token 认证时为空字符串
        except Exception as e:
            # 如果连接已存在，先断开再连接
            try:
                connections.disconnect(alias)
                connections.connect(
                    alias=alias,
                    host=host,
                    port=port,
                    token=self.token
                )
            except Exception as reconnect_error:
                # 如果重新连接失败，抛出原始错误
                raise e from reconnect_error

    def create_collection(
        self,
        collection_name: str,
        fields: List[FieldSchema],
        description: str = "",
        enable_dynamic_field: bool = False
    ) -> Collection:
        """
        创建 collection
        
        Args:
            collection_name: collection 名称
            fields: 字段列表
            description: collection 描述
            enable_dynamic_field: 是否启用动态字段
            
        Returns:
            Collection 对象
            
        
        examples:
        # 定义字段 schema
        fields = [
            FieldSchema(name="pk", dtype=DataType.VARCHAR, is_primary=True, auto_id=False, max_length=1024),
            FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=256),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535, enable_analyzer=True),
            FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=vector_dim),
            FieldSchema(name="metadata", dtype=DataType.JSON)
        ]    
        
        """
        # 检查 collection 是否已存在
        if self.has_collection(collection_name):
            raise ValueError(f"Collection '{collection_name}' already exists")
        
        # 创建 schema
        
        
        schema = CollectionSchema(
            fields=fields,
            description=description,
            enable_dynamic_field=enable_dynamic_field
        )
        
        # 创建 collection
        collection = Collection(
            name=collection_name,
            schema=schema,
            using=self.alias
        )
        
        return collection

    def delete_collection(self, collection_name: str) -> bool:
        """
        删除 collection
        
        Args:
            collection_name: collection 名称
            
        Returns:
            是否删除成功
        """
        if not self.has_collection(collection_name):
            return False
        
        utility.drop_collection(collection_name)
        return True

    def has_collection(self, collection_name: str) -> bool:
        """
        检查 collection 是否存在
        
        Args:
            collection_name: collection 名称
            
        Returns:
            是否存在
        """
        return utility.has_collection(collection_name)

    def list_collections(self) -> List[str]:
        """
        列出所有 collection
        
        Returns:
            collection 名称列表
        """
        return utility.list_collections()

    def check_connection(self) -> bool:
        """
        检查连接是否有效
        
        Returns:
            连接是否有效
        """
        try:
            # 尝试获取连接地址，如果连接存在会返回地址
            # 返回格式: {'address': 'host:port', 'user': 'username'}
            # user 字段：用于用户名/密码认证时的用户名，使用 token 认证时为空字符串
            addr_info = connections.get_connection_addr(self.alias)
            return addr_info is not None and 'address' in addr_info
        except Exception:
            return False

    def get_connection_info(self) -> Dict[str, str]:
        """
        获取连接信息
        
        Returns:
            连接信息字典，包含 address 和 user 字段
            - address: 连接地址 (host:port)
            - user: 用户名（使用 token 认证时为空字符串）
        """
        try:
            return connections.get_connection_addr(self.alias)
        except Exception:
            return {}

    def get_collection(self, collection_name: str) -> Collection:
        """
        获取 collection 对象
        
        Args:
            collection_name: collection 名称
            
        Returns:
            Collection 对象
        """
        if not self.has_collection(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")
        
        return Collection(collection_name, using=self.alias)

    def create_index(
        self,
        collection_name: str,
        field_name: str,
        index_params: Dict[str, Any]
    ) -> None:
        """
        为指定字段创建索引
        
        Args:
            collection_name: collection 名称
            field_name: 字段名称
            index_params: 索引参数，例如:
                {
                    "index_type": "HNSW",
                    "metric_type": "COSINE",
                    "params": {"M": 32, "efConstruction": 300}
                }
        """
        collection = self.get_collection(collection_name)
        
        # 创建索引
        # pymilvus 的 create_index 方法接受字段名和索引参数字典
        collection.create_index(
            field_name=field_name,
            index_params=index_params
        )

    def insert_texts(
        self,
        collection_name: str,
        texts: List[str],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        doc_ids: Optional[List[str]] = None,
        auto_embed: bool = True
    ) -> List[str]:
        """
        插入文本块到指定 collection
        
        Args:
            collection_name: collection 名称
            texts: 文本列表
            ids: 主键列表（可选，如果不提供会自动生成）
            metadatas: 元数据列表（可选）
            doc_ids: 文档ID列表（可选）
            auto_embed: 是否自动进行向量化（需要提供 embedding_service）
            
        Returns:
            插入的主键列表
        """
        if not self.has_collection(collection_name):
            raise ValueError(f"Collection '{collection_name}' does not exist")
        
        collection = self.get_collection(collection_name)
        
        # 准备数据
        num_texts = len(texts)
        
        # 生成主键
        if ids is None:
            import uuid
            ids = [str(uuid.uuid4()) for _ in range(num_texts)]
        
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
        
        # 添加主键
        insert_data[pk_field_name] = ids
        
        # 添加 doc_id 字段（如果 schema 中有）
        if "doc_id" in schema_fields:
            if doc_ids is None:
                doc_ids = [None] * num_texts
            insert_data["doc_id"] = doc_ids
        
        # 添加 text 字段
        if "text" in schema_fields:
            insert_data["text"] = texts
        
        # 添加 dense_vector 字段（如果需要自动向量化）
        if "dense_vector" in schema_fields and auto_embed:
            if self.embedding_service is None:
                raise ValueError(
                    "embedding_service is required for auto embedding. "
                    "Either provide embedding_service or set auto_embed=False"
                )
            # 批量向量化（自动分批处理，每批最多 10 个）
            print(f"  正在向量化 {num_texts} 个文本块...")
            vectors = self.embedding_service.embeddings(texts, batch_size=10, show_progress=True)
            insert_data["dense_vector"] = vectors
        
        # 添加 metadata 字段
        if "metadata" in schema_fields:
            if metadatas is None:
                metadatas = [{}] * num_texts
            insert_data["metadata"] = metadatas
        
        # pymilvus 的 insert 方法需要按行组织的数据
        # 将按列组织的数据转换为按行组织：列表，每个元素是一个字典
        data_rows = []
        for i in range(num_texts):
            row = {}
            for field_name, field_values in insert_data.items():
                row[field_name] = field_values[i]
            data_rows.append(row)
        
        # 插入数据
        insert_result = collection.insert(data_rows)
        collection.flush()
        
        return ids

    def search(
        self,
        collection_name: str,
        query_vectors: List[List[float]],
        limit: int = 10,
        expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
        search_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        向量搜索
        
        Args:
            collection_name: collection 名称
            query_vectors: 查询向量列表
            limit: 返回结果数量
            expr: 过滤表达式（可选）
            output_fields: 返回的字段列表（可选）
            search_params: 搜索参数（可选）
            
        Returns:
            搜索结果列表
        """
        collection = self.get_collection(collection_name)
        
        # 加载 collection
        collection.load()
        
        # 默认搜索参数
        if search_params is None:
            search_params = {"metric_type": "COSINE", "params": {"ef": 64}}
        
        # 执行搜索
        results = collection.search(
            data=query_vectors,
            anns_field="dense_vector",
            param=search_params,
            limit=limit,
            expr=expr,
            output_fields=output_fields if output_fields else []
        )
        
        # 格式化结果
        formatted_results = []
        for hits in results:
            hit_list = []
            for hit in hits:
                hit_dict = {
                    "id": hit.id,
                    "distance": hit.distance,
                    "entity": hit.entity
                }
                hit_list.append(hit_dict)
            formatted_results.append(hit_list)
        
        return formatted_results

    def search_by_text(
        self,
        collection_name: str,
        query_text: str,
        limit: int = 10,
        expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
        search_params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        通过文本进行搜索（自动向量化）
        
        Args:
            collection_name: collection 名称
            query_text: 查询文本
            limit: 返回结果数量
            expr: 过滤表达式（可选）
            output_fields: 返回的字段列表（可选）
            search_params: 搜索参数（可选）
            
        Returns:
            搜索结果列表
        """
        if self.embedding_service is None:
            raise ValueError("embedding_service is required for text search")
        
        # 向量化查询文本
        query_vector = self.embedding_service.embedding(query_text)
        
        # 执行搜索
        return self.search(
            collection_name=collection_name,
            query_vectors=[query_vector],
            limit=limit,
            expr=expr,
            output_fields=output_fields,
            search_params=search_params
        )[0]  # 返回第一个查询的结果

