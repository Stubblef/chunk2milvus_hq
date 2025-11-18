from typing import List, Optional
import os
from openai import OpenAI


class EmbeddingService:
    """向量化服务"""

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        dimension: Optional[int] = None
    ):
        """
        初始化向量化服务
        
        Args:
            api_key: OpenAI API Key，如果为 None 则从环境变量 EMBEDDING_KEY 读取
            base_url: API 基础URL，如果为 None 则从环境变量 EMBEDDING_URL 读取，默认为 "https://api.openai.com/v1"
            model: 使用的模型名称，如果为 None 则从环境变量 EMBEDDING_MODEL 读取，默认为 "text-embedding-3-large"
            dimension: 向量维度，如果为 None 则从环境变量 EMBEDDING_DIMENSION 读取，默认为 3072
        """
        # 从环境变量读取配置
        self.api_key = api_key or os.getenv("EMBEDDING_KEY")
        self.base_url = base_url or os.getenv("EMBEDDING_URL", "https://api.openai.com/v1")
        self.model = model or os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
        self.dimension = dimension or int(os.getenv("EMBEDDING_DIMENSION", "3072"))
        
        if not self.api_key:
            raise ValueError(
                "API key is required for embedding service. "
                "Please provide api_key parameter or set EMBEDDING_KEY environment variable."
            )

        self.client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def embedding(self, text: str) -> List[float]:
        """获取文本的向量表示"""
        response = self.client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dimension
        )
        embedding = response.data[0].embedding
        if len(embedding) != self.dimension:
            raise ValueError(
                f"Expected embedding dimension {self.dimension}, "
                f"got {len(embedding)}"
            )
        return embedding

    def embeddings(self, texts: List[str]) -> List[List[float]]:
        """批量获取文本的向量表示"""
        response = self.client.embeddings.create(
            model=self.model,
            input=texts,
            dimensions=self.dimension
        )

        embeddings = [item.embedding for item in response.data]

        # 验证向量维度
        for i, embedding in enumerate(embeddings):
            if len(embedding) != self.dimension:
                raise ValueError(
                    f"Expected embedding dimension {self.dimension} for text {i}, "
                    f"got {len(embedding)}"
                )

        return embeddings