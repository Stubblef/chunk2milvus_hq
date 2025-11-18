"""
chunk2milvus 模块

在使用前，确保环境变量已正确设置。
如果 MILVUS_URI 没有 http:// 前缀，会自动添加。
"""
from dotenv import load_dotenv
import os

# 加载环境变量（必须在导入 pymilvus 之前）
load_dotenv()

# 规范化 MILVUS_URI 环境变量（如果存在）
# pymilvus 在导入时会读取这个环境变量，要求格式为 http[s]://...
# 必须在导入任何会触发 pymilvus 导入的模块之前执行
milvus_uri = os.getenv("MILVUS_URI")
if milvus_uri and not milvus_uri.startswith("http://") and not milvus_uri.startswith("https://"):
    # 自动添加 http:// 前缀
    os.environ["MILVUS_URI"] = f"http://{milvus_uri}"

# 现在可以安全地导入会触发 pymilvus 导入的模块
from .milvus_client import MilvusClient
from .embedding import EmbeddingService

__all__ = ["MilvusClient", "EmbeddingService"]
