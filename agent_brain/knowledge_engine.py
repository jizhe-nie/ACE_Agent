import logging
import os

# 强制禁言 transformers 和 chromadb 的底层日志
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_OFFLINE"] = "1"
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("chromadb").setLevel(logging.ERROR)

import json
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions
from loguru import logger


class KnowledgeEngine:
    """优化版知识引擎：支持静默加载与可靠去重"""

    def __init__(self, db_path: str = "agent_brain/memory_vdb"):
        self.db_path = db_path
        self.manifest_path = Path(db_path) / "ingested_manifest.json"

        # 1. 尝试初始化 Embedding 模型
        try:
            self.embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        except Exception as e:
            logger.error(f"Embedding 模型加载失败 (离线模式需确保已下载过模型): {e}")
            # 自动降级：如果没有网络且没模型，可能会报错，建议用户先联网运行一次
            self.embed_fn = None

        # 2. 初始化 ChromaDB
        self.client = chromadb.PersistentClient(path=db_path)
        self.collection = self.client.get_or_create_collection(name="ace_knowledge", embedding_function=self.embed_fn)

        # 3. 加载已处理文件清单 (比从数据库查询快且准)
        self.ingested_files = self._load_manifest()

    def ingest_docs(self, docs_dir: str = "docs"):
        """扫描并入库，只处理新文件"""
        docs_path = Path(docs_dir)
        if not docs_path.exists():
            return

        new_files = []
        for file_path in docs_path.glob("*.*"):
            if file_path.suffix.lower() in [".txt", ".md", ".pdf"]:
                if file_path.name not in self.ingested_files:
                    new_files.append(file_path)

        if not new_files:
            return

        for file_path in new_files:
            logger.info(f"📚 正在索引新知识: {file_path.name}...")
            content = self._read_file(file_path)
            if content:
                chunks = [content[i : i + 800] for i in range(0, len(content), 600)]
                ids = [f"{file_path.name}_{i}" for i in range(len(chunks))]
                metadatas = [{"source": file_path.name} for _ in range(len(chunks))]

                self.collection.add(documents=chunks, ids=ids, metadatas=metadatas)
                self.ingested_files.add(file_path.name)

        self._save_manifest()
        logger.success(f"✅ 知识库更新完成，新增 {len(new_files)} 个文档。")

    def query(self, text: str, n_results: int = 3) -> str:
        """语义检索"""
        if not self.embed_fn:
            return ""
        try:
            results = self.collection.query(query_texts=[text], n_results=n_results)
            if not results or not results["documents"][0]:
                return ""
            context = "\n---\n".join(results["documents"][0])
            return f"\n【参考背景知识】:\n{context}\n"
        except Exception:
            return ""

    def _read_file(self, path: Path) -> str:
        try:
            if path.suffix.lower() == ".pdf":
                from pypdf import PdfReader

                return " ".join([p.extract_text() for p in PdfReader(path).pages])
            return path.read_text(encoding="utf-8")
        except Exception:
            return ""

    def _load_manifest(self) -> set:
        if self.manifest_path.exists():
            try:
                return set(json.loads(self.manifest_path.read_text()))
            except Exception:
                return set()
        return set()

    def _save_manifest(self):
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(list(self.ingested_files)))


if __name__ == "__main__":
    engine = KnowledgeEngine()
    engine.ingest_docs()
