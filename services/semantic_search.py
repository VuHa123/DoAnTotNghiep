import os
from typing import List, Dict, Any, Tuple
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# --- CONFIG ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION = "psychology_chunks"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class SemanticIndexer:
    def __init__(self, 
                 qdrant_host: str = QDRANT_HOST, 
                 qdrant_port: int = QDRANT_PORT, 
                 collection: str = QDRANT_COLLECTION, 
                 embedding_model: str = EMBEDDING_MODEL):
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.collection = collection
        self.qdrant = None
        self.model = None
        
        try:
            self.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
            self.model = SentenceTransformer(embedding_model)
            # Tạo collection nếu chưa có
            if self.collection not in [c.name for c in self.qdrant.get_collections().collections]:
                self.qdrant.recreate_collection(
                    collection_name=self.collection,
                    vectors_config=models.VectorParams(size=384, distance=models.Distance.COSINE)
                )
            print(f"✅ Qdrant connected at {qdrant_host}:{qdrant_port}")
        except Exception as e:
            print(f"⚠️ Qdrant connection failed: {e}")
            print("Semantic search will be disabled")

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        if self.model is None:
            raise Exception("Model not loaded - Qdrant connection failed")
        return self.model.encode(texts, show_progress_bar=False)

    def upsert_chunks(self, chunk_records: List[Dict[str, Any]]):
        if self.qdrant is None:
            print("⚠️ Cannot upsert chunks - Qdrant not connected")
            return
            
        batch_size = 64
        for i in range(0, len(chunk_records), batch_size):
            batch = chunk_records[i:i+batch_size]
            texts = [c["chunk_text"] for c in batch]
            embeddings = self.embed_texts(texts)
            payloads = [
                {
                    "article_id": c["article_id"],
                    "title": c["title"],
                    "year": c["year"],
                    "source": c["source"],
                    "chunk_id": c["chunk_id"],
                    "url": c["url"],
                    "chunk_text": c["chunk_text"],  # Thêm chunk_text vào payload
                } for c in batch
            ]
            self.qdrant.upsert(
                collection_name=self.collection,
                points=[
                    models.PointStruct(
                        id=c["chunk_id"],
                        vector=embeddings[j],
                        payload=payloads[j]
                    ) for j, c in enumerate(batch)
                ]
            )

    def query(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if self.qdrant is None:
            print("⚠️ Cannot query - Qdrant not connected")
            return []
            
        query_emb = self.embed_texts([query])[0]
        search_result = self.qdrant.search(
            collection_name=self.collection,
            query_vector=query_emb,
            limit=top_k
        )
        # Trả về list dict chứa chunk_text + metadata
        results = []
        for hit in search_result:
            payload = hit.payload
            payload["score"] = hit.score
            results.append(payload)
        return results

    def query_with_similarity_threshold(self, query: str, top_k: int = 5, similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Query knowledge với ngưỡng độ tương đồng
        Chỉ trả về các knowledge chunks có độ tương đồng > threshold
        """
        if self.qdrant is None:
            print("⚠️ Cannot query - Qdrant not connected")
            return []
            
        # Lấy tất cả kết quả trước
        all_results = self.query(query, top_k=top_k * 2)  # Lấy nhiều hơn để lọc
        
        # Lọc theo ngưỡng độ tương đồng
        filtered_results = []
        for result in all_results:
            similarity_score = result.get("score", 0)
            if similarity_score >= similarity_threshold:
                filtered_results.append(result)
        
        # Giới hạn số lượng kết quả trả về
        return filtered_results[:top_k]

    def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Tính độ tương đồng cosine giữa hai đoạn text
        """
        if self.model is None:
            return 0.0
            
        try:
            # Embed cả hai text
            embeddings = self.embed_texts([text1, text2])
            
            # Tính cosine similarity
            similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
            
            return float(similarity)
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0

    def filter_knowledge_by_similarity(self, user_input: str, knowledge_chunks: List[Dict[str, Any]], 
                                     similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        Lọc knowledge chunks dựa trên độ tương đồng với user input
        Chỉ giữ lại những chunks có độ tương đồng > threshold
        """
        filtered_chunks = []
        
        for chunk in knowledge_chunks:
            chunk_text = chunk.get("chunk_text", "")
            if not chunk_text:
                continue
                
            # Tính độ tương đồng
            similarity = self.calculate_similarity(user_input, chunk_text)
            
            # Thêm similarity score vào chunk
            chunk["similarity_score"] = similarity
            
            # Chỉ giữ lại nếu độ tương đồng > threshold
            if similarity >= similarity_threshold:
                filtered_chunks.append(chunk)
                print(f"✅ Knowledge chunk được chọn - Độ tương đồng: {similarity:.3f}")
            else:
                print(f"❌ Knowledge chunk bị loại - Độ tương đồng: {similarity:.3f} < {similarity_threshold}")
        
        return filtered_chunks 