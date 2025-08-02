import os
from typing import List, Dict, Any, Tuple
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from .reranker import CrossEncoderReranker, HybridReranker, RerankedPassage

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
                 embedding_model: str = EMBEDDING_MODEL,
                 use_reranker: bool = True,
                 reranker_type: str = "cross_encoder"):
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.collection = collection
        self.qdrant = None
        self.model = None
        self.use_reranker = use_reranker
        self.reranker = None
        
        try:
            self.qdrant = QdrantClient(host=qdrant_host, port=qdrant_port)
            self.model = SentenceTransformer(embedding_model)
            
            # Khởi tạo re-ranker nếu được yêu cầu
            if self.use_reranker:
                if reranker_type == "hybrid":
                    self.reranker = HybridReranker()
                else:
                    self.reranker = CrossEncoderReranker()
                print(f"✅ Re-ranker initialized: {reranker_type}")
            
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
        """
        Query cơ bản từ vector database
        KHÔNG áp dụng re-ranker để tránh conflict với query_with_reranker
        """
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
        
        # Lọc theo ngưỡng độ tương đồng (sử dụng rerank_score nếu có)
        filtered_results = []
        for result in all_results:
            # Ưu tiên rerank_score nếu có, nếu không thì dùng score gốc
            similarity_score = result.get("rerank_score", result.get("score", 0))
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
        
        # Sử dụng re-ranker nếu có để tính điểm chính xác hơn
        if self.use_reranker and self.reranker:
            print("🔄 Using re-ranker for similarity filtering...")
            
            # Tạo danh sách passages để re-rank
            passages_for_rerank = []
            for chunk in knowledge_chunks:
                chunk_text = chunk.get("chunk_text", "")
                if chunk_text:
                    passages_for_rerank.append(chunk)
            
            if passages_for_rerank:
                # Sử dụng re-ranker để lọc
                if isinstance(self.reranker, HybridReranker):
                    reranked_results = self.reranker.rerank_with_heuristics(
                        user_input, passages_for_rerank, len(passages_for_rerank)
                    )
                else:
                    reranked_results = self.reranker.rerank_passages(
                        user_input, passages_for_rerank, len(passages_for_rerank)
                    )
                
                # Lọc theo ngưỡng và chuyển về dict format
                for reranked_passage in reranked_results:
                    if reranked_passage.rerank_score >= similarity_threshold:
                        result_dict = reranked_passage.metadata.copy()
                        result_dict["similarity_score"] = reranked_passage.rerank_score
                        result_dict["is_relevant"] = reranked_passage.is_relevant
                        filtered_chunks.append(result_dict)
                        print(f"✅ Knowledge chunk được chọn - Re-rank score: {reranked_passage.rerank_score:.3f}")
                    else:
                        print(f"❌ Knowledge chunk bị loại - Re-rank score: {reranked_passage.rerank_score:.3f} < {similarity_threshold}")
        else:
            # Fallback về phương pháp cũ
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

    def query_with_reranker(self, query: str, top_k: int = 5, relevance_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """
        Query với re-ranker để có kết quả chính xác hơn
        Trả về rỗng nếu không tìm được thông tin phù hợp
        
        Args:
            query: Câu query
            top_k: Số lượng kết quả trả về
            relevance_threshold: Ngưỡng relevance cho re-ranker
            
        Returns:
            List các passages đã được re-rank và lọc, hoặc rỗng nếu không phù hợp
        """
        if self.qdrant is None:
            print("⚠️ Cannot query - Qdrant not connected")
            return []
        
        # Lấy kết quả ban đầu từ vector search (KHÔNG áp dụng re-ranker)
        initial_top_k = top_k * 3  # Lấy nhiều hơn để re-rank
        
        query_emb = self.embed_texts([query])[0]
        search_result = self.qdrant.search(
            collection_name=self.collection,
            query_vector=query_emb,
            limit=initial_top_k
        )
        
        # Trả về list dict chứa chunk_text + metadata
        initial_results = []
        for hit in search_result:
            payload = hit.payload
            payload["score"] = hit.score
            initial_results.append(payload)
        
        if not initial_results:
            print("⚠️ No initial results found from vector search")
            return []
        
        # Áp dụng re-ranker nếu có
        if self.use_reranker and self.reranker:
            print(f"🔄 Applying re-ranker to {len(initial_results)} initial results...")
            
            if isinstance(self.reranker, HybridReranker):
                reranked_results = self.reranker.rerank_with_heuristics(query, initial_results, top_k)
            else:
                reranked_results = self.reranker.rerank_passages(query, initial_results, top_k)
            
            # Lọc theo relevance threshold - CHỈ trả về kết quả thực sự liên quan
            filtered_results = []
            for reranked_passage in reranked_results:
                if reranked_passage.rerank_score >= relevance_threshold:
                    result_dict = reranked_passage.metadata.copy()
                    result_dict["rerank_score"] = reranked_passage.rerank_score
                    result_dict["is_relevant"] = reranked_passage.is_relevant
                    filtered_results.append(result_dict)
            
            print(f"✅ Re-ranker filtered to {len(filtered_results)} relevant results")
            
            # Nếu không có kết quả nào đạt threshold, trả về rỗng
            if not filtered_results:
                print("⚠️ No results meet relevance threshold, returning empty knowledge")
                return []
            
            return filtered_results
        else:
            # Fallback: kiểm tra điểm vector similarity
            print("⚠️ Re-ranker not available, checking vector similarity")
            high_quality_results = []
            for result in initial_results[:top_k]:
                score = result.get("score", 0)
                if score >= 0.7:  # Chỉ lấy kết quả có điểm cao
                    high_quality_results.append(result)
            
            if not high_quality_results:
                print("⚠️ No high-quality results found, returning empty knowledge")
                return []
            
            return high_quality_results 