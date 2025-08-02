import os
from typing import List, Dict, Any, Tuple
from sentence_transformers import CrossEncoder
import numpy as np
import torch
from dataclasses import dataclass

@dataclass
class RerankedPassage:
    """Class để lưu trữ thông tin passage sau khi re-rank"""
    passage_id: str
    chunk_text: str
    original_score: float
    rerank_score: float
    metadata: Dict[str, Any]
    is_relevant: bool

class CrossEncoderReranker:
    """
    Re-ranker sử dụng Cross-Encoder để cải thiện độ chính xác của semantic search
    Cross-Encoder sẽ chấm điểm chính xác hơn cho các cặp (query, passage)
    """
    
    def __init__(self, 
                 model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 device: str = None,
                 relevance_threshold: float = 0.5):
        """
        Args:
            model_name: Tên model cross-encoder
            device: Device để chạy model (cuda/cpu)
            relevance_threshold: Ngưỡng điểm để xác định passage có liên quan
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.relevance_threshold = relevance_threshold
        self.model = None
        
        try:
            print(f"🔄 Loading Cross-Encoder model: {model_name}")
            self.model = CrossEncoder(model_name, device=self.device)
            print(f"✅ Cross-Encoder loaded successfully on {self.device}")
        except Exception as e:
            print(f"❌ Failed to load Cross-Encoder: {e}")
            print("⚠️ Re-ranking will be disabled")
    
    def rerank_passages(self, 
                       query: str, 
                       passages: List[Dict[str, Any]], 
                       top_k: int = 5) -> List[RerankedPassage]:
        """
        Re-rank các passages dựa trên query
        
        Args:
            query: Câu query của người dùng
            passages: List các passages từ vector search
            top_k: Số lượng passages trả về
            
        Returns:
            List các RerankedPassage đã được sắp xếp theo điểm số
        """
        if self.model is None:
            print("⚠️ Cross-Encoder not loaded, returning original passages")
            return self._create_fallback_results(passages, top_k)
        
        if not passages:
            return []
        
        try:
            # Tạo các cặp (query, passage) để chấm điểm
            query_passage_pairs = []
            for passage in passages:
                chunk_text = passage.get("chunk_text", "")
                if chunk_text:
                    query_passage_pairs.append([query, chunk_text])
            
            if not query_passage_pairs:
                return []
            
            # Chấm điểm bằng Cross-Encoder
            print(f"🔄 Re-ranking {len(query_passage_pairs)} passages...")
            scores = self.model.predict(query_passage_pairs)
            
            # Tạo RerankedPassage objects
            reranked_passages = []
            for i, passage in enumerate(passages):
                if i < len(scores):
                    rerank_score = float(scores[i])
                    original_score = passage.get("score", 0.0)
                    
                    reranked_passage = RerankedPassage(
                        passage_id=passage.get("chunk_id", f"passage_{i}"),
                        chunk_text=passage.get("chunk_text", ""),
                        original_score=original_score,
                        rerank_score=rerank_score,
                        metadata=passage,
                        is_relevant=rerank_score >= self.relevance_threshold
                    )
                    reranked_passages.append(reranked_passage)
            
            # Sắp xếp theo điểm re-rank (cao nhất trước)
            reranked_passages.sort(key=lambda x: x.rerank_score, reverse=True)
            
            # Lọc chỉ lấy passages có liên quan
            relevant_passages = [p for p in reranked_passages if p.is_relevant]
            
            # Nếu không có passages nào đạt threshold, lấy top_k passages có điểm cao nhất
            if not relevant_passages:
                print(f"⚠️ No passages meet relevance threshold {self.relevance_threshold}, using top {top_k} by score")
                final_results = reranked_passages[:top_k]
            else:
                # Giới hạn số lượng kết quả
                final_results = relevant_passages[:top_k]
            
            print(f"✅ Re-ranking completed:")
            print(f"   - Original passages: {len(passages)}")
            print(f"   - Relevant passages: {len(relevant_passages)}")
            print(f"   - Final results: {len(final_results)}")
            
            # Log điểm số của các passages được chọn
            for i, passage in enumerate(final_results):
                print(f"   {i+1}. Score: {passage.rerank_score:.3f} | Relevant: {passage.is_relevant}")
            
            return final_results
            
        except Exception as e:
            print(f"❌ Error during re-ranking: {e}")
            return self._create_fallback_results(passages, top_k)
    
    def _create_fallback_results(self, passages: List[Dict[str, Any]], top_k: int) -> List[RerankedPassage]:
        """Tạo kết quả fallback khi re-ranker không hoạt động"""
        fallback_results = []
        for i, passage in enumerate(passages[:top_k]):
            fallback_passage = RerankedPassage(
                passage_id=passage.get("chunk_id", f"passage_{i}"),
                chunk_text=passage.get("chunk_text", ""),
                original_score=passage.get("score", 0.0),
                rerank_score=passage.get("score", 0.0),  # Sử dụng original score
                metadata=passage,
                is_relevant=True  # Giả sử tất cả đều liên quan
            )
            fallback_results.append(fallback_passage)
        return fallback_results
    
    def get_relevance_score(self, query: str, passage: str) -> float:
        """
        Lấy điểm relevance cho một cặp (query, passage)
        
        Args:
            query: Câu query
            passage: Đoạn văn cần chấm điểm
            
        Returns:
            Điểm relevance (0-1)
        """
        if self.model is None:
            return 0.0
        
        try:
            score = self.model.predict([[query, passage]])
            return float(score[0])
        except Exception as e:
            print(f"❌ Error calculating relevance score: {e}")
            return 0.0
    
    def filter_by_relevance(self, 
                          query: str, 
                          passages: List[Dict[str, Any]], 
                          min_relevance: float = None) -> List[Dict[str, Any]]:
        """
        Lọc passages dựa trên điểm relevance
        
        Args:
            query: Câu query
            passages: List passages cần lọc
            min_relevance: Ngưỡng điểm tối thiểu (nếu None thì dùng self.relevance_threshold)
            
        Returns:
            List passages đã được lọc
        """
        if min_relevance is None:
            min_relevance = self.relevance_threshold
        
        if self.model is None:
            print("⚠️ Cross-Encoder not loaded, returning all passages")
            return passages
        
        filtered_passages = []
        
        for passage in passages:
            chunk_text = passage.get("chunk_text", "")
            if not chunk_text:
                continue
            
            relevance_score = self.get_relevance_score(query, chunk_text)
            
            if relevance_score >= min_relevance:
                passage["relevance_score"] = relevance_score
                filtered_passages.append(passage)
                print(f"✅ Passage relevant - Score: {relevance_score:.3f}")
            else:
                print(f"❌ Passage filtered out - Score: {relevance_score:.3f} < {min_relevance}")
        
        return filtered_passages

class HybridReranker:
    """
    Hybrid re-ranker kết hợp Cross-Encoder với các heuristic rules
    """
    
    def __init__(self, 
                 cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 relevance_threshold: float = 0.5,
                 length_penalty: float = 0.1):
        """
        Args:
            cross_encoder_model: Model cross-encoder
            relevance_threshold: Ngưỡng relevance
            length_penalty: Penalty cho passages quá dài/ngắn
        """
        self.cross_encoder = CrossEncoderReranker(
            model_name=cross_encoder_model,
            relevance_threshold=relevance_threshold
        )
        self.length_penalty = length_penalty
    
    def rerank_with_heuristics(self, 
                              query: str, 
                              passages: List[Dict[str, Any]], 
                              top_k: int = 5) -> List[RerankedPassage]:
        """
        Re-rank với heuristic rules
        
        Heuristics:
        1. Cross-Encoder score
        2. Length penalty (không quá dài/ngắn)
        3. Source quality (journal > book > website)
        4. Recency (năm xuất bản)
        """
        if not passages:
            return []
        
        # Bước 1: Re-rank bằng Cross-Encoder
        reranked_passages = self.cross_encoder.rerank_passages(query, passages, top_k * 2)
        
        # Bước 2: Áp dụng heuristic rules
        for passage in reranked_passages:
            # Length penalty
            text_length = len(passage.chunk_text)
            if text_length < 50 or text_length > 1000:
                passage.rerank_score *= (1 - self.length_penalty)
            
            # Source quality bonus
            source = passage.metadata.get("source", "").lower()
            if "journal" in source or "research" in source:
                passage.rerank_score *= 1.1
            elif "book" in source:
                passage.rerank_score *= 1.05
            
            # Recency bonus (ưu tiên tài liệu mới)
            year = passage.metadata.get("year", 0)
            if year >= 2020:
                passage.rerank_score *= 1.05
            elif year >= 2015:
                passage.rerank_score *= 1.02
        
        # Sắp xếp lại theo điểm số cuối cùng
        reranked_passages.sort(key=lambda x: x.rerank_score, reverse=True)
        
        # Trả về top_k kết quả
        return reranked_passages[:top_k] 