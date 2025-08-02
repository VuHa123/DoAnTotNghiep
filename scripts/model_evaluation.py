#!/usr/bin/env python3
"""
Script đánh giá và so sánh model chatbot tâm lý
Tác giả: AI Assistant
Ngày tạo: 2024
"""

import sys
import os
import json
import pandas as pd
import numpy as np
import requests
import time
import logging
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
from pathlib import Path
import re

# Import torch nếu có
try:
    import torch
except ImportError:
    torch = None

# Thêm đường dẫn để import từ các module khác
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'services', 'chatbot'))

# Import các hàm cần thiết
try:
    from services.chatbot.response_generator import extract_main_response, final_cleanup
except ImportError:
    print("Warning: Không thể import response_generator, sẽ sử dụng fallback functions")
    
    def extract_main_response(text: str) -> str:
        """Fallback function cho extract_main_response"""
        if not isinstance(text, str):
            return ""
        # Loại bỏ các ký tự đặc biệt và làm sạch text
        text = re.sub(r'<\|[^|]*\|>', '', text)
        text = re.sub(r'[◆◇■□●○▲△▽▼♠♣♥♦★☆♤♧♡♢♪♫♬♩♭♮♯]', '', text)
        return text.strip()
    
    def final_cleanup(text: str) -> str:
        """Fallback function cho final_cleanup"""
        if not isinstance(text, str):
            return ""
        return text.strip()

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, config: Dict = None):
        """
        Khởi tạo ModelEvaluator
        
        Args:
            config: Cấu hình cho việc đánh giá
        """
        self.config = config or {
            "your_bot_url": "http://localhost:8000/chat",
            "model_server_url": "http://localhost:8001/model/generate/",
            "openai_api_key": os.getenv("OPENAI_API_KEY"),
            "max_tokens": 1024,
            "temperature": 0.7,
            "timeout": 30
        }
        
        # Khởi tạo các model cần test
        self.models = {
            "your_bot": self.call_your_bot,
            "gpt4": self.call_gpt4,
            "deepseek": self.call_deepseek,
            "llama": self.call_llama
        }
        
        # Cache cho các model để tránh load lại
        self.model_cache = {}
        
    def preprocess_data(self, data_source: str) -> Dict[str, List[str]]:
        """
        Phần 0: Tiền xử lý dữ liệu
        Gom nhóm dữ liệu theo câu hỏi
        
        Args:
            data_source: Đường dẫn đến file dữ liệu
            
        Returns:
            Dict với key là câu hỏi, value là list các reference answers
        """
        logger.info(f"Bắt đầu tiền xử lý dữ liệu từ: {data_source}")
        
        grouped_data = defaultdict(list)
        
        if data_source.endswith('.csv'):
            # Xử lý file CSV
            try:
                df = pd.read_csv(data_source)
                logger.info(f"Đọc được {len(df)} dòng từ CSV")
                
                # Giả sử cấu trúc: Context_translated, Response_translated
                if 'Context_translated' in df.columns and 'Response_translated' in df.columns:
                    for _, row in df.iterrows():
                        question = str(row['Context_translated']).strip()
                        answer = str(row['Response_translated']).strip()
                        
                        if question and answer and len(question) > 10 and len(answer) > 10:
                            grouped_data[question].append(answer)
                            
            except Exception as e:
                logger.error(f"Lỗi khi đọc CSV: {e}")
                
        elif data_source.endswith('.jsonl'):
            # Xử lý file JSONL
            try:
                with open(data_source, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f, 1):
                        try:
                            data = json.loads(line.strip())
                            
                            # Xử lý format JSONL với instruction, input, output
                            if 'instruction' in data and 'input' in data and 'output' in data:
                                # Trích xuất câu hỏi từ input
                                input_text = data['input']
                                # Tìm câu hỏi cuối cùng từ người dùng
                                user_pattern = r'Người dùng:\s*(.*?)(?=\nAn:|$)'
                                matches = re.findall(user_pattern, input_text, re.DOTALL)
                                
                                if matches:
                                    question = matches[-1].strip()  # Lấy câu hỏi cuối cùng
                                    answer = data['output'].strip()
                                    
                                    if question and answer and len(question) > 10 and len(answer) > 10:
                                        grouped_data[question].append(answer)
                                        
                        except json.JSONDecodeError as e:
                            logger.warning(f"Lỗi JSON ở dòng {line_num}: {e}")
                            continue
                            
            except Exception as e:
                logger.error(f"Lỗi khi đọc JSONL: {e}")
        
        # Lọc dữ liệu: chỉ giữ những câu hỏi có ít nhất 2 reference answers
        filtered_data = {q: refs for q, refs in grouped_data.items() if len(refs) >= 2}
        
        logger.info(f"Hoàn thành tiền xử lý. Có {len(filtered_data)} câu hỏi với {sum(len(refs) for refs in filtered_data.values())} reference answers")
        
        # Lưu dữ liệu đã xử lý
        with open('processed_test_data.json', 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)
            
        return filtered_data
    
    def call_your_bot(self, question: str) -> str:
        """
        Gọi bot của bạn thông qua API
        """
        try:
            payload = {
                "user_input": question,
                "history": [],
                "session_id": "evaluation"
            }
            
            response = requests.post(
                self.config["your_bot_url"],
                json=payload,
                timeout=self.config["timeout"]
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("bot_response", "")
            else:
                logger.error(f"Lỗi API your_bot: {response.status_code}")
                return ""
                
        except Exception as e:
            logger.error(f"Lỗi khi gọi your_bot: {e}")
            return ""
    
    def call_model_server(self, question: str) -> str:
        """
        Gọi model server trực tiếp
        """
        try:
            prompt = f"Bạn là một chatbot hỗ trợ tâm lý thân thiện và cảm thông. Hãy trả lời người dùng một cách nhẹ nhàng, hỗ trợ và chuyên nghiệp. Bắt đầu câu trả lời bằng 'Chào bạn' và chỉ đưa ra nội dung chính, không cần giải thích thêm hay kết luận.\n\nUser: {question}\n\nAssistant:"
            
            payload = {
                "prompt": prompt,
                "max_new_tokens": self.config["max_tokens"]
            }
            
            response = requests.post(
                self.config["model_server_url"],
                json=payload,
                timeout=self.config["timeout"]
            )
            
            if response.status_code == 200:
                raw_response = response.text
                cleaned_response = final_cleanup(raw_response)
                final_response = extract_main_response(cleaned_response)
                return final_response
            else:
                logger.error(f"Lỗi model server: {response.status_code}")
                return ""
                
        except Exception as e:
            logger.error(f"Lỗi khi gọi model server: {e}")
            return ""
    
    def call_gpt4(self, question: str) -> str:
        """
        Gọi GPT-4 thông qua OpenAI API
        """
        if not self.config.get("openai_api_key"):
            logger.warning("Không có OpenAI API key, bỏ qua GPT-4")
            return ""
            
        try:
            import openai
            openai.api_key = self.config["openai_api_key"]
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "Bạn là một chatbot hỗ trợ tâm lý thân thiện và cảm thông. Hãy trả lời người dùng một cách nhẹ nhàng, hỗ trợ và chuyên nghiệp."},
                    {"role": "user", "content": question}
                ],
                max_tokens=self.config["max_tokens"],
                temperature=self.config["temperature"]
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Lỗi khi gọi GPT-4: {e}")
            return ""
    
    def call_deepseek(self, question: str) -> str:
        """
        Gọi DeepSeek thông qua Hugging Face
        """
        try:
            from transformers import pipeline
            
            if "deepseek" not in self.model_cache:
                logger.info("Loading DeepSeek model...")
                self.model_cache["deepseek"] = pipeline(
                    "text-generation",
                    model="deepseek-ai/deepseek-coder-6.7b-instruct",
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
            
            model = self.model_cache["deepseek"]
            
            prompt = f"<|im_start|>system\nBạn là một chatbot hỗ trợ tâm lý thân thiện và cảm thông.<|im_end|>\n<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"
            
            output = model(
                prompt,
                max_length=len(prompt.split()) + 100,
                num_return_sequences=1,
                temperature=self.config["temperature"],
                do_sample=True
            )
            
            response = output[0]["generated_text"]
            # Trích xuất phần response từ assistant
            assistant_pattern = r'<\|im_start\|>assistant\n(.*?)(?=<\|im_end\|>|$)'
            match = re.search(assistant_pattern, response, re.DOTALL)
            
            if match:
                return match.group(1).strip()
            else:
                return response.split("<|im_start|>assistant\n")[-1].strip()
                
        except Exception as e:
            logger.error(f"Lỗi khi gọi DeepSeek: {e}")
            return ""
    
    def call_llama(self, question: str) -> str:
        """
        Gọi LLaMA thông qua Hugging Face
        """
        try:
            from transformers import pipeline
            
            if "llama" not in self.model_cache:
                logger.info("Loading LLaMA model...")
                self.model_cache["llama"] = pipeline(
                    "text-generation",
                    model="meta-llama/Llama-2-7b-chat-hf",
                    device="cuda" if torch.cuda.is_available() else "cpu"
                )
            
            model = self.model_cache["llama"]
            
            prompt = f"[INST] Bạn là một chatbot hỗ trợ tâm lý thân thiện và cảm thông. Hãy trả lời người dùng một cách nhẹ nhàng, hỗ trợ và chuyên nghiệp. [/INST]\n\n{question}"
            
            output = model(
                prompt,
                max_length=len(prompt.split()) + 100,
                num_return_sequences=1,
                temperature=self.config["temperature"],
                do_sample=True
            )
            
            response = output[0]["generated_text"]
            # Trích xuất phần response sau prompt
            return response[len(prompt):].strip()
            
        except Exception as e:
            logger.error(f"Lỗi khi gọi LLaMA: {e}")
            return ""
    
    def evaluate_response(self, hypothesis: str, references: List[str]) -> Dict[str, float]:
        """
        Phần 2: Đánh giá response
        Tính các metric: BLEU, ROUGE, Cosine Similarity
        """
        if not hypothesis or not references:
            return {
                "bleu": 0.0,
                "rouge1": 0.0,
                "rougeL": 0.0,
                "cosine": 0.0
            }
        
        try:
            # BLEU Score
            from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
            import nltk
            
            # Download NLTK data nếu cần
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
            
            # Tokenize
            hypothesis_tokens = nltk.word_tokenize(hypothesis.lower())
            reference_tokens = [nltk.word_tokenize(ref.lower()) for ref in references]
            
            # Tính BLEU
            smoothing = SmoothingFunction().method1
            bleu_score = sentence_bleu(reference_tokens, hypothesis_tokens, smoothing_function=smoothing)
            
            # ROUGE Score (simplified)
            def calculate_rouge(hypothesis, references):
                hyp_words = set(hypothesis.lower().split())
                ref_words = set()
                for ref in references:
                    ref_words.update(ref.lower().split())
                
                if not ref_words:
                    return 0.0, 0.0
                
                # ROUGE-1
                intersection = hyp_words.intersection(ref_words)
                rouge1 = len(intersection) / len(hyp_words) if hyp_words else 0.0
                
                # ROUGE-L (simplified)
                rougeL = len(intersection) / len(ref_words) if ref_words else 0.0
                
                return rouge1, rougeL
            
            rouge1, rougeL = calculate_rouge(hypothesis, references)
            
            # Cosine Similarity
            try:
                from sentence_transformers import SentenceTransformer, util
                
                if "embedder" not in self.model_cache:
                    self.model_cache["embedder"] = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
                
                embedder = self.model_cache["embedder"]
                
                # Encode hypothesis và references
                embeddings_ref = embedder.encode(references)
                embedding_hyp = embedder.encode([hypothesis])
                
                # Tính cosine similarity với tất cả references
                cosine_scores = []
                for emb_ref in embeddings_ref:
                    cos_sim = util.cos_sim(embedding_hyp, emb_ref.reshape(1, -1)).item()
                    cosine_scores.append(cos_sim)
                
                cosine_sim = sum(cosine_scores) / len(cosine_scores)
                
            except Exception as e:
                logger.warning(f"Không thể tính cosine similarity: {e}")
                cosine_sim = 0.0
            
            return {
                "bleu": bleu_score,
                "rouge1": rouge1,
                "rougeL": rougeL,
                "cosine": cosine_sim
            }
            
        except Exception as e:
            logger.error(f"Lỗi khi đánh giá response: {e}")
            return {
                "bleu": 0.0,
                "rouge1": 0.0,
                "rougeL": 0.0,
                "cosine": 0.0
            }
    
    def run_evaluation(self, data_source: str, max_samples: int = 100) -> pd.DataFrame:
        """
        Phần 3: Chạy toàn bộ tập dữ liệu
        """
        logger.info(f"Bắt đầu đánh giá với tối đa {max_samples} mẫu")
        
        # Tiền xử lý dữ liệu
        grouped_data = self.preprocess_data(data_source)
        
        # Giới hạn số lượng mẫu
        if max_samples and len(grouped_data) > max_samples:
            import random
            questions = random.sample(list(grouped_data.keys()), max_samples)
            grouped_data = {q: grouped_data[q] for q in questions}
        
        results = []
        
        for i, (question, references) in enumerate(grouped_data.items(), 1):
            logger.info(f"Đang xử lý mẫu {i}/{len(grouped_data)}")
            
            for model_name, model_func in self.models.items():
                try:
                    # Sinh phản hồi
                    logger.info(f"  Gọi model {model_name}...")
                    hypothesis = model_func(question)
                    
                    if not hypothesis:
                        logger.warning(f"  Model {model_name} trả về response rỗng")
                        continue
                    
                    # Đánh giá
                    metrics = self.evaluate_response(hypothesis, references)
                    
                    results.append({
                        "question": question,
                        "model": model_name,
                        "hypothesis": hypothesis,
                        "references": " | ".join(references),
                        "num_references": len(references),
                        "bleu": metrics["bleu"],
                        "rouge1": metrics["rouge1"],
                        "rougeL": metrics["rougeL"],
                        "cosine": metrics["cosine"]
                    })
                    
                    # Nghỉ giữa các request để tránh rate limit
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Lỗi khi xử lý model {model_name}: {e}")
                    continue
        
        # Lưu kết quả
        df_results = pd.DataFrame(results)
        df_results.to_csv("evaluation_results.csv", index=False, encoding="utf-8")
        
        logger.info(f"Hoàn thành đánh giá. Kết quả được lưu vào evaluation_results.csv")
        
        return df_results
    
    def analyze_results(self, df_results: pd.DataFrame) -> Dict:
        """
        Phần 4: So sánh & Thống kê
        """
        logger.info("Bắt đầu phân tích kết quả")
        
        if df_results.empty:
            logger.warning("Không có dữ liệu để phân tích")
            return {}
        
        # Tính trung bình metric theo model
        avg_metrics = df_results.groupby("model")[["bleu", "rouge1", "rougeL", "cosine"]].mean().reset_index()
        
        # Tính độ lệch chuẩn
        std_metrics = df_results.groupby("model")[["bleu", "rouge1", "rougeL", "cosine"]].std().reset_index()
        
        # Xếp hạng theo từng metric
        for metric in ["bleu", "rouge1", "rougeL", "cosine"]:
            avg_metrics[f"{metric}_rank"] = avg_metrics[metric].rank(ascending=False)
        
        # Xếp hạng tổng hợp (trung bình của tất cả metrics)
        avg_metrics["overall_score"] = avg_metrics[["bleu", "rouge1", "rougeL", "cosine"]].mean(axis=1)
        avg_metrics["overall_rank"] = avg_metrics["overall_score"].rank(ascending=False)
        
        # Thống kê chi tiết
        stats = {
            "total_samples": len(df_results),
            "models_evaluated": df_results["model"].nunique(),
            "avg_metrics": avg_metrics.to_dict("records"),
            "std_metrics": std_metrics.to_dict("records"),
            "best_model_overall": avg_metrics.loc[avg_metrics["overall_rank"].idxmin(), "model"],
            "best_model_cosine": avg_metrics.loc[avg_metrics["cosine_rank"].idxmin(), "model"],
            "best_model_bleu": avg_metrics.loc[avg_metrics["bleu_rank"].idxmin(), "model"]
        }
        
        # Lưu kết quả phân tích
        avg_metrics.to_csv("average_metrics.csv", index=False, encoding="utf-8")
        
        # In kết quả
        print("\n" + "="*80)
        print("KẾT QUẢ ĐÁNH GIÁ MODEL")
        print("="*80)
        print(f"Tổng số mẫu đánh giá: {stats['total_samples']}")
        print(f"Số model được đánh giá: {stats['models_evaluated']}")
        print("\nBảng xếp hạng (theo điểm tổng hợp):")
        print("-" * 80)
        
        for _, row in avg_metrics.sort_values("overall_rank").iterrows():
            print(f"{row['model']:15} | Overall: {row['overall_score']:.4f} | BLEU: {row['bleu']:.4f} | ROUGE-1: {row['rouge1']:.4f} | ROUGE-L: {row['rougeL']:.4f} | Cosine: {row['cosine']:.4f}")
        
        print(f"\nModel tốt nhất (tổng hợp): {stats['best_model_overall']}")
        print(f"Model tốt nhất (cosine): {stats['best_model_cosine']}")
        print(f"Model tốt nhất (BLEU): {stats['best_model_bleu']}")
        
        return stats

def main():
    """
    Hàm main để chạy toàn bộ quá trình đánh giá
    """
    import argparse
    
    parser = argparse.ArgumentParser(description="Đánh giá và so sánh model chatbot")
    parser.add_argument("--data_source", type=str, default="Dataset/llama_instruction_data.jsonl",
                       help="Đường dẫn đến file dữ liệu test")
    parser.add_argument("--max_samples", type=int, default=50,
                       help="Số lượng mẫu tối đa để đánh giá")
    parser.add_argument("--models", type=str, nargs="+", 
                       default=["your_bot", "gpt4"],
                       help="Danh sách model cần đánh giá")
    
    args = parser.parse_args()
    
    # Khởi tạo evaluator
    evaluator = ModelEvaluator()
    
    # Lọc model theo argument
    if args.models:
        evaluator.models = {k: v for k, v in evaluator.models.items() if k in args.models}
    
    try:
        # Chạy đánh giá
        df_results = evaluator.run_evaluation(args.data_source, args.max_samples)
        
        # Phân tích kết quả
        stats = evaluator.analyze_results(df_results)
        
        print("\nĐánh giá hoàn thành! Kết quả được lưu vào:")
        print("- evaluation_results.csv: Kết quả chi tiết từng mẫu")
        print("- average_metrics.csv: Thống kê trung bình theo model")
        print("- processed_test_data.json: Dữ liệu test đã xử lý")
        print("- model_evaluation.log: Log chi tiết quá trình đánh giá")
        
    except Exception as e:
        logger.error(f"Lỗi trong quá trình đánh giá: {e}")
        raise

if __name__ == "__main__":
    main() 