# 🧠 Mental Health Chatbot

**Mental Health Chatbot** là hệ thống hỗ trợ tâm lý ứng dụng AI, sử dụng mô hình ngôn ngữ lớn (LLM) và các tầng phân tích rủi ro để đảm bảo an toàn, cá nhân hóa phản hồi và xử lý khẩn cấp khi cần thiết.

---

## 🚦 Kiến trúc & Flow tổng quan

Chatbot sử dụng **Gating Network (QuickCheck)** để phân tầng rủi ro của từng tin nhắn, từ đó quyết định luồng xử lý phù hợp:

```mermaid
flowchart TD
    A["User Message"] --> B["Gating Network: QuickCheck (20ms)"]
    B -->|"Low Risk\n🟢 BÌNH THƯỜNG\n(Confidence > 0.9)"| C["Expert 1: Chatbot Gemini (fast)"]
    B -->|"Medium Risk\n🟡 CÓ VẤN ĐỀ\n(0.5–0.9)"| D["Expert 1 + Expert 2, 3"]
    B -->|"High Risk\n🔴 KHẨN CẤP\n(< 0.5 hoặc suicidal)"| E["All Experts + Emergency Logic"]
    C --> F["Phản hồi"]
    D --> G["Mental State\nSentiment Intensity"]
    G --> H["Sinh prompt → Gemini"]
    H --> I["Phản hồi"]
    E --> J["Notify Support\nSpecial Response"]
    J --> K["Alert or escalate"]

    %% Chú thích chi tiết từng nhánh
    B -.-> B1["Gating Network: Phân tích nhanh mức độ rủi ro của message dựa trên model LogisticRegression hoặc tương tự.\nKết quả chia 3 mức: Bình thường, Có vấn đề, Khẩn cấp."]
    C -.-> C1["Expert 1: Sử dụng Gemini (Google LLM) trả lời nhanh các câu hỏi thông thường, không nhạy cảm."]
    D -.-> D1["Expert 2, 3: Có thể là các mô hình chuyên biệt về tâm lý, cảm xúc, hoặc kiểm tra sâu hơn.\nKết hợp kết quả với Expert 1 để sinh phản hồi phù hợp."]
    G -.-> G1["Mental State: Phân tích trạng thái tâm lý.\nSentiment Intensity: Đánh giá cường độ cảm xúc.\nKết quả dùng để xây dựng prompt phù hợp cho LLM."]
    E -.-> E1["All Experts: Kết hợp mọi mô hình kiểm tra,\nEmergency Logic: Kích hoạt quy trình hỗ trợ khẩn cấp (gọi hotline, cảnh báo nhân viên, v.v.)"]
    J -.-> J1["Notify Support: Gửi thông báo cho đội ngũ hỗ trợ hoặc chuyên gia.\nSpecial Response: Sinh phản hồi đặc biệt trấn an, hướng dẫn user giữ an toàn."]
    K -.-> K1["Alert or escalate: Có thể gọi hotline, gửi cảnh báo, hoặc chuyển tiếp cho chuyên gia can thiệp."]

    style C fill:#bff,stroke:#333,stroke-width:2px
    style D fill:#fffbcc,stroke:#333,stroke-width:2px
    style E fill:#faa,stroke:#333,stroke-width:2px
    style F fill:#bff
    style I fill:#fffbcc
    style J fill:#faa
    style K fill:#faa
```

### Giải thích các tầng xử lý:
- **Gating Network (QuickCheck):** Phân tích nhanh mức độ rủi ro của message, chia 3 mức: Bình thường, Có vấn đề, Khẩn cấp.
- **Low Risk:** Chỉ dùng Expert 1 (Gemini LLM) trả lời nhanh, phù hợp với các câu hỏi thông thường.
- **Medium Risk:** Kết hợp nhiều expert (ví dụ: mô hình phân tích tâm lý, cảm xúc) để kiểm tra sâu hơn, sinh prompt cá nhân hóa trước khi gửi Gemini.
- **High Risk:** Kích hoạt tất cả expert và logic khẩn cấp: cảnh báo, gọi hotline, gửi thông báo cho nhân viên hỗ trợ, sinh phản hồi đặc biệt trấn an user.

---

## 🏗️ Kiến trúc Model Server

Hệ thống sử dụng **Model Inference Server** riêng biệt để serve fine-tuned LLaMA model:

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │  API Gateway     │    │ Model Server    │
│   (Gradio)      │◄──►│  (FastAPI)       │◄──►│  (Fine-tuned    │
│                 │    │                  │    │   LLaMA)        │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                              │
                              ▼
                       ┌──────────────────┐
                       │  Gemini API      │
                       │  (Fallback)      │
                       └──────────────────┘
```

### Components:

#### 1. Model Server (`model_server.py`)
- **Port**: 8001
- **Chức năng**: Serve fine-tuned LLaMA model
- **Endpoints**:
  - `GET /health` - Health check
  - `POST /generate` - Generate response
  - `GET /model-info` - Model information

#### 2. API Gateway (`api_gateway/main.py`)
- **Port**: 8000
- **Chức năng**: Main API với fallback logic
- **Endpoints**:
  - `POST /api/v1/chat` - Main chat endpoint
  - `GET /health` - Health check
  - `GET /api-stats` - API statistics

#### 3. Fallback Logic
```python
# Auto mode (default)
if llama_available:
    try_llama()
    if llama_failed:
        fallback_to_gemini()
else:
    use_gemini()

# Manual mode
if prefer_model == "llama":
    use_llama()
elif prefer_model == "gemini":
    use_gemini()
```

---

## 📦 Cấu trúc thư mục đã tối ưu hóa

```bash
DoAnTotNghiep/
├── api_gateway/             # FastAPI backend (gộp từ server.py)
│   ├── main.py             # Main API server
│   └── chatbot_api.py      # Chatbot-specific routes
├── frontend/                # Enhanced Gradio interface
│   └── app.py              # Main frontend application
├── services/                # Core services (giữ nguyên)
│   ├── chatbot/            # Chatbot logic
│   ├── emergency_handler/   # Emergency handling
│   ├── gating_router/      # Risk assessment
│   ├── mental_state_classifier/ # Mental state analysis
│   ├── setiment_analysis/  # Sentiment analysis
│   ├── context_tracking/   # Conversation context
│   └── summarization/      # Text summarization
├── models/                  # Model definitions
│   ├── chatbot_model.py    # Chatbot model interface
│   ├── exllama_chatbot.py  # ExLlama integration
│   ├── gemini.py           # Gemini integration
│   ├── llama.py            # Llama integration
│   └── model_router.py     # Model routing
├── scripts/                 # Consolidated scripts
│   ├── training.py         # Comprehensive training script
│   ├── setup.py            # Setup utilities
│   ├── install_dependencies.py
│   └── run_tests.py
├── tests/                   # All test files
│   ├── test_api.py         # API tests (pytest)
│   ├── test_emergency_handler.py
│   ├── test_generator.py   # Generator tests
│   └── test_tokens.py      # Token tests
├── utils/                   # Utilities
│   ├── common.py           # Common utilities
│   ├── api_manager.py      # API management
│   ├── data_loader.py      # Data loading
│   ├── logger.py           # Logging utilities
│   ├── semantic_search.py  # Semantic search
│   └── token_loader.py     # Token management
├── Database/                # Database management
├── Dataset/                 # Data files
├── Notebook/                # Jupyter notebooks
├── logs/                    # Log files
├── config.py                # Configuration
├── requirements.txt         # Dependencies
├── docker-compose.yml       # Docker setup
├── Dockerfile               # Docker configuration
├── README.md                # Documentation
└── PROJECT_STATUS.md        # Project status
```

---

## ⚡️ Cài đặt & chạy thử

### 1. Clone repo & cài thư viện
```bash
git clone https://github.com/your-username/mental-health-chatbot.git
cd mental-health-chatbot
pip install -r requirements.txt
```

### 2. Thiết lập biến môi trường
Tạo file `.env` từ mẫu `.env.example` và điền các thông tin:
```env
GEMINI_API_KEY=your_gemini_key
DATABASE_URL=sqlite:///chatbot.db
DEBUG=true
```

### 3. Chạy hệ thống

#### Development Mode
```bash
# Chạy cả API Gateway và Model Server
python run_servers.py
```

#### Production Mode
```bash
# Chạy với Docker
docker-compose up -d
```

### 4. Test API

#### Health Check
```bash
# API Gateway
curl http://localhost:8000/health

# Model Server  
curl http://localhost:8001/health
```

#### Chat với LLaMA
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tôi cảm thấy buồn", "prefer_model": "llama"}'
```

#### Chat với Gemini
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tôi cảm thấy buồn", "prefer_model": "gemini"}'
```

#### Auto mode (thử LLaMA trước, fallback Gemini)
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tôi cảm thấy buồn", "prefer_model": "auto"}'
```

---

## 🚀 ExLlama GPTQ Inference Setup

### 1. Setup Environment
```bash
# Cài đặt ExLlama và dependencies
python scripts/setup_exllama.py
```

### 2. Convert Fine-tuned Model to GPTQ
Sau khi fine-tune xong, convert model từ LoRA adapter sang GPTQ format:

```bash
python scripts/convert_to_gptq.py \
    --base_model "meta-llama/Llama-3.2-1B-Instruct" \
    --lora_path "models/weights/chatbot_finetuned_nf4" \
    --output_path "models/weights/chatbot_gptq" \
    --bits 4 \
    --group_size 128
```

### 3. Run Inference

#### Test Generation
```bash
python scripts/exllama_inference.py \
    --model_path "models/weights/chatbot_gptq" \
    --test
```

#### Interactive Chat
```bash
python scripts/exllama_inference.py \
    --model_path "models/weights/chatbot_gptq" \
    --interactive
```

#### Tích hợp vào hệ thống hiện tại
```python
from models.exllama_chatbot import create_exllama_chatbot

# Tạo chatbot instance
chatbot = create_exllama_chatbot("models/weights/chatbot_gptq")

# Sử dụng
response, emotion = chatbot.chat_response("Tôi cảm thấy căng thẳng")
print(f"Response: {response}")
print(f"Emotion: {emotion}")
```

### Performance Comparison

| Method | Memory Usage | Speed | Quality |
|--------|-------------|-------|---------|
| Original (FP16) | ~6GB | 1x | Baseline |
| LoRA (4-bit) | ~2GB | 0.8x | Good |
| GPTQ (4-bit) | ~1.5GB | 1.2x | Good |
| ExLlama GPTQ | ~1.2GB | 1.5x | Good |

---

## 📊 API Response Format

```json
{
  "response": "Tôi hiểu cảm xúc của bạn...",
  "sentiment": "negative",
  "mental_state": "depression", 
  "risk_level": "risky",
  "source": "llama_model_server",
  "warning": "⚠️ RỦI RO: Bạn có thể cân nhắc...",
  "model_used": "llama"
}
```

---

## 🛠️ Development

### 1. Fine-tune Model
```bash
# Fix và chạy training
python fix_and_train.py
```

### 2. Test Model Server
```bash
# Chạy riêng Model Server
python model_server.py

# Test generation
curl -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Tôi cảm thấy buồn"}'
```

### 3. Test API Gateway
```bash
# Chạy riêng API Gateway
uvicorn api_gateway.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 🚨 Troubleshooting

### Model Server không start
```bash
# Kiểm tra GPU
nvidia-smi

# Kiểm tra model path
ls models/weights/chatbot_finetuned/

# Check logs
tail -f logs/model_server.log
```

### API Gateway lỗi
```bash
# Kiểm tra Model Server
curl http://localhost:8001/health

# Check logs
tail -f logs/api_gateway.log
```

### CUDA Out of Memory (ExLlama)
```bash
# Giảm max_seq_len và max_input_len
python scripts/exllama_inference.py \
    --model_path "models/weights/chatbot_gptq" \
    --max_seq_len 1024 \
    --max_input_len 256
```

---

## 📝 Tùy chỉnh & mở rộng
- **Thay đổi templates hội thoại:** Sửa trong `_create_conversation_templates()`
- **Điều chỉnh số lượt hội thoại:** Sửa trong `_generate_multi_turn_conversation()`
- **Thêm expert mới:** Thêm mô-đun vào `services/` và cập nhật logic routing.

---

## 📚 Tham khảo & đóng góp
- Nếu bạn muốn đóng góp, hãy tạo pull request hoặc issue mới.
- Đọc thêm tài liệu chi tiết trong thư mục `Docs/`.
- Xem `services/emergency_handler/README.md` cho thông tin về Emergency Handler.
- Xem `IMPROVED_GENERATOR_README.md` cho thông tin về Data Generation.

---

**Liên hệ hỗ trợ:**
- Email: your.email@example.com
- Hotline khẩn cấp: 0984.104.115

