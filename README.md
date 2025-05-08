
# Chatbot Tư vấn Tâm lý

## Giới thiệu
Đây là một dự án chatbot hỗ trợ tâm lý được xây dựng bằng mô hình ngôn ngữ lớn LLaMA-3.2-1B và Gemini. Mục tiêu của dự án là tạo ra một hệ thống chatbot giúp hỗ trợ người dùng trong việc đối mặt với stress và các vấn đề tâm lý khác. Hệ thống sử dụng các mô hình ngôn ngữ hiện đại để cung cấp các phản hồi đồng cảm và có tính chất hỗ trợ.

## 🚀 Tính năng

* **Semantic Search**: Tìm phản hồi gần nhất từ tập Q\&A gốc.
* **Generative Model**: Sinh phản hồi mới khi không có câu trả lời tương tự.
* **Phân loại cảm xúc**: Nhãn 3 lớp (Normal, Anxiety, Depression), gộp suicidal vào Depression.
* **Gợi ý hành động**: Đề xuất phương pháp hỗ trợ phù hợp với từng trạng thái.
* **Giao diện Gradio**: Cho phép chọn mô hình (Llama hoặc Gemini), hiển thị lịch sử trò chuyện và lưu log.

## 📁 Cấu trúc thư mục

```bash
project_root/
├── app.py                     # Giao diện Gradio
├── .env                       # Biến môi trường: HF_TOKEN, GEMINI_API_KEY
├── README.md                  # Hướng dẫn sử dụng
├── requirements.txt           # Các thư viện cần cài đặt
├── data/
│   ├── conversations.json     # Dữ liệu Q&A gốc (~950k cặp)
│   └── demo_1000.json         # 1000 mẫu demo đã gán nhãn
├── logs/
│   └── chat_logs/             # Thư mục lưu log trò chuyện
├── models/
│   ├── __init__.py
│   ├── llama_model.py
│   ├── gemini_api.py
│   ├── chatbot_model.py
│   └── model_router.py
├── utils/
│   ├── __init__.py
│   ├── data_loader.py
│   ├── semantic_search.py
│   └── logger.py
└── scripts/
    ├── prepare_data.py        # Chuẩn bị demo_1000.json với label
    ├── finetune_qlora.py      # Fine-tune LLaMA với QLoRA
    └── run_inference.py       # Chạy inference model fine-tuned
```

## 💻 Cài đặt

### 1. Clone repository

```bash
git clone <repo_url>
cd project_root
```

### 2. Tạo virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 4. Cấu hình biến môi trường

Tạo file `.env` với nội dung:

```ini
HF_TOKEN=your_huggingface_token_here
GEMINI_API_KEY=your_gemini_api_key_here
```

## 📊 Chạy demo local

### 1. Chuẩn bị dữ liệu demo

```bash
python scripts/prepare_data.py
```

### 2. Fine-tune mô hình (tùy chọn)

```bash
python scripts/finetune_qlora.py
```

### 3. Chạy giao diện Gradio

```bash
python app.py
```

### 4. Sử dụng

* Nhập tin nhắn, chọn mô hình Llama hoặc Gemini.
* Xem phản hồi, nhãn cảm xúc và gợi ý hành động.
* Lịch sử trò chuyện sẽ hiển thị.
* Nhấn **Lưu lại buổi trò chuyện** để ghi log.

## 🔧 Triển khai

* Dễ dàng deploy lên HuggingFace Spaces hoặc server riêng.
* Đóng gói Docker bằng cách tạo `Dockerfile` và `docker-compose.yml`.

## 🎓 Nâng cao

* Huấn luyện mô hình phân loại riêng để tăng độ chính xác.
* Mở rộng nhãn (ví dụ: Stress, Loneliness).
* Tích hợp chatbot vào website hoặc ứng dụng di động.

---

**Liên hệ**: \[Your Name] – Email: [your\_email@example.com](mailto:your_email@example.com)
