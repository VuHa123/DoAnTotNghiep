# Chatbot Tâm Lý - LLaMA-3.2-1B (Demo)

## Giới thiệu
Đây là một dự án chatbot hỗ trợ tâm lý được xây dựng bằng mô hình ngôn ngữ lớn LLaMA-3.2-1B và Gemini. Mục tiêu của dự án là tạo ra một hệ thống chatbot giúp hỗ trợ người dùng trong việc đối mặt với stress và các vấn đề tâm lý khác. Hệ thống sử dụng các mô hình ngôn ngữ hiện đại để cung cấp các phản hồi đồng cảm và có tính chất hỗ trợ.

## Cấu trúc thư mục

project_root/
├── app.py
├── .env
├── README.md
├── requirements.txt
├── data/
│   └── conversations.json
├── logs/
│   └── chat_logs/
├── models/
│   ├── __init__.py
│   ├── llama_model.py
│   ├── gemini_api.py
│   ├── model_router.py
│   └── chatbot_model.py
├── utils/
│   ├── __init__.py
│   ├── semantic_search.py
│   ├── logger.py
│   └── data_loader.py


## Các bước cài đặt và chạy dự án

### 1. Cài đặt các thư viện yêu cầu
Trước tiên, bạn cần cài đặt các thư viện cần thiết từ file `requirements.txt`:

```bash
pip install -r requirements.txt

### 2. Cấu hình môi trường
Tạo một file .env trong thư mục gốc của dự án và thêm token HuggingFace API của bạn vào file
Trước tiên, bạn cần cài đặt các thư viện cần thiết từ 

### 3. Chạy ứng dụng
Chạy ứng dụng Gradio để khởi động giao diện chatbot:
```bash
python app.py

Ứng dụng sẽ mở trong trình duyệt của bạn, cho phép bạn trò chuyện với chatbot hỗ trợ tâm lý.

### 4. Cấu trúc các module
app.py: Đây là file chính để chạy ứng dụng Gradio. Nó sẽ tạo giao diện người dùng và liên kết các hàm xử lý từ chatbot_model.py.

models/:

llama_model.py: Chứa các hàm để xử lý phản hồi từ mô hình LLaMA.

gemini_api.py: Chứa các hàm để xử lý phản hồi từ mô hình Gemini.

model_router.py: Chịu trách nhiệm chọn lựa mô hình cần sử dụng dựa trên input từ người dùng.

utils/:

semantic_search.py: Xử lý tìm kiếm câu trả lời phù hợp từ dữ liệu đã lưu.

logger.py: Ghi lại lịch sử trò chuyện vào file JSON trong thư mục logs/chat_logs/.

### 5. Lịch sử trò chuyện
Mỗi khi người dùng trò chuyện với chatbot, lịch sử trò chuyện sẽ được lưu vào thư mục logs/chat_logs/ dưới dạng file JSON. Mỗi file có tên theo timestamp, ví dụ: conversation_2025-04-16_12-30-45.json.

### 6. Cập nhật mô hình
Hiện tại, hệ thống hỗ trợ 2 mô hình chính:

LLaMA-3.2-1B: Được tải từ HuggingFace.

Gemini: Mô hình API được tích hợp sẵn.

Mô hình được lựa chọn tự động qua tham số model_name trong hàm get_response.

### 7. Thêm mô hình mới
Để thêm mô hình mới vào hệ thống, bạn cần thực hiện các bước sau:

Tạo một file mới trong thư mục models/, ví dụ: new_model.py.

Thêm hàm xử lý phản hồi trong file này.

Cập nhật model_router.py để hỗ trợ mô hình mới.