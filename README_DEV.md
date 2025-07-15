# Mental Health Chatbot - Development Mode

Hướng dẫn triển khai ứng dụng Mental Health Chatbot trong môi trường development thay vì sử dụng Docker.

## 🚀 Quick Start

### 1. Setup môi trường development

```bash
# Chạy script setup tự động
python setup_dev.py
```

Script này sẽ:
- Tạo virtual environment
- Cài đặt tất cả dependencies
- Tạo các thư mục cần thiết
- Kiểm tra model files
- Tạo các script development

### 2. Kích hoạt virtual environment

```bash
# Linux/Mac
source .venv/bin/activate

# Windows
.venv\Scripts\activate
```

### 3. Chạy ứng dụng

#### Chạy tất cả services cùng lúc:
```bash
python run_dev.py
```

#### Chạy từng service riêng biệt:
```bash
# Chỉ chạy Model Server
python run_individual_services.py model_server

# Chỉ chạy API Gateway
python run_individual_services.py api_gateway
```

### 4. Test ứng dụng

```bash
python test_dev.py
```

## 📋 Services

### Model Server (Port 8001)
- **URL**: http://localhost:8001
- **Docs**: http://localhost:8001/docs
- **Logs**: `logs/model_server.log`

**Endpoints:**
- `GET /health` - Health check
- `POST /generate` - Generate response
- `GET /model-info` - Model information

### API Gateway (Port 8000)
- **URL**: http://localhost:8000
- **Docs**: http://localhost:8000/docs
- **Logs**: `logs/api_gateway.log`

**Endpoints:**
- `GET /health` - Health check
- `POST /chat` - Chat endpoint
- `POST /emergency` - Emergency handling
- `GET /context/{user_id}` - Get context
- `DELETE /context/{user_id}` - Clear context

## 🛠️ Development Scripts

### `setup_dev.py`
Setup môi trường development:
- Tạo virtual environment
- Cài đặt dependencies
- Tạo thư mục cần thiết
- Kiểm tra model files

### `run_dev.py`
Chạy tất cả services cùng lúc:
- Model Server (port 8001)
- API Gateway (port 8000)
- Health monitoring
- Graceful shutdown

### `run_individual_services.py`
Chạy từng service riêng biệt:
```bash
python run_individual_services.py model_server
python run_individual_services.py api_gateway
```

### `test_dev.py`
Test các services:
- Health check API Gateway
- Health check Model Server
- Verify endpoints

## 📁 Project Structure

```
DoAnTotNghiep/
├── api_gateway/           # API Gateway service
│   ├── main.py           # FastAPI app
│   └── chatbot_api.py    # Chatbot endpoints
├── services/             # Core services
│   ├── chatbot/         # Chatbot logic
│   ├── gating_router/   # Risk assessment
│   ├── mental_state_classifier/  # Mental state detection
│   ├── setiment_analysis/       # Sentiment analysis
│   └── emergency_handler/       # Emergency handling
├── models/              # Model weights
│   └── weights/
├── logs/               # Log files
├── run_dev.py          # Main development script
├── setup_dev.py        # Setup script
├── run_individual_services.py  # Individual service runner
└── test_dev.py         # Test script
```

## 🔧 Configuration

### Environment Variables
- `PYTHONPATH`: Đường dẫn đến thư mục gốc
- `DEV_MODE`: Bật chế độ development

### Model Paths
- Base model: `models/weights/base_model/meta-llama/Llama-3.2-1B-Instruct/`
- Fine-tuned model: `models/weights/chatbot_finetuned/final_model/`
- Gating router: `models/weights/gating_router/`
- Mental state classifier: `models/weights/mental_state/`
- Sentiment analyzer: `models/weights/sentiment/`

## 🐛 Troubleshooting

### Common Issues

#### 1. Port already in use
```bash
# Kiểm tra process đang sử dụng port
lsof -i :8000
lsof -i :8001

# Kill process
kill -9 <PID>
```

#### 2. Model files missing
```bash
# Kiểm tra model files
ls -la models/weights/

# Download models nếu cần
python download_llama_model.py
```

#### 3. Dependencies missing
```bash
# Cài đặt lại dependencies
python install_dev.py
```

#### 4. Virtual environment issues
```bash
# Xóa và tạo lại virtual environment
rm -rf .venv
python setup_dev.py
```

### Logs
- Model Server: `logs/model_server.log`
- API Gateway: `logs/api_gateway.log`

### Health Checks
```bash
# Test API Gateway
curl http://localhost:8000/health

# Test Model Server
curl http://localhost:8001/health
```

## 🧪 Testing

### Manual Testing
```bash
# Test chat endpoint
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_input": "Hello", "history": [], "session_id": "test"}'

# Test model generation
curl -X POST http://localhost:8001/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello"}'
```

### Automated Testing
```bash
python test_dev.py
```

## 📊 Monitoring

### Health Monitoring
- Automatic health checks
- Process monitoring
- Graceful shutdown

### Logs
- Structured logging
- Separate log files per service
- Error tracking

## 🔄 Development Workflow

1. **Setup**: `python setup_dev.py`
2. **Activate**: `source .venv/bin/activate`
3. **Run**: `python run_dev.py`
4. **Test**: `python test_dev.py`
5. **Develop**: Edit code, auto-reload enabled
6. **Stop**: `Ctrl+C`

## 🚀 Production vs Development

### Development Mode
- Auto-reload enabled
- Detailed logging
- Debug information
- Local model loading

### Production Mode
- Optimized performance
- Minimal logging
- Docker deployment
- Load balancing

## 📝 Notes

- Tất cả services chạy trên localhost
- Auto-reload cho API Gateway
- Model Server cần thời gian load model
- Logs được lưu trong thư mục `logs/`
- Graceful shutdown với `Ctrl+C`

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Make changes
4. Test with `python test_dev.py`
5. Submit pull request

## 📞 Support

Nếu gặp vấn đề:
1. Kiểm tra logs trong thư mục `logs/`
2. Chạy `python test_dev.py`
3. Kiểm tra health endpoints
4. Xem troubleshooting section 