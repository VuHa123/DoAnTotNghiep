# Emergency Handler

Module xử lý các tình huống khẩn cấp trong hệ thống chatbot sức khỏe tâm thần.

## 🚨 Chức năng chính

### 1. **Emergency Detection**
- Phát hiện các tin nhắn có dấu hiệu khủng hoảng
- Phân loại mức độ rủi ro: normal, risky, emergency
- Tích hợp với Gating Router để routing thông minh

### 2. **Hotline Integration**
- Gọi hotline 0984.104.115 trong giờ hoạt động (07:30 - 22:00)
- Log đầy đủ thông tin cuộc gọi
- Xử lý lỗi và fallback

### 3. **Staff Notification**
- Gửi cảnh báo cho nhân viên hỗ trợ ngoài giờ
- Log timestamp và thông tin chi tiết
- Tích hợp với hệ thống thông báo

### 4. **Database Logging**
- Lưu trữ tất cả emergency events
- Tracking user_id, action, status
- Hỗ trợ audit trail

## 📁 Cấu trúc files

```
services/emergency_handler/
├── __init__.py           # Module exports
├── handler.py            # Main EmergencyHandler class
├── hotline_caller.py     # Hotline integration
├── staff_notifier.py     # Staff notification
└── README.md            # This file
```

## 🔧 Sử dụng

### Khởi tạo
```python
from services.emergency_handler.handler import EmergencyHandler

handler = EmergencyHandler()
```

### Check Emergency
```python
result = handler.check_emergency(user_id="user123", message="Tôi muốn tự tử")
print(result["status"])    # hotline_called, staff_alerted, error
print(result["message"])   # Response message
print(result["action"])    # hotline, staff_notification, manual_contact
```

### Handle Emergency Endpoint
```python
result = handler.handle_emergency(
    user_id="user123",
    location="Hà Nội",
    contact="0123456789"
)
```

### Legacy Function
```python
from services.emergency_handler.handler import check_and_handle_emergency

message = check_and_handle_emergency("Tôi đang gặp khủng hoảng")
```

## ⚙️ Cấu hình

### Hotline Settings
- **Số điện thoại:** 0984.104.115
- **Giờ hoạt động:** 07:30 - 22:00
- **Ngoài giờ:** Gửi cảnh báo staff

### Database Schema
```sql
CREATE TABLE emergency_logs (
    id INTEGER PRIMARY KEY,
    user_id VARCHAR,
    message TEXT,
    action VARCHAR,
    status VARCHAR,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);
```

## 🔗 Tích hợp

### API Gateway
```python
# Trong api_gateway/main.py
emergency_handler = EmergencyHandler()
emergency_result = emergency_handler.check_emergency(user_id, message)
```

### Server
```python
# Trong server.py
emergency_handler = EmergencyHandler()
if risk_level == "high":
    emergency_result = emergency_handler.check_emergency(user_id, message)
```

## 🧪 Testing

Chạy test script:
```bash
python test_emergency_handler.py
```

## 📝 TODO

### Tích hợp thực tế
- [ ] Twilio API cho gọi điện
- [ ] Slack/Discord webhook cho staff notification
- [ ] Email notification system
- [ ] SMS integration

### Monitoring
- [ ] Dashboard cho emergency events
- [ ] Alert thresholds
- [ ] Performance metrics

### Security
- [ ] Rate limiting
- [ ] User authentication
- [ ] Data encryption

## 🚀 Deployment

1. **Cài đặt dependencies:**
```bash
pip install sqlalchemy
```

2. **Tạo database:**
```python
from Database.core import create_db
create_db()
```

3. **Test functionality:**
```bash
python test_emergency_handler.py
```

## 📞 Support

- **Hotline:** 0984.104.115
- **Email:** support@mentalhealth.com
- **Documentation:** [API Docs](./API.md) 

✅ Emergency Handler tests completed!
- Test 1: Hotline call trong giờ hoạt động
- Test 2: Staff alert ngoài giờ hoạt động  
- Test 3: Emergency endpoint
- Test 4: Legacy function compatibility 