# 🧠 Kiến trúc Hệ thống Chatbot Tâm lý - Tích hợp Feedback System

## 🚦 Kiến trúc & Flow tổng quan (Tích hợp Feedback)

Hệ thống chatbot được thiết kế với các tầng xử lý rõ ràng, bao gồm cả tính năng feedback để cải thiện chất lượng phản hồi. Sơ đồ dưới đây thể hiện kiến trúc tổng quan với tích hợp feedback system:

```mermaid
flowchart TD
    %% Main Chat Flow
    User["Người dùng<br/>(Web UI)"]
    APIGW["API Gateway<br/>(FastAPI)"]
    Gating["Gating Router<br/>(Phân loại risk_level)"]
    Mental["Mental State Classifier"]
    Sentiment["Sentiment Analysis"]
    Emergency["Emergency Handler"]
    Semantic["Semantic Search<br/>(RAG, truy vấn DB)"]
    DB[(MongoDB Database<br/>qdrant_mongoDB)]
    Prompt["Prompt Builder<br/>(Tổng hợp thông tin)"]
    LLM["LLM Server<br/>(MentalGPT)"]
    Clean["Làm sạch phản hồi<br/>(API Gateway)"]
    
    %% Feedback Flow
    FeedbackUI["Feedback UI<br/>(👍/👎 Buttons)"]
    FeedbackAPI["Feedback API<br/>(/feedback)"]
    FeedbackDB[(Feedback Collections<br/>user_feedback<br/>feedback_analytics)]
    AnalyticsAPI["Analytics API<br/>(/feedback/stats)"]
    Admin["Admin/Developer<br/>(Dashboard)"]
    
    %% Main Chat Flow Connections
    User -->|POST /chat| APIGW
    APIGW -->|REST| Gating
    
    Gating -- "Normal" --> Prompt
    Gating -- "Risk" --> Mental
    Gating -- "Risk" --> Sentiment
    Mental --> Prompt
    Sentiment --> Prompt
    Gating -- "Emergency" --> Emergency
    Emergency --> Prompt
    
    Prompt --> Semantic
    Semantic -- "Truy vấn RAG" --> DB
    Semantic --> Prompt
    
    Prompt --> LLM
    LLM --> Clean
    Clean --> APIGW
    APIGW --> User
    
    %% Feedback Flow Connections
    User -->|Nhấn 👍/👎| FeedbackUI
    FeedbackUI -->|POST /feedback| FeedbackAPI
    FeedbackAPI -->|Tạo Feedback object| FeedbackDB
    FeedbackDB -->|Lưu vào user_feedback collection| FeedbackDB
    FeedbackAPI -->|Trả về thông báo| FeedbackUI
    FeedbackUI -->|Hiển thị cảm ơn| User
    
    %% Analytics Flow
    Admin -->|GET /feedback/stats| AnalyticsAPI
    AnalyticsAPI -->|Truy vấn thống kê| FeedbackDB
    FeedbackDB -->|Trả về analytics| AnalyticsAPI
    AnalyticsAPI -->|JSON response| Admin
    
    %% Database Collections
    subgraph "MongoDB Collections"
        RAGCollections["RAG Collections<br/>• conversations<br/>• emergency_logs<br/>• user_sessions<br/>• mental_state_history"]
        FeedbackCollections["Feedback Collections<br/>• user_feedback<br/>• feedback_analytics"]
    end
    
    DB --> RAGCollections
    FeedbackDB --> FeedbackCollections
    
    %% Style với màu trắng và viền đen
    style User fill:#ffffff,stroke:#000000,stroke-width:1px
    style APIGW fill:#ffffff,stroke:#000000,stroke-width:1px
    style Gating fill:#ffffff,stroke:#000000,stroke-width:1px
    style Mental fill:#ffffff,stroke:#000000,stroke-width:1px
    style Sentiment fill:#ffffff,stroke:#000000,stroke-width:1px
    style Emergency fill:#ffffff,stroke:#000000,stroke-width:1px
    style Semantic fill:#ffffff,stroke:#000000,stroke-width:1px
    style DB fill:#ffffff,stroke:#000000,stroke-width:1px
    style Prompt fill:#ffffff,stroke:#000000,stroke-width:1px
    style LLM fill:#ffffff,stroke:#000000,stroke-width:1px
    style Clean fill:#ffffff,stroke:#000000,stroke-width:1px
    style FeedbackUI fill:#ffffff,stroke:#000000,stroke-width:1px
    style FeedbackAPI fill:#ffffff,stroke:#000000,stroke-width:1px
    style FeedbackDB fill:#ffffff,stroke:#000000,stroke-width:1px
    style AnalyticsAPI fill:#ffffff,stroke:#000000,stroke-width:1px
    style Admin fill:#ffffff,stroke:#000000,stroke-width:1px
```

## 📊 Chi tiết Luồng Xử lý Feedback

### **1. Luồng Thu thập Feedback**

```mermaid
sequenceDiagram
    participant User as Người dùng
    participant UI as Frontend UI
    participant API as API Gateway
    participant DB as MongoDB
    participant Admin as Admin/Developer
    
    Note over User,DB: Luồng thu thập feedback
    User->>UI: Nhấn 👍/👎 sau phản hồi bot
    UI->>API: POST /feedback
    API->>DB: Tạo Feedback object
    DB->>DB: Lưu vào user_feedback collection
    API->>UI: Trả về thông báo thành công
    UI->>User: Hiển thị "Cảm ơn bạn đã đưa ra phản hồi!"
    
    Note over User,DB: Luồng phân tích feedback
    Admin->>API: GET /feedback/stats
    API->>DB: Truy vấn thống kê feedback
    DB->>API: Trả về analytics data
    API->>Admin: JSON response với thống kê
```

### **2. Cấu trúc Database Collections**

```mermaid
graph TD
    MongoDB[MongoDB Database<br/>qdrant_mongoDB] --> RAG[RAG Collections<br/>Knowledge Base & Chatbot Operations]
    MongoDB --> Feedback[Feedback Collections<br/>User Feedback & Analytics]
    
    RAG --> Conversations[conversations<br/>• user_input<br/>• bot_response<br/>• created_at]
    RAG --> EmergencyLogs[emergency_logs<br/>• user_id<br/>• message<br/>• action<br/>• status<br/>• timestamp]
    RAG --> UserSessions[user_sessions<br/>• started_at]
    RAG --> MentalStateHistory[mental_state_history<br/>• mental_state<br/>• detected_at]
    
    Feedback --> UserFeedback[user_feedback<br/>• session_id<br/>• user_input<br/>• bot_response<br/>• feedback_type<br/>• user_feedback_text<br/>• risk_level<br/>• emotion_label<br/>• timestamp]
    Feedback --> FeedbackAnalytics[feedback_analytics<br/>• date<br/>• session_id<br/>• satisfaction_rate<br/>• total_feedback]
```

## 🔄 Tích hợp với Kiến trúc Hiện tại

### **Chức năng từng module (Cập nhật):**

- **Người dùng (Web UI)** (*Giao diện trò chuyện + Feedback buttons*)
- **API Gateway (FastAPI)** (*Trung tâm điều phối + Feedback endpoints*)
- **Gating Router** (*Phân loại mức độ rủi ro: Normal, Risk, Emergency*)
- **Mental State Classifier** (*Phân loại trạng thái tâm thần*)
- **Sentiment Analysis** (*Phân tích cảm xúc*)
- **Emergency Handler** (*Xử lý khẩn cấp*)
- **Semantic Search (RAG)** (*Truy vấn database để lấy thông tin liên quan*)
- **Database** (*Lưu lịch sử hội thoại, tri thức, log, cảnh báo + Feedback data*)
- **Prompt Builder** (*Tổng hợp tất cả thông tin*)
- **LLM Server** (*MentalGPT model sinh phản hồi*)
- **Làm sạch phản hồi** (*Chuẩn hóa, format phản hồi*)
- **Feedback System** (*Thu thập và phân tích phản hồi người dùng*)

### **Luồng xử lý chính (Tích hợp Feedback):**

1. **User** gửi message qua Web UI
2. **API Gateway** nhận message qua REST, chuyển cho **Gating Router**
3. **Gating Router** phân loại risk_level:
   - **Normal:** → Tạo prompt trực tiếp
   - **Risk:** → Gọi **Mental State Classifier** & **Sentiment Analysis**
   - **Emergency:** → Gọi **Emergency Handler**
4. **Semantic Search** truy vấn DB (RAG) để lấy thông tin liên quan
5. **Prompt Builder** tổng hợp tất cả thông tin thành prompt hoàn chỉnh
6. **LLM Server** sinh phản hồi dựa trên prompt → trả về **API Gateway**
7. **API Gateway** làm sạch, chuẩn hóa phản hồi → gửi về **UI** cho người dùng
8. **User** đánh giá phản hồi (👍/👎) → **Feedback System** thu thập và lưu trữ
9. **Admin/Developer** truy cập analytics để phân tích và cải thiện hệ thống

## 📈 API Endpoints (Tích hợp Feedback)

### **API Gateway (Port 8000)**
- `POST /chat` — Main chat endpoint
- `POST /feedback` — Thu thập feedback từ người dùng
- `GET /feedback/stats` — Thống kê feedback
- `GET /feedback/dislikes` — Danh sách feedback tiêu cực
- `POST /feedback/export` — Xuất dữ liệu feedback
- `GET /health` — Health check
- `POST /emergency` — Emergency handling
- `POST /semantic_search` — RAG search

### **LLM Server (Port 8001)**
- `POST /model/generate/` — Generate streaming response
- `GET /health` — Health check Model Server

## 🎯 Lợi ích của Tích hợp Feedback

1. **Cải thiện chất lượng**: Sử dụng feedback để fine-tune model
2. **Phân tích xu hướng**: Theo dõi satisfaction rate theo thời gian
3. **Phát hiện vấn đề**: Xác định các phản hồi có vấn đề để khắc phục
4. **Cá nhân hóa**: Điều chỉnh phản hồi dựa trên feedback pattern
5. **Ra quyết định**: Cung cấp dữ liệu để đưa ra quyết định cải tiến

## 🔧 Cấu hình Database Indexes

```python
# RAG Collections indexes
conversations.create_index([("created_at", -1)])
emergency_logs.create_index([("user_id", 1)])
emergency_logs.create_index([("timestamp", -1)])
user_sessions.create_index([("started_at", -1)])
mental_state_history.create_index([("detected_at", -1)])

# Feedback Collections indexes
user_feedback.create_index([("session_id", 1)])
user_feedback.create_index([("feedback_type", 1)])
user_feedback.create_index([("timestamp", -1)])
user_feedback.create_index([("risk_level", 1)])
user_feedback.create_index([("emotion_label", 1)])
feedback_analytics.create_index([("date", -1)])
feedback_analytics.create_index([("session_id", 1)])
```

Hệ thống được thiết kế với kiến trúc microservice, tách biệt rõ ràng giữa các thành phần và đảm bảo khả năng mở rộng, bảo trì dễ dàng. Feedback system được tích hợp một cách mượt mà vào luồng xử lý chính mà không làm ảnh hưởng đến hiệu suất của hệ thống. 