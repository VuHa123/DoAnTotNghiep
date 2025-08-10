## POST /chat
Yêu cầu:
```json
{
  "user_input": "Tôi cảm thấy rất mệt mỏi",
  "history": [
    {
      "user": "Tôi lo lắng về kỳ thi",
      "bot": "Tôi hiểu cảm giác của bạn. Kỳ thi có thể gây căng thẳng rất lớn."
    },
    {
      "user": "Tôi không thể ngủ được",
      "bot": "Mất ngủ do lo lắng là rất phổ biến. Bạn có thể thử các kỹ thuật thư giãn."
    }
  ]
}
```

**Lưu ý về format history:**
- **Format mới (khuyến nghị)**: `list[dict]` với mỗi dict chứa `user` và `bot` messages
- **Format cũ (tương thích ngược)**: `list[str]` chỉ chứa user messages
- Hệ thống sẽ tự động lấy 3 lượt gần nhất để tạo context
Phản hồi:
```json
{
  "bot_response": "Tôi hiểu bạn đang rất áp lực..."
}
