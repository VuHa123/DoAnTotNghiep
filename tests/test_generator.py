import pandas as pd
import json
import os
from multi_turn_conversation_generator import MultiTurnConversationGenerator

def create_test_data():
    """Tạo dữ liệu test nhỏ"""
    test_data = {
        'question': [
            "Tôi cảm thấy rất buồn và không có hy vọng về tương lai",
            "Tôi lo lắng về công việc và tài chính",
            "Tôi cảm thấy căng thẳng với các mối quan hệ",
            "Tôi có những suy nghĩ về việc kết thúc cuộc sống",
            "Tôi có những thay đổi tâm trạng mạnh mẽ từ vui vẻ đến buồn bã"
        ],
        'answer': [
            "Tôi hiểu cảm giác của bạn. Điều này phải rất khó khăn. Hãy chia sẻ thêm với tôi.",
            "Lo lắng về tương lai là điều bình thường. Bạn có thể nói rõ hơn về những lo lắng này không?",
            "Các mối quan hệ có thể gây căng thẳng. Bạn có thể chia sẻ thêm về tình huống này không?",
            "Tôi rất quan tâm đến sự an toàn của bạn. Hãy nói chuyện với tôi về những suy nghĩ này.",
            "Những thay đổi tâm trạng có thể khó khăn để đối phó. Bạn có thể mô tả thêm không?"
        ],
        'predicted_status': ['Depression', 'Anxiety', 'Stress', 'Suicidal', 'Bipolar'],
        'confidence': [0.95, 0.88, 0.92, 0.99, 0.85],
        'pred_intensity': [2.5, 1.8, 1.5, 3.0, 2.0],
        'conf_intensity': [0.90, 0.85, 0.88, 0.95, 0.82],
        'pred_strategy': ['Empathize', 'Explore', 'Validate', 'Encourage', 'Explore'],
        'conf_strategy': [0.92, 0.87, 0.89, 0.94, 0.84]
    }
    
    df = pd.DataFrame(test_data)
    df.to_csv('test_data.csv', index=False, encoding='utf-8')
    print("Đã tạo file test_data.csv")
    return 'test_data.csv'

def test_generator():
    """Test generator với dữ liệu nhỏ"""
    print("=== TEST MULTI-TURN CONVERSATION GENERATOR ===")
    
    # Tạo dữ liệu test
    test_file = create_test_data()
    
    # Khởi tạo generator với chunk size nhỏ
    generator = MultiTurnConversationGenerator(
        chunk_size=2,  # Chỉ 2 sample/chunk để test
        output_dir="test_conversations"
    )
    
    # Xử lý dữ liệu test
    print(f"\nĐang xử lý file: {test_file}")
    generator.process_data_in_chunks(test_file, start_chunk=0)
    
    # Kiểm tra kết quả
    print("\n=== KIỂM TRA KẾT QUẢ ===")
    
    # Đọc file summary
    summary_file = os.path.join("test_conversations", "processing_summary.json")
    if os.path.exists(summary_file):
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        print(f"Tổng số chunks đã xử lý: {summary['total_chunks_processed']}")
        print(f"Tổng số cuộc hội thoại: {summary['total_conversations']}")
        
        # Đọc một file chunk để kiểm tra format
        if summary['chunks']:
            chunk_file = summary['chunks'][0]['filepath']
            if os.path.exists(chunk_file):
                with open(chunk_file, 'r', encoding='utf-8') as f:
                    chunk_data = json.load(f)
                
                print(f"\nThông tin chunk đầu tiên:")
                print(f"- Chunk ID: {chunk_data['metadata']['chunk_id']}")
                print(f"- Số cuộc hội thoại: {chunk_data['metadata']['total_conversations']}")
                print(f"- Dòng bắt đầu: {chunk_data['metadata']['start_index']}")
                print(f"- Dòng kết thúc: {chunk_data['metadata']['end_index']}")
                
                # Kiểm tra một cuộc hội thoại mẫu
                if chunk_data['conversations']:
                    conv = chunk_data['conversations'][0]
                    print(f"\nVí dụ cuộc hội thoại:")
                    print(f"- ID: {conv['conversation_id']}")
                    print(f"- Trạng thái: {conv['predicted_status']}")
                    print(f"- Cường độ: {conv['pred_intensity']}")
                    print(f"- Chiến lược: {conv['pred_strategy']}")
                    print(f"- Số lượt: {len(conv['turns'])}")
                    
                    # Hiển thị 2 lượt đầu
                    for i, turn in enumerate(conv['turns'][:2]):
                        print(f"\nLượt {i+1}:")
                        print(f"  User: {turn['user'][:100]}...")
                        print(f"  Assistant: {turn['assistant'][:100]}...")
                        if turn['follow_up']:
                            print(f"  Follow-up: {turn['follow_up'][:100]}...")
    
    print("\n=== TEST HOÀN THÀNH ===")

def test_resume_functionality():
    """Test tính năng resume"""
    print("\n=== TEST RESUME FUNCTIONALITY ===")
    
    # Tạo dữ liệu lớn hơn
    large_test_data = {
        'question': [f"Test question {i}" for i in range(10)],
        'answer': [f"Test answer {i}" for i in range(10)],
        'predicted_status': ['Depression'] * 5 + ['Anxiety'] * 5,
        'confidence': [0.9] * 10,
        'pred_intensity': [2.0] * 10,
        'conf_intensity': [0.9] * 10,
        'pred_strategy': ['Empathize'] * 10,
        'conf_strategy': [0.9] * 10
    }
    
    df = pd.DataFrame(large_test_data)
    df.to_csv('large_test_data.csv', index=False, encoding='utf-8')
    
    # Khởi tạo generator
    generator = MultiTurnConversationGenerator(
        chunk_size=3,  # 3 sample/chunk
        output_dir="test_resume"
    )
    
    # Xử lý từ chunk 1 (bỏ qua chunk 0)
    print("Đang test resume từ chunk 1...")
    generator.process_data_in_chunks('large_test_data.csv', start_chunk=1)
    
    # Kiểm tra summary
    summary_file = os.path.join("test_resume", "processing_summary.json")
    if os.path.exists(summary_file):
        with open(summary_file, 'r', encoding='utf-8') as f:
            summary = json.load(f)
        
        print(f"Chunks đã xử lý: {[c['chunk_id'] for c in summary['chunks']]}")
        print("Resume test hoàn thành!")

if __name__ == "__main__":
    # Test cơ bản
    test_generator()
    
    # Test resume
    test_resume_functionality()
    
    print("\nTất cả test hoàn thành!") 