import os

def load_url(filepath="API_LLM_SERVER.txt") -> str:
    # Sử dụng đường dẫn tuyệt đối đến file trong thư mục llmserver
    current_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(current_dir, filepath)
    with open(full_path, "r", encoding="utf-8") as f:
        url = f.readline().strip()
    return url + "/model/generate/"

API_LLM = load_url()
# if __name__ == "__main__":
#     print(API_LLM)