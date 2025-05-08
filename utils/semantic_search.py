import json
from sentence_transformers import SentenceTransformer, util
from utils.data_loader import load_chat_format_data

DATA_PATH = "data/conversations.json"
embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

chat_data = load_chat_format_data(DATA_PATH)
questions = [item["messages"][0]["content"] for item in chat_data]
answers = [item["messages"][1]["content"] for item in chat_data]
question_embeddings = embedder.encode(questions, convert_to_tensor=True)

def find_similar_answer(user_input, threshold=0.75):
    input_embedding = embedder.encode(user_input, convert_to_tensor=True)
    cos_scores = util.pytorch_cos_sim(input_embedding, question_embeddings)[0]
    best_score = cos_scores.max().item()
    best_idx = cos_scores.argmax().item()
    if best_score >= threshold:
        return answers[best_idx]
    return None