from sentence_transformers import SentenceTransformer, util
from utils.data_loader import load_chat_format_data

embedder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
chat_data = load_chat_format_data("data/conversations.json")
questions = [x["messages"][0]["content"] for x in chat_data]
answers = [x["messages"][1]["content"] for x in chat_data]
question_embeddings = embedder.encode(questions, convert_to_tensor=True)

def find_similar_answer(query, threshold=0.75):
    q_emb = embedder.encode(query, convert_to_tensor=True)
    scores = util.pytorch_cos_sim(q_emb, question_embeddings)[0]
    best_idx = scores.argmax().item()
    if scores[best_idx] >= threshold:
        return answers[best_idx]
    return None