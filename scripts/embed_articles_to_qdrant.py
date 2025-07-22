import os
import pymongo
from qdrant_client import QdrantClient, models
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import logging
from tqdm import tqdm
import uuid
from services.semantic_search import SemanticIndexer

# --- CONFIG ---
MONGO_URI = "mongodb://localhost:27017/"
MONGO_DB = "psychology_papers"  # Sửa lại nếu bạn đổi tên DB
MONGO_COLLECTION = "papers"     # Sửa lại nếu bạn đổi tên collection
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
QDRANT_COLLECTION = "psychology_chunks"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 400  # words
CHUNK_OVERLAP = 50  # words

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- FUNCTIONS ---
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        return text
    except Exception as e:
        logging.warning(f"Could not extract PDF {pdf_path}: {e}")
        return ""

def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", ".", " "]
    )
    return splitter.split_text(text)

def main():
    # 1. Kết nối MongoDB
    mongo_client = pymongo.MongoClient(MONGO_URI)
    db = mongo_client[MONGO_DB]
    collection = db[MONGO_COLLECTION]

    # 2. Khởi tạo SemanticIndexer
    indexer = SemanticIndexer()

    # 4. Lấy tất cả bài báo CHƯA embedding
    articles = list(collection.find({"$or": [{"embedded": {"$exists": False}}, {"embedded": False}]}))
    logging.info(f"Found {len(articles)} articles in MongoDB to embed.")

    chunk_records = []
    for article in tqdm(articles, desc="Processing articles"):
        article_id = str(article.get('_id'))
        title = article.get('title', '')
        abstract = article.get('abstract', '')
        year = article.get('publication_year', None)
        source = article.get('source', '')
        pdf_path = article.get('pdf_path', '')
        content = article.get('content', '')

        # Chỉ xử lý nếu có pdf_path và file thực sự tồn tại
        if not pdf_path or not os.path.exists(pdf_path):
            continue  # Bỏ qua bài báo không có file PDF thực sự

        # Trích xuất nội dung từ PDF
        content = extract_text_from_pdf(pdf_path)

        # Gộp abstract + content (nếu muốn chỉ lấy content thì bỏ abstract)
        full_text = (abstract or "") + "\n" + (content or "")
        if not full_text.strip():
            continue

        # Chunking
        chunks = chunk_text(full_text)
        for idx, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            chunk_records.append({
                "article_id": article_id,
                "title": title,
                "year": year,
                "source": source,
                "chunk_id": chunk_id,
                "chunk_text": chunk,
                "url": article.get('url', ''),
            })

    logging.info(f"Total chunks to embed: {len(chunk_records)}")

    # 5. Embedding và lưu vào Qdrant (dùng module)
    indexer.upsert_chunks(chunk_records)

    # Cập nhật trạng thái embedded cho các article_id trong batch
    article_ids = list(set([c["article_id"] for c in chunk_records]))
    for aid in article_ids:
        try:
            if isinstance(aid, str):
                from bson import ObjectId
                aid_obj = ObjectId(aid)
            else:
                aid_obj = aid
            collection.update_one({"_id": aid_obj}, {"$set": {"embedded": True}})
        except Exception as e:
            logging.warning(f"Could not update embedded status for article_id {aid}: {e}")
    logging.info("All chunks embedded and uploaded to Qdrant.")

if __name__ == "__main__":
    main() 