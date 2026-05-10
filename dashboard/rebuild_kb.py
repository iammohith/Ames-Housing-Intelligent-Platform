import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

CHROMA_PATH = "/app/chroma"
ARTIFACTS_DIR = "/app/artifacts/knowledge"

def rebuild():
    if not os.path.exists(ARTIFACTS_DIR):
        print(f"Artifacts dir {ARTIFACTS_DIR} not found.")
        return

    runs = os.listdir(ARTIFACTS_DIR)
    if not runs:
        print("No runs found in artifacts dir.")
        return

    runs_with_time = [
        (r, os.path.getmtime(os.path.join(ARTIFACTS_DIR, r))) for r in runs
    ]
    runs_with_time.sort(key=lambda x: x[1], reverse=True)
    run_dir = os.path.join(ARTIFACTS_DIR, runs_with_time[0][0])
    
    print(f"Using latest run dir: {run_dir}")
    documents = []
    if os.path.isdir(run_dir):
        for f in os.listdir(run_dir):
            if f.endswith(".txt"):
                with open(os.path.join(run_dir, f)) as fh:
                    documents.append(
                        {"title": f.replace(".txt", ""), "content": fh.read()}
                    )

    if not documents:
        print("No text documents found.")
        return

    print(f"Loaded {len(documents)} documents. Chunking...")
    
    chunks = []
    CHUNK_SIZE = 512
    import re
    
    for doc in documents:
        content = doc["content"]
        title = doc["title"]
        full_content = f"DOCUMENT: {title}\n\n{content}"
        sentences = re.split(r'(?<=[.!?])\s+', full_content)
        
        current_chunk_words = []
        chunk_word_count = 0
        
        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_word_count = len(sentence_words)
            
            if chunk_word_count + sentence_word_count > CHUNK_SIZE and current_chunk_words:
                chunks.append({
                    "title": title,
                    "content": " ".join(current_chunk_words),
                    "chunk_index": len(chunks),
                })
                overlap_words = current_chunk_words[-20:]
                current_chunk_words = overlap_words + sentence_words
                chunk_word_count = len(current_chunk_words)
            else:
                current_chunk_words.extend(sentence_words)
                chunk_word_count += sentence_word_count
        
        if current_chunk_words:
            chunks.append({
                "title": title,
                "content": " ".join(current_chunk_words),
                "chunk_index": len(chunks),
            })

    print(f"Created {len(chunks)} chunks. Indexing...")

    emb_fn = SentenceTransformerEmbeddingFunction(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    
    try:
        client.delete_collection("ames_knowledge")
    except Exception:
        pass
        
    collection = client.create_collection(
        "ames_knowledge", embedding_function=emb_fn
    )

    ids = [f"manual_{c['chunk_index']}" for c in chunks]
    docs = [c["content"] for c in chunks]
    metadatas = [{"title": c["title"], "run_id": "manual"} for c in chunks]

    if docs:
        collection.add(ids=ids, documents=docs, metadatas=metadatas)
        
    print(f"Successfully indexed {len(chunks)} chunks into ChromaDB.")

if __name__ == "__main__":
    rebuild()
