"""
Demonstrates text chunking, Huggingface embedding with sentence-transformer, and storage in a PicoVectorDB for similarity queries.
"""

from sentence_transformers import SentenceTransformer
import set_path
from picovdb import PicoVectorDB

CHUNK_SIZE = 256
model_name = "all-MiniLM-L6-v2"

print(f"Loading st model: {model_name} ...")
model = SentenceTransformer(model_name)
dim: int = model.get_sentence_embedding_dimension() or 384
print("model embedding dimensions:", dim)

with open("A_Christmas_Carol.txt", encoding="UTF8") as f:
    content = f.read()
    num_chunks = len(content) // CHUNK_SIZE + 1
    chunks = [content[i * CHUNK_SIZE : (i + 1) * CHUNK_SIZE] for i in range(num_chunks)]
    embeddings = model.encode(chunks, show_progress_bar=True)
    data = [
        {
            "_vector_": embeddings[i],
            "_id_": i,
            "content": chunks[i],
        }
        for i in range(num_chunks)
    ]
    db = PicoVectorDB(embedding_dim=dim, storage_file="_acc")
    db.upsert(data)
    db.save()


txt = "ghost said it wear a chain it forged in life."
emb = model.encode(txt)
r = db.query(emb, top_k=3)
print("\nquery:", txt)
print("\nresult:\n", r[0]["content"])

txts = [
    "my business are mankind and the common welfare",
    "there are no workhouses aren't there?",
]
emb = model.encode(txts)
rs = db.query(emb, top_k=5)
print("\nqueries:\n", txts)
print("\nresult 0:\n", rs[0][0]["content"])
print("\nresult 1:\n", rs[1][1]["content"])

all_docs = db.get_all()
print(len(all_docs))
print(all_docs[0])
print(all_docs[-1])
