from sentence_transformers import SentenceTransformer
import set_path
from picovdb import PicoVectorDB

CHUNK_SIZE = 256

model = SentenceTransformer('all-MiniLM-L6-v2')
print('model embedding dimensions:', model.get_sentence_embedding_dimension())

with open('dulce.txt', encoding='UTF8') as f:
    content = f.read()
    num_chunks = len(content) // CHUNK_SIZE + 1
    chunks = [content[i * CHUNK_SIZE: (i + 1) * CHUNK_SIZE] for i in range(num_chunks)]
    embeddings = model.encode(chunks)
    data = [
        {
            "_vector_": embeddings[i],
            "_id_": i,
            "content": chunks[i],
        }
        for i in range(num_chunks)
    ]
    db = PicoVectorDB(
        embedding_dim=model.get_sentence_embedding_dimension(),
        storage_file='_dulce',
    )
    db.upsert(data)
    db.save()


#txt = 'their eyes revealing a spark of understanding'
#txt = 'ns. Yet, beneath the veneer of duty, the enticement of the vast unknown pulled them inexorably together, coalescing their distinct desires into a shared pulse of anticipation.\n\nMarshaled back to the moment by the blink of lights and whir of machinery, the'
#emb = model.encode(txt)
txt = "I'd laugh if we run into Martians playing poker down there—just to lighten the mood, you know?"
emb = model.encode(txt)

q = db.query(emb, top_k=3)
print('query results:', q[0]['content'])

all_docs = db.get_all()
print(len(all_docs))
print(all_docs[0])
print(all_docs[-1])