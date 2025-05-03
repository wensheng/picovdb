from sentence_transformers import SentenceTransformer
import set_path
from picovdb import PicoVectorDB

CHUNK_SIZE = 256

model = SentenceTransformer('all-MiniLM-L6-v2')
print('model embedding dimensions:', model.get_sentence_embedding_dimension())

with open('A_Christmas_Carol.txt', encoding='UTF8') as f:
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
        storage_file='_acc',
    )
    db.upsert(data)
    db.save()


txt = "'I wear the chain I forged in life,' replied the Ghost."
emb = model.encode(txt)
r = db.query(emb, top_k=3)
print('\nquery result:\n', r[0]['content'])

txts = [
    "Mankind was my business. The common welfare was my business; charity, mercy, forbearance, and benevolence were, all, my business.",
    "Are there no prisons? Are there no workhouses?",
]
emb = model.encode(txts)
rs = db.query(emb, top_k=5)
print('\nquery result 0:\n', rs[0][0]['content'])
print('\nquery result 1:\n', rs[1][1]['content'])

all_docs = db.get_all()
print(len(all_docs))
print(all_docs[0])
print(all_docs[-1])
