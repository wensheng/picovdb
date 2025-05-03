from sentence_transformers import SentenceTransformer
import set_path
from picovdb import PicoVectorDB

CHUNK_SIZE = 256

model = SentenceTransformer('all-MiniLM-L6-v2')
print('model embedding dimensions:', model.get_sentence_embedding_dimension())

chunks = []
embeddings = []
db = PicoVectorDB(
    embedding_dim=model.get_sentence_embedding_dimension(),
    storage_file='dulce',
)

data = []
with open('dulce.txt', encoding='UTF8') as f:
    content = f.read()
    num_chunks = len(content) // CHUNK_SIZE + 1
    for i in range(num_chunks):
        # Split the content into chunks of size CHUNK_SIZE
        chunk = content[i * CHUNK_SIZE: (i + 1) * CHUNK_SIZE]
        d = {
            "_vector_": model.encode(chunk),
            "_id_": i,
            "content": chunk,
        }
        data.append(d)

db.upsert(data)
db.save()

print('number of chunks:', len(data))
