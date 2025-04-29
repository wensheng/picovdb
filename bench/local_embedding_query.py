from sentence_transformers import SentenceTransformer, util
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

#txt = 'their eyes revealing a spark of understanding'
#txt = 'ns. Yet, beneath the veneer of duty, the enticement of the vast unknown pulled them inexorably together, coalescing their distinct desires into a shared pulse of anticipation.\n\nMarshaled back to the moment by the blink of lights and whir of machinery, the'
#emb = model.encode(txt)
txt = "I'd laugh if we run into Martians playing poker down there—just to lighten the mood, you know?"
emb = model.encode(txt)

q = db.query(emb, top_k=5, better_than=0.1)
print('query results:', q)

all_docs = db.get_all()
print(len(all_docs))
print(all_docs[0])
