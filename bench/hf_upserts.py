from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sentence_transformers.quantization import quantize_embeddings
from datasets import load_dataset
import set_path
from picovdb import PicoVectorDB

DIMENSION = 512
BATCH_SIZE = 500
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", truncate_dim=DIMENSION)
rag_full = load_dataset("neural-bridge/rag-full-20000")
rows = rag_full["train"]["clear_prompt"]
hf_data_vdb = PicoVectorDB(embedding_dim=DIMENSION, storage_file="hfdata")


for i in range(len(rows) // BATCH_SIZE + 1):
    start = i * BATCH_SIZE
    end = (i + 1) * BATCH_SIZE
    if end > len(rows):
        end = len(rows)
    print(f"Processing rows {start} to {end}")
    embeddings = model.encode(rows[start:end], show_progress_bar=True, convert_to_tensor=True)
    docs = [{
        "_id_": str(j),
        "_vector_": embeddings[j - start].cpu().numpy(),
        "text": rows[j],
    } for j in range(start, end)]
    hf_data_vdb.upsert(docs)
hf_data_vdb.save()

# query = "A man is eating a piece of bread"
# docs = [
#     "A man is eating food.",
#     "A man is eating pasta.",
#     "The girl is carrying a baby.",
#     "A man is riding a horse.",
# ]
# # The prompt used for query retrieval tasks:
# # query_prompt = 'Represent this sentence for searching relevant passages: '

# # 2. Encode
# query_embedding = model.encode(query, prompt_name="query")
# # Equivalent Alternatives:
# # query_embedding = model.encode(query_prompt + query)
# # query_embedding = model.encode(query, prompt=query_prompt)

# docs_embeddings = model.encode(docs)

# # Optional: Quantize the embeddings
# binary_query_embedding = quantize_embeddings(query_embedding, precision="ubinary")
# binary_docs_embeddings = quantize_embeddings(docs_embeddings, precision="ubinary")

# similarities = cos_sim(query_embedding, docs_embeddings)
# print('similarities:', similarities)