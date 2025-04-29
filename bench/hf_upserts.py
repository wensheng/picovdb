from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
from sentence_transformers.quantization import quantize_embeddings
from datasets import load_dataset
import set_path
from picovdb import PicoVectorDB

DIMENSION = 512
model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1", truncate_dim=DIMENSION)
rag_data = load_dataset("neural-bridge/rag-dataset-1200")

embeddings = model.encode(
    rag_data['train']['context'],
    show_progress_bar=True,
    convert_to_tensor=True
)
rows = rag_data["train"].to_list()
docs = [{
    "_id_": str(i),
    "_vector_": embeddings[i].cpu().numpy(),
    "context": row['context'],
    "question": row['question'],
    "answer": row['answer'],
} for i, row in enumerate(rows)]

hf_data_vdb = PicoVectorDB(embedding_dim=DIMENSION, storage_file="hfdata")
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
