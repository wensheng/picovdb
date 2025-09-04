import random
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
from picovdb import PicoVectorDB

DIMENSION = 512
USE_QUESTION = True
# should have no mismatch if 0
# no mismatches should decrease from 100 to 500
PASSAGE_LENGTH = 0
model = SentenceTransformer(
    "mixedbread-ai/mxbai-embed-large-v1", truncate_dim=DIMENSION
)

rag_data = load_dataset("neural-bridge/rag-dataset-1200")
rows = rag_data["train"].to_list()
rows = [r | {"id": str(i)} for i, r in enumerate(rows)]
samples = random.sample(rows, 50)

db = PicoVectorDB(embedding_dim=DIMENSION, storage_file="hfdata")

num_mismatches = 0
for sample in samples:
    if USE_QUESTION:
        emb = model.encode(sample["question"], prompt_name="query")
    else:
        if PASSAGE_LENGTH:
            context_length = len(sample["context"])
            idx = (
                random.randint(0, context_length - PASSAGE_LENGTH)
                if context_length > PASSAGE_LENGTH
                else 0
            )
            query = sample["context"][idx : idx + PASSAGE_LENGTH]
        else:
            query = sample["context"]
        emb = model.encode(query)
    results = db.query(emb, top_k=5, better_than=0.2)
    if results[0]["_id_"] != sample["id"]:
        print(
            "#mismatch#: expected id:", sample["id"], "actual id:", results[0]["_id_"]
        )
        print("#question#:", sample["question"])
        print("#expected answer#:", sample["answer"])
        print("#top answer#:", results[0]["answer"])
        if results[1]["_id_"] == sample["id"]:
            print("-- 2nd result DID match! --")
        else:
            print("Not 2nd result either:", results[1]["_id_"])
        num_mismatches += 1

print("total num of mismatches:", num_mismatches)
