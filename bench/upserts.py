import time
import numpy as np

import set_path
from picovdb import PicoVectorDB

# Configuration
DIMENSION = 1024  # Example dimension, adjust if needed
NUM_VECTORS_TO_INSERT = 100_000

print("Configuration:")
print(f"  Vector Dimension: {DIMENSION}")
print(f"  Vectors to Insert: {NUM_VECTORS_TO_INSERT:,}")
print("-" * 30)

# Initialize PicoVectorDB
print("Initializing PicoVectorDB...")
db = PicoVectorDB(embedding_dim=DIMENSION, storage_file="demo")
print("Initialization complete.")
print("-" * 30)

# --- Insertion Phase ---
print(f"Generating {NUM_VECTORS_TO_INSERT:,} random vectors for insertion...")
# Generate random vectors (replace with your actual data if needed)
vectors_to_insert = np.random.rand(NUM_VECTORS_TO_INSERT, DIMENSION).astype(np.float32)
# Generate simple IDs
ids_to_insert = [f"id_{i}" for i in range(NUM_VECTORS_TO_INSERT)]
print("Vector generation complete.")

print("Starting vector insertion...")
start_time_insert = time.time()
data = [
    {
        "_vector_": vectors_to_insert[i],
        "_id_": i,
        "content": f"data {i}",
    }
    for i in range(vectors_to_insert.shape[0])
]
db.upsert(data)
end_time_insert = time.time()
insertion_time = end_time_insert - start_time_insert
print("Insertion complete.")
print(
    f"Time taken for inserting {NUM_VECTORS_TO_INSERT:,} vectors: {insertion_time:.4f} seconds"
)
print(f"Vectors per second (insertion): {NUM_VECTORS_TO_INSERT / insertion_time:.2f}")
print(f"Current DB size: {len(db):,} vectors")
print("-" * 30)

db.save()
end_time_save = time.time()
saving_time = end_time_save - end_time_insert
print("Save complete.")
print(f"Time taken for save: {saving_time:.4f} seconds")


# --- Query Phase ---
# query_vector = np.random.rand(DIMENSION).astype(np.float32)
# result = db.query(query_vector, top_k=10, better_than=0.1)
# print(f"result: {result}")
