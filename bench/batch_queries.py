import sys
import time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))
from picovdb import PicoVectorDB

# Configuration
DIMENSION = 1024  # Example dimension, adjust if needed
NUM_BATCHES = 20 
NUM_QUERIES_PER_BATCH = 50
K_NEIGHBORS = 10  # Number of nearest neighbors to retrieve

print("Configuration:")
print(f"  Vector Dimension: {DIMENSION}")
print(f"  Number of batches: {NUM_BATCHES:,}")
print(f"  Number of Queries per batch: {NUM_QUERIES_PER_BATCH:,}")
print(f"  K Neighbors: {K_NEIGHBORS}")
print("-" * 30)

db = PicoVectorDB(embedding_dim=DIMENSION, storage_file="demo")

# --- Query Phase ---
num_queries = NUM_BATCHES * NUM_QUERIES_PER_BATCH
print(f"Generating {num_queries:,} random query vectors...")
query_vectors = np.random.rand(num_queries, DIMENSION).astype(np.float32)
print("Query vector generation complete.")

print(f"Starting {num_queries:,} queries...")
start_time_query = time.time()
query_results = []
for i in range(NUM_BATCHES):
    batch = np.array(query_vectors[i * NUM_QUERIES_PER_BATCH: (i + 1) * NUM_QUERIES_PER_BATCH])
    #batch = np.array(query_vectors[i * NUM_QUERIES_PER_BATCH: (i + 1) * NUM_QUERIES_PER_BATCH], dtype=np.float32)
    #batch = np.stack(query_vectors[i * NUM_QUERIES_PER_BATCH: (i + 1) * NUM_QUERIES_PER_BATCH], axis=0)
    results = db.query(batch, top_k=K_NEIGHBORS, better_than=0.1)
    query_results.extend(results)
end_time_query = time.time()
query_time = end_time_query - start_time_query
print("Querying complete.")
print(f"Time taken for {num_queries:,} queries: {query_time:.4f} seconds")
print(f"Queries per second: {num_queries / query_time:.2f}")
print("-" * 30)

# Optional: Print a sample query result
if query_results:
    print('Number of query results:', len(query_results))
    print("Sample query result (first query):")
    # for score, doc_id in query_results[0]:
    #    print(f'  ID: {doc_id}, Score: {score:.4f}')
    print(f"result 0: {query_results[0]}")
else:
    print("No query results to display.")

print("\nPerformance test finished.")
