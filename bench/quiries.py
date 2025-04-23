import time
import numpy as np

from picovdb import PicoVectorDB

# Configuration
DIMENSION = 1024  # Example dimension, adjust if needed
NUM_QUERIES = 100
K_NEIGHBORS = 10  # Number of nearest neighbors to retrieve

print("Configuration:")
print(f"  Vector Dimension: {DIMENSION}")
print(f"  Number of Queries: {NUM_QUERIES:,}")
print(f"  K Neighbors: {K_NEIGHBORS}")
print("-" * 30)

db = PicoVectorDB(embedding_dim=DIMENSION, storage_file="demo")

# --- Query Phase ---
print(f"Generating {NUM_QUERIES:,} random query vectors...")
query_vectors = np.random.rand(NUM_QUERIES, DIMENSION).astype(np.float32)
print("Query vector generation complete.")

print(f"Starting {NUM_QUERIES:,} queries...")
start_time_query = time.time()
query_results = []
for i in range(NUM_QUERIES):
    results = db.query(query_vectors[i], top_k=K_NEIGHBORS, better_than=0.1)
    query_results.append(results)
end_time_query = time.time()
query_time = end_time_query - start_time_query
print("Querying complete.")
print(f"Time taken for {NUM_QUERIES:,} queries: {query_time:.4f} seconds")
print(f"Queries per second: {NUM_QUERIES / query_time:.2f}")
print("-" * 30)

# Optional: Print a sample query result
if query_results:
    print("Sample query result (first query):")
    # for score, doc_id in query_results[0]:
    #    print(f'  ID: {doc_id}, Score: {score:.4f}')
    print(f"result 0: {query_results[0]}")
else:
    print("No query results to display.")

print("\nPerformance test finished.")
