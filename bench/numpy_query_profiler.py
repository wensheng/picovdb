import time
import numpy as np
import cProfile
import pstats
import io
import os
import argparse
import set_path
from picovdb import PicoVectorDB, K_ID, K_VECTOR

# Configuration
DIMENSION = 1024  # Default dimension
NUM_QUERIES_TO_PROFILE = 50 # Number of random queries to run for profiling each scenario
K_NEIGHBORS = 10

def get_db_paths(base_path: str):
    """Helper to get all file paths associated with a DB base name."""
    return [
        f"{base_path}.ids.json",
        f"{base_path}.meta.json",
        f"{base_path}.vecs.npy",
        f"{base_path}.vecs.npy.faiss", # Though we're not using faiss, it might be created if logic changes
    ]

def cleanup_db_files(base_path: str):
    """Remove database files."""
    print(f"Cleaning up DB files for base: {base_path}")
    for p in get_db_paths(base_path):
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError as e:
                print(f"Error removing file {p}: {e}")

def setup_db(db_size: int, storage_file_base: str) -> PicoVectorDB:
    """Initializes and populates PicoVectorDB with no_faiss=True."""
    print(f"\n--- Setting up DB: {storage_file_base} with {db_size:,} vectors ---")
    
    # Ensure a clean state for this specific DB size if files exist
    cleanup_db_files(storage_file_base)

    db = PicoVectorDB(
        embedding_dim=DIMENSION,
        storage_file=storage_file_base,
        no_faiss=True # Crucial for NumPy path
    )

    print(f"Generating {db_size:,} random vectors for insertion...")
    vectors_to_insert = np.random.rand(db_size, DIMENSION).astype(np.float32)
    
    data_to_upsert = []
    for i in range(db_size):
        data_to_upsert.append({
            K_VECTOR: vectors_to_insert[i],
            K_ID: f"id_{i}",
            "content": f"Document content for id_{i}",
            "category_id": i % 10 # For 'where' clause testing
        })
    
    print("Starting vector insertion...")
    insert_start_time = time.time()
    db.upsert(data_to_upsert)
    insert_end_time = time.time()
    print(f"Insertion of {db_size:,} vectors took {insert_end_time - insert_start_time:.4f} seconds.")
    # db.save() # Optional: save if you want to inspect/reuse, but adds time
    print(f"DB setup complete. Current DB size: {len(db):,} vectors.")
    return db

def run_query_scenario(
    db: PicoVectorDB,
    query_vector: np.ndarray,
    scenario_name: str,
    profile: bool = True,
    where_fn: callable = None,
    better_than_val: float = None
):
    """Runs a single query under specified conditions and profiles it."""
    print(f"\n  Scenario: {scenario_name}")
    
    pr = None
    if profile:
        pr = cProfile.Profile()
        pr.enable()

    start_time = time.time()
    results = db.query(query_vector, top_k=K_NEIGHBORS, where=where_fn, better_than=better_than_val)
    end_time = time.time()

    if profile and pr:
        pr.disable()
        s = io.StringIO()
        sortby = pstats.SortKey.CUMULATIVE # or .TIME
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats(20) # Print top 20 functions
        print(s.getvalue())

    print(f"  Query time for '{scenario_name}': {end_time - start_time:.6f} seconds. Found {len(results)} results.")
    return results

def main(db_sizes_to_test: list[int]):
    print("Starting NumPy Query Path Profiling")
    print("=" * 40)

    for db_size in db_sizes_to_test:
        db_file_base = f"temp_db_{db_size}"
        db = None
        try:
            db = setup_db(db_size, db_file_base)
            
            print(f"\n--- Profiling Queries for DB size: {db_size:,} ---")
            query_vectors = np.random.rand(NUM_QUERIES_TO_PROFILE, DIMENSION).astype(np.float32)

            for i in range(NUM_QUERIES_TO_PROFILE):
                print(f"\n-- Profiling Query {i+1}/{NUM_QUERIES_TO_PROFILE} for DB size {db_size:,} --")
                current_query_vector = query_vectors[i]

                # Scenario 1: Baseline (no filters)
                run_query_scenario(db, current_query_vector, "Baseline (No Filters)")

                # Scenario 2: With 'where' filter
                simple_where_fn = lambda meta: meta.get('category_id', -1) % 2 == 0
                run_query_scenario(db, current_query_vector, "With 'where' filter (category_id % 2 == 0)", where_fn=simple_where_fn)

                # Scenario 3: With 'better_than' filter
                run_query_scenario(db, current_query_vector, "With 'better_than' filter (0.1)", better_than_val=0.1)
                
                # Scenario 4: With 'where' AND 'better_than'
                run_query_scenario(db, current_query_vector, "With 'where' AND 'better_than'", where_fn=simple_where_fn, better_than_val=0.1)

        finally:
            if db: # Ensure DB files are cleaned up even if errors occur
                # db.save() # If you want to persist for inspection
                pass
            cleanup_db_files(db_file_base)
            print(f"--- Finished processing for DB size: {db_size:,} ---")
            print("=" * 40)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile PicoVectorDB NumPy query path.")
    parser.add_argument(
        "--db_sizes",
        type=str,
        default="10000,50000", # Smaller default for quick tests
        help="Comma-separated list of database sizes to test (e.g., 10000,50000,100000)."
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=DIMENSION,
        help=f"Dimension of vectors (default: {DIMENSION})"
    )
    parser.add_argument(
        "--num_queries",
        type=int,
        default=NUM_QUERIES_TO_PROFILE,
        help=f"Number of queries to profile per DB size and scenario (default: {NUM_QUERIES_TO_PROFILE})"
    )
    args = parser.parse_args()

    DIMENSION = args.dim
    NUM_QUERIES_TO_PROFILE = args.num_queries
    
    try:
        db_sizes = [int(s.strip()) for s in args.db_sizes.split(',')]
    except ValueError:
        print("Error: Invalid format for --db_sizes. Please use comma-separated integers.")
        exit(1)
        
    main(db_sizes)
    print("\nProfiling script finished.")