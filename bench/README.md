# Picovdb Performance Test Bench

## Files

1. upserts.py
   * `python upserts.py`
   * a 100k vector database will be created
   * This will take 2 seconds if faiss is not enabled (`pip uninstall faiss-cpu`), otherwise this will take ~2 minutes.
2. queries.py
   * `python queries`
   * will do 100 single queries.
3. batch_queries.py
   * `python batch_queries`
   * will do 20 batch (size 50) queries.
3. st_embedding.py
   * `python st_embedding.py`
   * use Huggingface embedding with sentence-transformer
   * Store and query text from 'A Christmas Carol'
4. numpy_query_profiler.py
   * `python bench/numpy_query_profiler.py --db_sizes "10000,100000,250000"` 
   * profile query speed with different db size 
   * example:
     - `python numpy_query_profiler.py --db_sizes 10000,30000 --num_queries 5 --dim 256 --batch_sizes 1,8,32 --top_k 10 --csv out_summary.csv --json out_summary.json`
