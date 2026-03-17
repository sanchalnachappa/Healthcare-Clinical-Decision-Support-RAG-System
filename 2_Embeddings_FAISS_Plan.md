# Person 2: Embeddings + FAISS — Notebook Implementation Plan

## Overview

This notebook implements **Section 6 (Step 2: Embeddings and Vector Index)** of the MediQuery paper. It takes the `all_chunks.json` file (8,563 chunks produced by the chunking notebook) as input, encodes every chunk into a dense vector using `BAAI/bge-base-en-v1.5`, builds a FAISS index, saves all artifacts to Google Drive, and runs a sanity-check retrieval test.

**Environment:** Google Colab Pro (GPU runtime recommended for faster encoding)
**Input:** `all_chunks.json` on Google Drive
**Outputs:** FAISS index file, document-ID mapping file, NumPy embeddings array — all saved to Google Drive

---

## Cell-by-Cell Plan

### Cell 1 — Markdown: Section Title & Intro

Write a markdown cell explaining:
- This is Step 2 of the MediQuery RAG pipeline.
- Goal: convert the 8,563 text chunks from `all_chunks.json` into dense 768-dimensional vectors using `BAAI/bge-base-en-v1.5`, then index them with FAISS for fast similarity search.
- Briefly explain why dense embeddings are needed (semantic matching vs. keyword matching — reference the Semantic_Search_2026 class notebook concepts).

---

### Cell 2 — Install Dependencies

```python
!pip install -q sentence-transformers faiss-cpu
```

Notes:
- `sentence-transformers` provides the `SentenceTransformer` wrapper for loading `BAAI/bge-base-en-v1.5`.
- `faiss-cpu` is sufficient; use `faiss-gpu` only if Colab GPU runtime is available and you want faster index operations (encoding is the bottleneck, not indexing).

---

### Cell 3 — Mount Google Drive & Set Paths

```python
from google.colab import drive
drive.mount('/content/drive', force_remount=True)
```

Define all path constants:
```python
DATA_DIR     = '/content/drive/MyDrive/GenAI_project/Data'
CHUNKS_FILE  = f'{DATA_DIR}/all_chunks.json'
OUTPUT_DIR   = f'{DATA_DIR}/faiss_index'
INDEX_FILE   = f'{OUTPUT_DIR}/medicare.index'
DOCID_FILE   = f'{OUTPUT_DIR}/docids.txt'
EMBED_FILE   = f'{OUTPUT_DIR}/embeddings.npy'   # optional: save raw embeddings
```

Create the output directory if it doesn't exist:
```python
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)
```

---

### Cell 4 — Load Chunks from JSON

```python
import json

with open(CHUNKS_FILE, 'r', encoding='utf-8') as f:
    all_chunks = json.load(f)

print(f"Total chunks loaded: {len(all_chunks)}")
```

Print a breakdown by type to confirm data integrity:
```python
from collections import Counter
type_counts = Counter(c['type'] for c in all_chunks)
for t, n in type_counts.items():
    print(f"  {t}: {n}")
```

Expected output (from chunking notebook):
- NCD: 575
- LCD: 2,802
- BENEFIT_POLICY: 1,039
- CLAIMS_PROCESSING: 4,147
- **Total: 8,563**

---

### Cell 5 — Markdown: Embedding Model Choice

Explain:
- **Model:** `BAAI/bge-base-en-v1.5` from HuggingFace.
- **Why this model:**
  - SOTA on the MTEB benchmark for retrieval tasks (Xiao et al., 2023).
  - 768-dimensional output — good balance of quality and efficiency.
  - Open-source and free.
  - Optimized for **asymmetric retrieval** (short query vs. long passage) — exactly our setting.
  - The BGE family uses a query instruction prefix (`"Represent this sentence: "`) to distinguish query vs. passage embeddings. For **passages**, no prefix is needed. For **queries** at retrieval time, we prepend the instruction.
- **Embedding dimension:** 768 (float32) → each chunk = 3,072 bytes → ~8,563 chunks ≈ 25 MB index.

---

### Cell 6 — Load the Embedding Model

```python
from sentence_transformers import SentenceTransformer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

model = SentenceTransformer('BAAI/bge-base-en-v1.5', device=device)
print(f"Embedding dimension: {model.get_sentence_embedding_dimension()}")
```

Expected output: `Embedding dimension: 768`

---

### Cell 7 — Extract Texts & Encode All Chunks

Extract the `text` field from each chunk (this is the metadata-tagged version, which includes the `[TYPE: ... | TITLE: ...]` header — embed the full tagged text so the embedding captures both content and metadata context).

```python
import numpy as np
from tqdm import tqdm

texts = [chunk['text'] for chunk in all_chunks]

print(f"Encoding {len(texts)} chunks...")
embeddings = model.encode(
    texts,
    batch_size=64,
    show_progress_bar=True,
    normalize_embeddings=True,   # L2-normalize so inner product = cosine similarity
    convert_to_numpy=True
)

print(f"Embeddings shape: {embeddings.shape}")  # Expected: (8563, 768)
print(f"Dtype: {embeddings.dtype}")
```

Key decisions:
- **`normalize_embeddings=True`**: Critical — this means we can use `IndexFlatIP` (inner product) which is equivalent to cosine similarity on normalized vectors. This follows the same pattern used in `Semantic_Search_2026.ipynb` (Cell 20: `faiss.IndexFlatIP(dimension)`).
- **`batch_size=64`**: Balances GPU memory and speed. Increase to 128 if GPU memory permits.
- **Embed `text` (with metadata header), not `raw_text`**: The metadata header (e.g., `[TYPE: LCD | LCD_ID: 35125 | TITLE: Wound Care | STATES: TX, AR...]`) is semantically informative. Embedding it means a query about "wound care Texas" can match on both the policy content AND the metadata tags.

---

### Cell 8 — Build the FAISS Index

```python
import faiss

dimension = embeddings.shape[1]  # 768
print(f"Building FAISS IndexFlatIP with dimension {dimension}...")

index = faiss.IndexFlatIP(dimension)
index.add(embeddings.astype('float32'))

print(f"Index contains {index.ntotal} vectors")
```

Why `IndexFlatIP`:
- **Exact search** (no approximation) — with only ~8.5K vectors, brute-force is fast enough (~1–5ms per query).
- **Inner product** on L2-normalized vectors = **cosine similarity**.
- No training required (unlike IVF or HNSW indexes).
- This matches the approach in `Semantic_Search_2026.ipynb` Cell 20.

---

### Cell 9 — Save Index + Doc IDs + Embeddings to Drive

```python
# Save the FAISS index
faiss.write_index(index, INDEX_FILE)
print(f"FAISS index saved: {INDEX_FILE}")

# Save document IDs (preserve mapping: index position → chunk metadata)
with open(DOCID_FILE, 'w', encoding='utf-8') as f:
    for chunk in all_chunks:
        # Store type, source_id, and chunk_idx for traceability
        f.write(f"{chunk['type']}|{chunk['source_id']}|{chunk['chunk_idx']}\n")
print(f"Doc IDs saved: {DOCID_FILE}")

# Save raw embeddings (optional, useful for debugging/reranking experiments)
np.save(EMBED_FILE, embeddings)
print(f"Embeddings saved: {EMBED_FILE}")

# Report file sizes
for path in [INDEX_FILE, DOCID_FILE, EMBED_FILE]:
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  {os.path.basename(path)}: {size_mb:.1f} MB")
```

Also save the full chunk metadata as a companion JSON for retrieval lookups:
```python
META_FILE = f'{OUTPUT_DIR}/chunk_metadata.json'
metadata = []
for i, chunk in enumerate(all_chunks):
    metadata.append({
        'idx': i,
        'type': chunk['type'],
        'source_id': chunk['source_id'],
        'title': chunk['title'],
        'states': chunk.get('states', ['ALL']),
        'contractor': chunk.get('contractor', ''),
        'filename': chunk['filename'],
        'chunk_idx': chunk['chunk_idx']
    })

with open(META_FILE, 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)
print(f"Chunk metadata saved: {META_FILE}")
```

---

### Cell 10 — Markdown: Index Statistics Summary

Print a summary table:

| Metric | Value |
|---|---|
| Total chunks indexed | 8,563 |
| Embedding model | BAAI/bge-base-en-v1.5 |
| Embedding dimension | 768 |
| Normalization | L2-normalized (cosine sim via inner product) |
| FAISS index type | IndexFlatIP (exact brute-force) |
| Index file size | ~25 MB |
| Embeddings file size | ~25 MB |

---

### Cell 11 — Retrieval Sanity Test: Helper Function

Create a reusable search function:

```python
def search_faiss(query, model, index, all_chunks, top_k=5):
    """Embed a query and return top-k chunks from the FAISS index."""
    # BGE models recommend a query instruction prefix for retrieval
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
        convert_to_numpy=True
    ).reshape(1, -1).astype('float32')

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
        chunk = all_chunks[idx]
        results.append({
            'rank': rank + 1,
            'score': float(score),
            'type': chunk['type'],
            'source_id': chunk['source_id'],
            'title': chunk['title'],
            'states': chunk.get('states', ['ALL']),
            'text_preview': chunk['text'][:300]
        })
    return results
```

---

### Cell 12 — Retrieval Sanity Test: Run Sample Queries

Run 3 sample queries that cover different document types and demonstrate the index works:

**Query 1 — NCD query (national coverage):**
```python
query = "Does Medicare cover acupuncture for chronic lower back pain?"
results = search_faiss(query, model, index, all_chunks, top_k=5)
```
- **Expected:** Top results should include NCD 30.1 (Acupuncture for Chronic Low Back Pain).

**Query 2 — LCD query (state-specific):**
```python
query = "Is home oxygen therapy covered for patients in Texas?"
results = search_faiss(query, model, index, all_chunks, top_k=5)
```
- **Expected:** Should retrieve LCD chunks tagged with TX/Novitas for oxygen equipment.

**Query 3 — Manual/Benefit Policy query:**
```python
query = "What are the requirements for skilled nursing facility coverage?"
results = search_faiss(query, model, index, all_chunks, top_k=5)
```
- **Expected:** Should retrieve BENEFIT_POLICY chunks from Chapter 8 (SNF Coverage).

For each query, print results in a clear format:
```python
def print_results(query, results):
    print(f"\nQuery: \"{query}\"")
    print("-" * 80)
    for r in results:
        print(f"  #{r['rank']} [score={r['score']:.4f}] [{r['type']}] "
              f"{r['title']} (ID: {r['source_id']})")
        print(f"     States: {r['states']}")
        print(f"     Preview: {r['text_preview'][:150]}...")
        print()
```

---

### Cell 13 — Markdown: Observations & Next Steps

Summarize:
- Confirm that the FAISS index retrieves semantically relevant chunks for diverse query types (NCD, LCD, Benefit Policy, Claims Processing).
- Note any observations (e.g., "The acupuncture query correctly surfaces NCD 30.1 as the top result").
- Note limitations of dense retrieval alone (no state filtering yet, no reranking — those are Person 3's job).
- State that the saved artifacts (`medicare.index`, `docids.txt`, `chunk_metadata.json`, `embeddings.npy`) will be loaded by the next notebook (Person 3: Retrieval + Reranking).

---

## Key References for the Notebook

- **Chunking notebook** (`Chunking_GENAI_Project.ipynb`): Produced `all_chunks.json` with 8,563 chunks. Each chunk has `text` (metadata-tagged), `raw_text`, `type`, `source_id`, `title`, `states`, etc.
- **Semantic_Search_2026.ipynb** (class reference): Demonstrates the FAISS workflow pattern:
  - Cell 18: imports (`SentenceTransformer`, `faiss`, `numpy`, `torch`)
  - Cell 20: `create_faiss_index()` — encode corpus, create `IndexFlatIP`, `faiss.write_index()`
  - Cell 22: `retrieve_faiss()` — load index, encode query, `index.search()`
  - Cell 25: device selection (CUDA vs CPU)
  - Cell 42: multi-model evaluation loop
- **Paper Section 6 requirements** (from PDF p.18):
  1. Explain choice of `BAAI/bge-base-en-v1.5`
  2. Describe how all chunks were encoded into dense vectors
  3. Explain FAISS and why `IndexFlatIP` was used
  4. Show the embedding + indexing code
  5. Report index size and embedding dimension (768)
  6. Include a brief retrieval test: sample query + top-3 results before reranking

## Data Schema Reminder (from `all_chunks.json`)

Each chunk object has these fields:
```json
{
  "text":       "[TYPE: LCD | LCD_ID: 35125 | ...]\n\nMedicare covers...",
  "raw_text":   "Medicare covers wound care when the patient...",
  "source_id":  "35125",
  "title":      "Wound Care",
  "type":       "LCD",           // NCD | LCD | BENEFIT_POLICY | CLAIMS_PROCESSING
  "states":     ["TX", "AR", "CO", "LA"],
  "contractor": "Novitas",       // LCD only
  "filename":   "Wound_Care_35125.txt",
  "chunk_idx":  0
}
```
