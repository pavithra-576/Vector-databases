# 🗃️ Vector Databases

> The memory layer for AI — store, index, and search high-dimensional embeddings at scale.

[![Docs](https://img.shields.io/badge/topic-vector--search-blue)]()
[![AI](https://img.shields.io/badge/use--case-RAG%20%7C%20semantic--search%20%7C%20recommendation-green)]()

---

## 📖 What is a Vector Database?

A **Vector Database** is a specialized database designed to store and query **high-dimensional vectors** (embeddings) — numerical representations of text, images, audio, or any data produced by ML models.

Unlike traditional databases that match exact values, vector databases perform **approximate nearest neighbor (ANN)** search — finding the most *semantically similar* items to a given query vector.

```
"What is the capital of France?"
         │
         ▼ Embedding Model
[0.23, -0.71, 0.88, 0.04, ..., 0.56]  ← 1536-dimensional vector
         │
         ▼ ANN Search in Vector DB
Top-K most similar stored vectors → relevant documents
```

---

## ❓ Why Vector Databases?

| Traditional DB | Vector DB |
|---|---|
| Exact keyword match | Semantic / meaning-based match |
| SQL queries | Similarity search (cosine, dot product, L2) |
| Structured data | Unstructured data (text, image, audio) |
| Row lookups | Nearest neighbor retrieval |
| Poor at fuzzy matching | Excellent at fuzzy / conceptual matching |

---

## 🔑 Core Concepts

### 📐 Embeddings
Dense numerical vectors output by an ML model that encode semantic meaning.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")
embedding = model.encode("The cat sat on the mat")
# Shape: (384,) — a 384-dimensional vector
```

### 📏 Similarity Metrics

| Metric | Formula | Best For |
|---|---|---|
| **Cosine Similarity** | angle between vectors | Text / NLP |
| **Dot Product** | magnitude + direction | Normalized embeddings |
| **Euclidean (L2)** | straight-line distance | Image embeddings |

### 🔍 ANN Algorithms
Exact search is O(n) — too slow at scale. ANN algorithms trade tiny accuracy loss for massive speed gains:

| Algorithm | Used By | Notes |
|---|---|---|
| **HNSW** | Weaviate, Qdrant, pgvector | Hierarchical graph, best speed/accuracy |
| **IVF** | FAISS | Inverted file index, fast at large scale |
| **ScaNN** | Google | Optimized for billion-scale |
| **Annoy** | Spotify | Tree-based, read-optimized |
| **DiskANN** | Microsoft | Disk-based, handles massive datasets |

---

## 🗂️ Major Vector Databases

### ⚡ FAISS (Facebook AI Similarity Search)

**Type:** Library (in-memory / on-disk)
**Best for:** Local development, prototyping, research

```bash
pip install faiss-cpu   # or faiss-gpu
```

```python
import faiss
import numpy as np

dim = 384
index = faiss.IndexFlatL2(dim)           # Exact L2 search

vectors = np.random.rand(1000, dim).astype("float32")
index.add(vectors)                        # Add 1000 vectors

query = np.random.rand(1, dim).astype("float32")
distances, indices = index.search(query, k=5)   # Top-5 results
print(indices)
```

**With LangChain:**
```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

vectorstore = FAISS.from_documents(docs, OpenAIEmbeddings())
vectorstore.save_local("faiss_index")
```

---

### 🎨 Chroma

**Type:** Open-source, embeddable / server mode
**Best for:** Local RAG apps, quick prototyping

```bash
pip install chromadb
```

```python
import chromadb

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.create_collection("my_docs")

collection.add(
    documents=["LangChain is a framework", "RAG grounds LLMs with data"],
    ids=["doc1", "doc2"]
)

results = collection.query(
    query_texts=["What is LangChain?"],
    n_results=2
)
print(results["documents"])
```

**With LangChain:**
```python
from langchain_community.vectorstores import Chroma

vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)
```

---

### 📌 Pinecone

**Type:** Fully managed cloud vector database
**Best for:** Production, serverless, large-scale RAG

```bash
pip install pinecone-client
```

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="YOUR_API_KEY")

pc.create_index(
    name="my-index",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(cloud="aws", region="us-east-1")
)

index = pc.Index("my-index")

index.upsert(vectors=[
    {"id": "doc1", "values": embedding1, "metadata": {"source": "file.pdf"}},
    {"id": "doc2", "values": embedding2, "metadata": {"source": "web"}},
])

results = index.query(vector=query_embedding, top_k=5, include_metadata=True)
```

---

### 🔷 Qdrant

**Type:** Open-source, self-hosted or cloud
**Best for:** High-performance production, rich filtering

```bash
pip install qdrant-client
docker run -p 6333:6333 qdrant/qdrant
```

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient("localhost", port=6333)

client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(size=384, distance=Distance.COSINE)
)

client.upsert(
    collection_name="documents",
    points=[
        PointStruct(id=1, vector=embedding, payload={"text": "sample doc"})
    ]
)

results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    limit=5
)
```

---

### 🕸️ Weaviate

**Type:** Open-source, cloud-native, hybrid search
**Best for:** Hybrid (vector + keyword) search, GraphQL API

```bash
pip install weaviate-client
```

```python
import weaviate

client = weaviate.connect_to_local()

collection = client.collections.create(
    name="Document",
    properties=[
        weaviate.classes.config.Property(name="text", data_type=weaviate.classes.config.DataType.TEXT)
    ]
)

collection.data.insert({"text": "Weaviate supports hybrid search"})

results = collection.query.near_text(query="semantic search", limit=5)
for obj in results.objects:
    print(obj.properties)

client.close()
```

---

### 🐘 pgvector (PostgreSQL)

**Type:** PostgreSQL extension
**Best for:** Teams already using PostgreSQL, avoiding a new DB

```bash
pip install pgvector psycopg2-binary
```

```sql
-- Install extension
CREATE EXTENSION vector;

-- Create table with vector column
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT,
    embedding vector(1536)
);

-- Insert
INSERT INTO documents (content, embedding) VALUES ('Hello world', '[0.1, 0.2, ...]');

-- Similarity search
SELECT content, 1 - (embedding <=> '[0.1, 0.2, ...]') AS similarity
FROM documents
ORDER BY embedding <=> '[0.1, 0.2, ...]'
LIMIT 5;
```

**With LangChain:**
```python
from langchain_community.vectorstores import PGVector

vectorstore = PGVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    connection_string="postgresql://user:password@localhost/mydb"
)
```

---

## 📊 Comparison Table

| | FAISS | Chroma | Pinecone | Qdrant | Weaviate | pgvector |
|---|---|---|---|---|---|---|
| **Type** | Library | Embedded/Server | Cloud SaaS | OSS + Cloud | OSS + Cloud | PG Extension |
| **Hosting** | Local | Local/Server | Managed | Self/Cloud | Self/Cloud | Self/Managed |
| **Scale** | Medium | Small-Medium | Large | Large | Large | Medium-Large |
| **Hybrid Search** | ❌ | ❌ | ❌ | ✅ | ✅ | ✅ (with FTS) |
| **Metadata Filter** | Limited | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Persistence** | Manual | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Best For** | Prototyping | Dev/RAG | Production | Performance | Hybrid search | Existing PG |
| **Cost** | Free | Free | Paid (free tier) | Free/Paid | Free/Paid | Free |

---

## 🔧 Advanced Features

### Metadata Filtering
Filter results by attributes alongside vector similarity:

```python
# Qdrant example
from qdrant_client.models import Filter, FieldCondition, MatchValue

results = client.search(
    collection_name="documents",
    query_vector=query_embedding,
    query_filter=Filter(
        must=[FieldCondition(key="source", match=MatchValue(value="pdf"))]
    ),
    limit=5
)
```

### Namespace / Tenancy
Isolate data per user or project (Pinecone namespaces, Qdrant collections):

```python
# Pinecone namespace per user
index.upsert(vectors=[...], namespace="user_123")
results = index.query(vector=q, namespace="user_123", top_k=5)
```

### Hybrid Search (Vector + BM25)
```python
# Weaviate hybrid search
results = collection.query.hybrid(
    query="machine learning",
    alpha=0.5,   # 0 = pure keyword, 1 = pure vector
    limit=5
)
```

---

## 🏆 Best Practices

- ✅ Choose **chunk size** carefully (256–512 tokens is a good default)
- ✅ Always store **metadata** (source, date, page) for filtering and citations
- ✅ Use **HNSW** index for best speed/accuracy tradeoff in production
- ✅ Normalize vectors before using **dot product** similarity
- ✅ Use **namespaces/tenancy** for multi-user applications
- ✅ Benchmark your retrieval with **RAGAS** or custom evals
- ❌ Don't store raw documents only — always include metadata
- ❌ Don't use exact search (IndexFlatL2) at scale — switch to IVF/HNSW

---

## 📚 Resources

- 📘 [FAISS Docs](https://faiss.ai)
- 📘 [Chroma Docs](https://docs.trychroma.com)
- 📘 [Pinecone Docs](https://docs.pinecone.io)
- 📘 [Qdrant Docs](https://qdrant.tech/documentation)
- 📘 [Weaviate Docs](https://weaviate.io/developers/weaviate)
- 📘 [pgvector GitHub](https://github.com/pgvector/pgvector)
- 📖 [ANN Benchmarks](https://ann-benchmarks.com)
