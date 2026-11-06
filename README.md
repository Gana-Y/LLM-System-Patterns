<div align="center">

  <h1>🧠 LLM System Patterns</h1>

  <p><strong>A comprehensive, production-grade guide to designing scalable, low-latency, and cost-efficient LLM architectures, RAG pipelines, and autonomous AI agents.</strong></p>

  <p>
    <img src="https://img.shields.io/badge/System%20Design-LLM%20%26%20Agents-blueviolet?style=for-the-badge&logo=openai" alt="System Design" />
    <img src="https://img.shields.io/badge/Production-Battle--Tested-success?style=for-the-badge&logo=fastapi" alt="Production Ready" />
    <img src="https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python" alt="Python Version" />
    <img src="https://img.shields.io/badge/License-MIT-blue?style=for-the-badge" alt="License" />
  </p>

  <p>
    <a href="#-the-master-architecture-deep-dive"><img src="https://img.shields.io/badge/Architecture-Master%20Deep%20Dive-blue?style=flat-square" /></a>
    <a href="#-the-8-core-production-patterns"><img src="https://img.shields.io/badge/Catalog-8%20Core%20Patterns-blueviolet?style=flat-square" /></a>
    <a href="#-production-latency--cost-budgets"><img src="https://img.shields.io/badge/Engineering-Latency%20%26%20Cost%20Math-orange?style=flat-square" /></a>
    <a href="#-the-production-disaster-manual"><img src="https://img.shields.io/badge/Reliability-Disaster%20Manual-red?style=flat-square" /></a>
    <a href="#-real-world-case-studies"><img src="https://img.shields.io/badge/Interviews-3%20Case%20Studies-brightgreen?style=flat-square" /></a>
    <img src="https://img.shields.io/badge/PRs-Welcome-brightgreen.svg?style=flat-square" alt="PRs Welcome" />
  </p>

  <sub>Star ⭐ this repository to bookmark this production manual for system design interviews and enterprise architectures!</sub>

</div>

---

## 📑 Table of Contents

- [💡 Why LLM System Patterns?](#-why-llm-system-patterns)
- [🏗️ The Master Architecture: Complete Production Lifecycle](#️-the-master-architecture-complete-production-lifecycle)
- [🔍 Master Architecture Deep Dive (Node-by-Node Breakdown)](#-master-architecture-deep-dive-node-by-node-breakdown)
  - [Step 1: User Query & Two-Tier Semantic Caching](#step-1-user-query--two-tier-semantic-caching)
  - [Step 2: Query Transformation via HyDE (Hypothetical Document Embeddings)](#step-2-query-transformation-via-hyde)
  - [Step 3: The Hybrid Retrieval Engine (Dense + Sparse)](#step-3-the-hybrid-retrieval-engine-dense--sparse)
  - [Step 4: Reciprocal Rank Fusion (RRF)](#step-4-reciprocal-rank-fusion-rrf)
  - [Step 5: Cross-Encoder Deep Attention Reranker](#step-5-cross-encoder-deep-attention-reranker)
  - [Step 6: Hierarchical Parent-Child Context Packing](#step-6-hierarchical-parent-child-context-packing)
  - [Step 7: Self-Corrective RAG (CRAG) Guardrails](#step-7-self-corrective-rag-crag-guardrails)
  - [Step 8: Prefix Prompt Caching & Streaming Generation](#step-8-prefix-prompt-caching--streaming-generation)
- [🧩 The 8 Core Production Patterns](#-the-8-core-production-patterns)
  - [1. Hybrid Retrieval & Reciprocal Rank Fusion (RRF)](#1-hybrid-retrieval--reciprocal-rank-fusion-rrf)
  - [2. Hypothetical Document Embeddings (HyDE)](#2-hypothetical-document-embeddings-hyde)
  - [3. Two-Tier Semantic Caching & Token Routing](#3-two-tier-semantic-caching--token-routing)
  - [4. Hierarchical Parent-Child Chunking](#4-hierarchical-parent-child-chunking)
  - [5. Self-Corrective RAG (CRAG) & Active Guardrails](#5-self-corrective-rag-crag--active-guardrails)
  - [6. Stateful Multi-Agent Supervisor & Tool Calling](#6-stateful-multi-agent-supervisor--tool-calling)
  - [7. Prompt Prefix Caching & TTFT Optimization](#7-prompt-prefix-caching--ttft-optimization)
  - [8. Vector Index Selection: HNSW vs. IVFFlat vs. DiskANN](#8-vector-index-selection-hnsw-vs-ivfflat-vs-diskann)
- [⏱️ Production Latency & Cost Budgets](#️-production-latency--cost-budgets)
- [🚨 The Production Disaster Manual (Failure Modes & Fixes)](#-the-production-disaster-manual-failure-modes--fixes)
- [💻 Runnable 50-Line Python Primitives](#-runnable-50-line-python-primitives)
- [🎯 Real-World System Design Case Studies](#-real-world-system-design-case-studies)
  - [Case Study 1: Real-Time Multimodal AI Academic Engine](#case-study-1-real-time-multimodal-ai-academic-engine)
  - [Case Study 2: Hardened Sandboxed Code Execution Agent](#case-study-2-hardened-sandboxed-code-execution-agent)
  - [Case Study 3: Low-Latency Financial Order Intelligence Engine](#case-study-3-low-latency-financial-order-intelligence-engine)
- [⭐ Explore My Other Repositories](#-explore-my-other-repositories)
- [🤝 Contributing](#-contributing)
- [📜 License](#-license)

---

## 💡 Why LLM System Patterns?

Traditional software engineering operates on deterministic state machines and well-defined API boundaries. In contrast, **LLM architectures are probabilistic, token-metered, and latency-heavy**. 

Moving a prototype into an enterprise production environment with 99.9% uptime requires answering critical engineering questions:
* *How do we handle out-of-vocabulary acronyms, serial numbers, and code variables that dense vector embeddings miss?*
* *How do we keep end-to-end Time to First Token (TTFT) under $300\text{ms}$ while executing vector search, reranking, and safety checks?*
* *How do we eliminate context window overflows while reducing LLM API token billing by 85%?*

This repository acts as an engineering manual documenting battle-tested architectural patterns, real mathematical budgets, and failure recovery protocols required to build scalable AI systems.

---

## 🏗️ The Master Architecture: Complete Production Lifecycle

The diagram below maps the complete end-to-end lifecycle of an enterprise-grade RAG and Agentic LLM request:

```mermaid
flowchart TD
    UserQuery(["User Query"]) --> SemanticCache{"Semantic Cache Hit?"}
    SemanticCache -- "Yes (Similarity >= 0.94)" --> CacheReturn["Return Cached Response (~10ms)"]
    SemanticCache -- "No" --> QueryRewrite["HyDE / Query Expansion"]
    
    subgraph Retrieval_Stage ["Hybrid Retrieval Engine"]
        QueryRewrite --> DenseSearch["Dense Vector Search (HNSW)"]
        QueryRewrite --> SparseSearch["Sparse Keyword Search (BM25)"]
        DenseSearch --> RRF["Reciprocal Rank Fusion (RRF)"]
        SparseSearch --> RRF
    end
    
    RRF --> CrossEncoder["Cross-Encoder Reranker (Top 25 -> Top 5)"]
    CrossEncoder --> ContextPacker["Parent-Child Context Packing"]
    ContextPacker --> GuardrailCheck{"CRAG Evaluation"}
    GuardrailCheck -- "Low Confidence" --> WebSearchFallback["Fallback Search / Human in Loop"]
    GuardrailCheck -- "High Confidence" --> PromptCache["Prefix Prompt Cache Check"]
    PromptCache --> LLMInference["LLM Generation (Streaming)"]
    LLMInference --> UserResponse(["Streamed Tokens to Client"])
```

---

## 🔍 Master Architecture Deep Dive (Node-by-Node Breakdown)

To understand how high-throughput AI services operate in production, we break down each node of the master flowchart into its mechanical, mathematical, and operational components.

---

### Step 1: User Query & Two-Tier Semantic Caching

```
User Query ➔ [Tier 1: Exact Hash Lookup] ➔ [Tier 2: Vector Cosine Index] ➔ Hit/Miss Decision
```

#### The Problem:
In consumer and enterprise chat interfaces, between **$25\%$ and $40\%$** of all incoming requests are semantically identical (e.g., *"How do I reset my password?"*, *"Where to change password"*, or *"Forgot password"*). Routing every identical question through vector retrieval, cross-encoder reranking, and LLM inference generates hundreds of dollars in unnecessary API costs and forces users to wait $1\text{--}3\text{ seconds}$ for answers that already exist.

#### Production Implementation:
We enforce a **Two-Tier Cache Hierarchy**:
1. **Tier 1 (Exact Match - $O(1)$ in $<1\text{ms}$):**
   * Normalizes the string: `query = trim(lowercase(raw_query))`
   * Generates a deterministic SHA-256 hash: `hash = SHA256(query)`
   * Performs an in-memory Redis key-value lookup. If found, the answer is returned immediately.
2. **Tier 2 (Semantic Vector Similarity - $O(\log N)$ in $\sim 8\text{ms}$):**
   * If Tier 1 misses, the query is converted into a vector embedding $\vec{q}$.
   * We search an in-memory vector cache index in Redis or Qdrant containing previously answered queries.
   * Cosine similarity is computed against previous queries $\vec{k}$:
     $$S_C(\vec{q}, \vec{k}) = \frac{\vec{q} \cdot \vec{k}}{\|\vec{q}\|_2 \|\vec{k}\|_2}$$
   * **The Production Threshold ($\tau$):** In real production systems, $\tau$ is strictly calibrated between **$0.92$ and $0.95$**.
     * Setting $\tau < 0.90$ introduces catastrophic *false positives* (e.g., matching *"Can I cancel my account?"* with *"How do I upgrade my account?"*).
     * Setting $\tau \ge 0.94$ guarantees accurate semantic reuse.
     * **Cache Hit Result:** The cached completion is returned in **$\sim 10\text{ms}$ at $\$0.00$ incremental cost**.

---

### Step 2: Query Transformation via HyDE

```
Raw Query ("403 s3 cors") ➔ Fast SLM Generates Hypothetical Passage ➔ Embed Hypothetical Doc
```

#### The Problem: The Asymmetric Search Gap
User prompts are typically **short, informal, and keyword-starved** (e.g., *"aws s3 403 cors error"* — 5 words). 
Conversely, the technical documentation, runbooks, or source files containing the answer are **dense, formal, and explanatory** (500 words).

In high-dimensional embedding space, a 5-word question embedding occupies a completely different vector cluster than a 500-word explanatory answer. This divergence causes standard vector search to miss the most relevant technical documents.

#### The HyDE Solution:
1. Instead of embedding the raw query directly, we route the user query to a fast, cheap Small Language Model (SLM) such as `Llama-3-8B` or `Gemini-Flash` with a specialized prompt:
   > *"Write a concise, 1-paragraph technical documentation excerpt that directly answers this question: [User Query]. Do not worry about factual perfection; focus on technical terminology and syntax."*
2. The SLM outputs a hypothetical passage loaded with high-signal domain vocabulary: `AccessDenied`, `BucketPolicy`, `CORSRules`, `AllowedOrigins`, `OPTIONS preflight`.
3. Even if the passage contains minor factual inaccuracies, **its embedding vector lands directly inside the cluster of real production documentation**.
4. Real-world benchmark: HyDE boosts retrieval recall by **$30\text{--}35\%$** over raw query embeddings.

---

### Step 3: The Hybrid Retrieval Engine (Dense + Sparse)

```
                            ┌──> Dense Search: HNSW on Embeddings  (Semantic Meaning) ──┐
Hypothetical Query Vector ──┤                                                           ├──> RRF
Raw Text Query Tokens    ───┴──> Sparse Search: BM25 on Inverted Index (Exact Keywords) ─┘
```

#### Why Dense Embeddings Alone Fail in Production:
Dense vector search (Cosine similarity over vector embeddings) understands *conceptual meaning*, but it is blind to *exact character sequences*.
* If an engineer searches for a specific error code like `ERR_SOCKET_TIMEOUT_0x82` or an internal SKU `INV-9021-X`, dense embeddings smooth those characters into generic networking vectors.
* If a doctor searches for a specific drug dosage `"Zestril 20mg"`, dense search may pull up `"Lisinopril 10mg"` because they share the same therapeutic classification.

#### The Dual-Engine Solution:
Execute two retrieval algorithms in parallel:
1. **Dense Retrieval (HNSW):** 
   * Traverses a Hierarchical Navigable Small World graph in $O(\log N)$ time ($\sim 5\text{ms}$).
   * Matches conceptual context, intent, synonyms, and cross-lingual queries.
2. **Sparse Retrieval (BM25 / SPLADE):**
   * Uses an inverted index based on Term Frequency-Inverse Document Frequency (TF-IDF).
   * Matches exact tokens, serial numbers, variable names, and error codes.

---

### Step 4: Reciprocal Rank Fusion (RRF)

#### The Problem: The Incompatible Scale Dilemma
You cannot simply add BM25 scores to Cosine Similarity scores:
* **Cosine Similarity** is bounded: $[-1.0, +1.0]$.
* **BM25 Scores** are unbounded: they can be $3.2$, $19.4$, or $74.0$ depending on document length and term frequency across the corpus.

Summing raw scores without calibration causes BM25 to completely overpower the dense vector score.

#### The Mathematical Fix: Reciprocal Rank Fusion
RRF eliminates raw scores entirely. Instead, it evaluates the **rank position** of each document in both candidate lists:

$$RRF\_Score(d \in D) = \sum_{m \in M} \frac{1}{k + rank_m(d)}$$

Where:
* $M$ is the set of retrieval systems (Dense + Sparse).
* $rank_m(d)$ is the 1-based index position of document $d$ in system $m$.
* $k$ is a smoothing constant (standard industry default: $k = 60$).

#### Practical Example:
Suppose Document $A$ ranks **#1** in Dense and **#2** in Sparse:
$$Score(A) = \frac{1}{60 + 1} + \frac{1}{60 + 2} = \frac{1}{61} + \frac{1}{62} \approx 0.01639 + 0.01612 = 0.03251$$

Suppose Document $B$ ranks **#1** in Dense, but does not appear in the Sparse top 100:
$$Score(B) = \frac{1}{60 + 1} + 0 = 0.01639$$

Document $A$ automatically wins because it demonstrated dual-engine consensus. RRF outputs an optimal merged pool of the **Top 25 candidates**.

---

### Step 5: Cross-Encoder Deep Attention Reranker

```
Top 25 Candidates from RRF ➔ Cross-Encoder Transformer ➔ Top 5 Ultra-Relevant Chunks
```

#### Bi-Encoder vs. Cross-Encoder:
* **Bi-Encoder (Used in Vector Search):** Encodes the query and document *independently* into isolated vectors, then computes their dot product. It is lightning-fast, but lacks deep token-to-token interaction.
* **Cross-Encoder (Used in Reranking):** Feeds the query AND document *together* into a single transformer encoder:
  $$\text{Input} = [\text{CLS}] \circ \text{Query Tokens} \circ [\text{SEP}] \circ \text{Document Tokens} \circ [\text{SEP}]$$
  This allows self-attention heads to cross-examine every single query word against every single document word simultaneously.

#### The Production Latency Trade-Off:
Cross-encoders are computationally expensive: running one across $1{,}000{,}000$ documents would take 30 seconds. However, running a Cross-Encoder (such as `bge-reranker-large` or `cohere-rerank`) on just the **Top 25 candidates takes only $\sim 40\text{ms}$**.

It filters out the $20$ borderline or irrelevant passages and produces the **Top 5 ultra-high-signal chunks**, cutting context window bloat and eliminating hallucinations.

---

### Step 6: Hierarchical Parent-Child Context Packing

```
Matched 128-Token Child Chunk ➔ Retrieve Parent ID from DB ➔ Pack Full 1024-Token Parent Section
```

* **Child Chunks ($128$ tokens):** Stored in the vector database index. Small chunks maximize embedding resolution and produce high cosine similarity matches.
* **Parent Chunks ($1024$ tokens):** Stored in a relational database (PostgreSQL) or document store. Contains the full surrounding section, parent headers, and following context.
* **Execution:** When the Cross-Encoder selects the Top 5 child chunks, the application swaps them for their **Parent Chunks**. The LLM receives full, coherent documentation paragraphs rather than isolated 2-sentence fragments.

---

### Step 7: Self-Corrective RAG (CRAG) Guardrails

```
Retrieved Context ➔ Fast Evaluator Classifier:
  ├── [Confidence >= 0.85] ➔ High Quality: Proceed to LLM Generator
  ├── [0.50 <= Conf < 0.85] ➔ Ambiguous: Strip noisy sentences + Synthesize with hedge warnings
  └── [Confidence < 0.50]  ➔ Low Quality: Trigger Web Search Fallback or Escalate to Human
```

Naive RAG blindly feeds retrieved text into the LLM even when search quality is terrible, directly causing hallucinations. CRAG introduces an automated evaluation gatekeeper:
1. A fast classifier checks: *"Do these passages contain the direct factual evidence needed to answer the user query?"*
2. **Three-Way Branching:**
   * **Correct (Confidence $\ge 0.85$):** Passages are clean. Forward to LLM.
   * **Ambiguous ($0.50 \le \text{Conf} < 0.85$):** Passages contain partial information. Strip irrelevant sentences via sentence-level filtering and instruct the LLM to output hedge warnings.
   * **Incorrect ($\text{Confidence} < 0.50$):** Retrieval completely failed (the document doesn't exist in the knowledge base). Automatically trigger an external web search API (Tavily/Perplexity) or return: *"The knowledge base does not contain sufficient verified data to answer this question."*

---

### Step 8: Prefix Prompt Caching & Streaming Generation

```
Static System Prompt + Tool Schemas (Cached at 90% Discount) + Dynamic Context ➔ LLM Streaming
```

1. **Prefix Prompt Caching:**
   * Modern LLM APIs (Anthropic Prompt Caching & OpenAI Cached Tokens) allow developers to cache static prompt prefixes.
   * **The Cardinal Ordering Rule:** Always order prompts from **static to dynamic**:
     $$\text{System Prompt} \rightarrow \text{Tool Schemas} \rightarrow \text{Few-Shot Examples} \rightarrow \text{Retrieved Docs} \rightarrow \text{Chat History} \rightarrow \text{User Query}$$
   * The provider recognizes identical byte blocks in the KV-cache, providing up to a **$90\%$ discount on input tokens** and slashing Time to First Token (TTFT) by $70\%$.
2. **Streaming Execution:**
   * Never wait for full generation completion ($2\text{--}5\text{ seconds}$).
   * Stream tokens immediately over Server-Sent Events (SSE) or WebSockets. The user sees output begin within **$\sim 200\text{ms}$**, delivering an instantaneous user experience.

---

## 🧩 The 8 Core Production Patterns

### 1. Hybrid Retrieval & Reciprocal Rank Fusion (RRF)
* **Goal:** Combine dense semantic search and sparse exact keyword matching into a single normalized candidate ranking.
* **Formula:** $RRF\_Score(d) = \sum_{m} \frac{1}{k + rank_m(d)}$ (Default $k=60$).
* **When to use:** In any production system dealing with domain-specific jargon, error codes, legal references, or product SKUs.

### 2. Hypothetical Document Embeddings (HyDE)
* **Goal:** Bridge the asymmetric search gap between terse user queries and long documentation paragraphs.
* **Mechanism:** Generate a zero-shot hypothetical answer via a cheap SLM and embed that answer to perform vector retrieval.

### 3. Two-Tier Semantic Caching & Token Routing
* **Goal:** Eliminate redundant LLM calls and cut API costs by $30\text{--}40\%$.
* **Mechanism:** Exact SHA-256 string hash in Redis ($<1\text{ms}$) combined with vector similarity matching ($\tau \ge 0.94$).

### 4. Hierarchical Parent-Child Chunking
* **Goal:** Decouple the retrieval unit from the generation context unit.
* **Mechanism:** Embed 128-token child chunks for precision; inject 1024-token parent chunks into the LLM for comprehension.

### 5. Self-Corrective RAG (CRAG) & Active Guardrails
* **Goal:** Prevent hallucinations when the retrieval engine returns low-quality or irrelevant passages.
* **Mechanism:** Pre-generation confidence scoring with 3-way branching: proceed, strip noise, or fallback to external search.

### 6. Stateful Multi-Agent Supervisor & Tool Calling
* **Goal:** Prevent tool confusion and infinite execution loops in complex agentic workflows.
* **Mechanism:** A supervisor router orchestrates narrow specialized worker agents ($3\text{--}5$ tools max) with Redis state checkpointing and hard cycle limits.

### 7. Prompt Prefix Caching & TTFT Optimization
* **Goal:** Minimize latency and reduce token costs by up to $90\%$ on repetitive agent turns.
* **Mechanism:** Strict ordering of static byte blocks (system prompts, tool definitions) before dynamic chat history.

### 8. Vector Index Selection: HNSW vs. IVFFlat vs. DiskANN

| Index Type | Search Latency | RAM Consumption | Build Time | Best Suited For |
| :--- | :--- | :--- | :--- | :--- |
| **Flat (Exact)** | $O(N)$ (Slow, $>150\text{ms}$) | Low (No graph overhead) | $0\text{s}$ | Datasets $<10{,}000$ vectors |
| **IVFFlat (Inverted File)** | $O(\sqrt{N})$ ($10\text{--}30\text{ms}$) | Medium (Centroid lists) | Fast | Up to $1\text{M}$ vectors with tight RAM |
| **HNSW (Hierarchical Navigable Small World)** | $O(\log N)$ ($2\text{--}6\text{ms}$) | **High** ($1.5\times\text{--}2\times$ raw vector size) | Moderate | Production real-time apps ($<10\text{M}$ vectors) |
| **DiskANN / Vamana** | $O(\log N)$ ($10\text{--}20\text{ms}$) | **Ultra-Low** (Graph on NVMe SSD) | Heavy | Massive scale ($10\text{M}\text{--}1\text{B}+$ vectors) |

---

## ⏱️ Production Latency & Cost Budgets

### The Sub-300ms End-to-End Latency Waterfall
```
[Client Request]
       │
       ├── (1) Semantic Cache Check:     ~8ms
       ├── (2) Query Embedding (API):    ~25ms
       ├── (3) HNSW Vector Index Search: ~6ms
       ├── (4) BM25 Keyword Search:      ~4ms
       ├── (5) Cross-Encoder Rerank:     ~45ms
       ├── (6) Context Assembly & Token: ~5ms
       └── (7) LLM Time to First Token:  ~180ms  (With Prompt Caching)
────────────────────────────────────────────────────────
Total End-to-End TTFT:                  ~273ms  ✅ [PASS]
```

### The Cost Reduction Rule of Thirds
1. **Tier 1 - Semantic Caching:** Eliminates $30\%$ of redundant API calls.
2. **Tier 2 - Model Tiering (Router):** Routes $60\%$ of simple user turns to small models ($0.15\$/1\text{M}$ tokens) and only $10\%$ of complex reasoning to frontier models ($3.00\$/1\text{M}$ tokens).
3. **Tier 3 - Context Pruning:** Eliminates irrelevant chunks via Cross-Encoder thresholds before generation, saving $40\%$ of prompt token volume.

---

## 🚨 The Production Disaster Manual (Failure Modes & Fixes)

### 1. The "Lost in the Middle" Phenomenon
* **Symptom:** LLMs accurately attend to information placed at the very beginning or very end of a context window, but retrieval accuracy drops by up to $40\%$ when critical facts reside in the middle of a $32\text{k}+$ token context.
* **Production Fix:** Implement **Reversed-Rank Packing**: Place the highest-ranked chunk at the very beginning of the prompt, the second-highest at the very end, and less relevant chunks in the center.

### 2. Chunk Boundary Truncation
* **Symptom:** An essential fact (e.g., *"Patient was prescribed 50mg Sertraline after exhibiting tremors."*) is split across chunk boundaries (Chunk A ends with *"prescribed 50mg"*, Chunk B starts with *"Sertraline"*), ruining embedding cosine distance.
* **Production Fix:** Enforce a minimum **$15\text{--}20\%$ sliding window overlap** and evaluate chunk boundaries strictly on sentence/paragraph AST tokens rather than arbitrary character lengths.

### 3. Vector Drift & Model Incompatibilities
* **Symptom:** Upgrading from `text-embedding-3-small` to `text-embedding-3-large` or changing dimensionalities breaks similarity scores silently.
* **Production Fix:** Never store raw embeddings without immutable metadata headers: `{"model": "text-embedding-3-small", "dim": 1536, "version": "v1.2"}`. Enforce blue-green re-indexing pipelines before flipping production query traffic.

---

## 💻 Runnable 50-Line Python Primitives

### Minimalist Pure NumPy Vector Search Engine
Zero heavy dependencies. Demonstrates exact cosine distance vector retrieval:

```python
import numpy as np

class MiniVectorDB:
    def __init__(self, dimension: int):
        self.dim = dimension
        self.vectors = np.empty((0, dimension), dtype=np.float32)
        self.metadata = []

    def insert(self, vector: list[float], meta: dict):
        vec = np.array(vector, dtype=np.float32)
        norm = np.linalg.norm(vec)
        normalized = vec / (norm if norm > 0 else 1.0)
        self.vectors = np.vstack([self.vectors, normalized])
        self.metadata.append(meta)

    def search(self, query_vector: list[float], top_k: int = 3) -> list[tuple[dict, float]]:
        if len(self.vectors) == 0:
            return []
        q = np.array(query_vector, dtype=np.float32)
        q_norm = q / (np.linalg.norm(q) if np.linalg.norm(q) > 0 else 1.0)
        
        # Vectorized dot product for cosine similarities
        similarities = np.dot(self.vectors, q_norm)
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [(self.metadata[i], float(similarities[i])) for i in top_indices]

# Test it in 2 seconds
if __name__ == "__main__":
    db = MiniVectorDB(dimension=3)
    db.insert([1.0, 0.0, 0.0], {"doc": "Document about Database Systems"})
    db.insert([0.0, 1.0, 0.0], {"doc": "Document about Neural Networks"})
    db.insert([0.9, 0.1, 0.0], {"doc": "Document about Distributed Storage"})

    results = db.search([1.0, 0.05, 0.0], top_k=2)
    for meta, score in results:
        print(f"[{score:.4f}] {meta['doc']}")
```

---

## 🎯 Real-World System Design Case Studies

### Case Study 1: Real-Time Multimodal AI Academic Engine
*Inspired by the architecture of [Florix AI](https://github.com/Gana-Y/Florix_AI).*
* **System Goal:** Ingest textbooks, PDFs, and YouTube transcripts to generate adaptive spaced-repetition quizzes and real-time interactive viva defenses with sub-$200\text{ms}$ query response times.
* **Architecture:**
  * Ingestion worker breaks documents into semantically coherent section nodes stored in PostgreSQL across $15+$ relational tables.
  * ChromaDB HNSW dense index handles semantic similarity with fallback to SQLite in local developer mode.
  * Async rate-limiting via SlowAPI prevents token exhaustion during simultaneous multi-user viva sessions.

### Case Study 2: Hardened Sandboxed Code Execution Agent
*Inspired by the architecture of [vesper-engine](https://github.com/Gana-Y/vesper-engine).*
* **System Goal:** Safely evaluate and execute arbitrary code generated by autonomous LLM agents in under $10\text{ms}$ with strict security envelopes.
* **Architecture:**
  * Linux `cgroups v2` enforce absolute memory caps ($128\text{MB}$) and strict CPU cycle budgets.
  * Seccomp filter blocks dangerous system calls (`ptrace`, unauthorized socket binds, filesystem escapes).
  * High-resolution sub-millisecond telemetry records exact execution timing and memory high-water marks.

### Case Study 3: Low-Latency Financial Order Intelligence Engine
*Inspired by the architecture of [OrderCore](https://github.com/Gana-Y/OrderCore).*
* **System Goal:** Process real-time market streams, execute order-book cancellations with $O(1)$ amortized latency, and run AI natural-language order queries concurrently.
* **Architecture:**
  * High-performance dual-heap matching engine processes $136{,}000\text{ orders/sec}$ at $4.3\mu\text{s}$ execution latency.
  * Python AsyncIO event loop handles bidirectional WebSocket streams while decoupled AI background workers query the order book telemetry asynchronously.

---

## ⭐ Explore My Other Repositories

| Repository | Description | Domain |
| :--- | :--- | :--- |
| 🚀 [**Florix_AI**](https://github.com/Gana-Y/Florix_AI) | Production-grade multimodal AI study platform with in-database RAG, adaptive spaced repetition, and interactive viva defenses. | `Full-Stack RAG / AI` |
| ⚡ [**vesper-engine**](https://github.com/Gana-Y/vesper-engine) | Hardened, high-performance sandboxed code execution engine & online judge arbiter enforcing Linux cgroups and microsecond telemetry. | `Systems / Linux / Python` |
| 📈 [**OrderCore**](https://github.com/Gana-Y/OrderCore) | High-throughput Limit Order Book & matching engine in Python (FastAPI/WebSockets) achieving 136k orders/sec with $4.3\mu\text{s}$ latency. | `Low-Latency Systems / FinTech` |
| 🎓 [**Free-Courses**](https://github.com/Gana-Y/Free-Courses) | Curated catalog of verified 100% free technical courses & certifications from NVIDIA, Anthropic, AWS, Google, Harvard, and MIT. | `Developer Education` |
| 📚 [**AI-ML-Notes**](https://github.com/Gana-Y/AI-ML-Notes) | Comprehensive study notes, summaries, and mathematical breakdowns across AI, ML, Deep Learning, and LLMs. | `Study Notes / AI & ML` |
| 🗺️ [**AI-ML-RoadMap**](https://github.com/Gana-Y/AI-ML-RoadMap) | The step-by-step 8-month roadmap to becoming an AI Engineer: courses, projects, habits & interview strategies. | `Career Roadmap / AI` |

---

## 🤝 Contributing

Contributions, architectural additions, and verified production benchmarks are welcome!

1. Fork the repository (`git checkout -b feature/new-pattern`).
2. Add your architectural pattern with clear diagrams, math, and code.
3. Commit with a semantic message (`git commit -m "feat: add GraphRAG community detection pattern"`).
4. Open a Pull Request.

---

## 🌟 Show Your Support — Star This Repo!

If this architectural guide helped you prepare for an AI system design interview, design production RAG pipelines, or understand low-latency LLM engineering:

* ⭐ **Give this repository a star** — it directly supports free, open-source AI education and helps more engineers find it!
* 🍴 **Fork it** — use it as your personal study guide or production architecture blueprint.
* 📢 **Share it** — spread the word to your college peers, engineering teams, or on LinkedIn & X (Twitter)!

---

## 👨‍💻 Author & Maintainer

Maintained with ❤️ by **[Ganesh Yandigeri](https://github.com/Gana-Y)**

- 🌐 **Portfolio:** [ganesh-y.vercel.app](https://ganesh-y.vercel.app)
- 💼 **LinkedIn:** [linkedin.com/in/ganesh-yandigeri](https://linkedin.com/in/ganesh-yandigeri)
- 🐙 **GitHub:** [@Gana-Y](https://github.com/Gana-Y)

---

## 📜 License

Distributed under the [MIT License](LICENSE). Feel free to learn from, use, and share!
