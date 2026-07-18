# Information Retrieval Engine

An end-to-end semantic Information Retrieval system developed for the **Information Retrieval Challenge**. The goal is to retrieve and rank the most relevant documents for a given query over a large multi-domain corpus containing more than **216,000 documents**.

The project explores multiple retrieval strategies, progressively improving the ranking quality through query expansion, dense retrieval, document chunking, and embedding fine-tuning.

---

## Project Highlights

* Exploratory analysis of the document collection
* Data cleaning and preprocessing
* Sparse retrieval with **TF-IDF**
* Query Expansion using **Rocchio Algorithm**
* Dense Retrieval using **BAAI/BGE-base-en-v1.5**
* Efficient similarity search with **FAISS**
* Token-aware document chunking
* Fine-tuning of BGE using the competition training set
* Evaluation using standard Information Retrieval metrics

---

## Dataset

The competition dataset contains:

* **216,041 documents**
* Multiple knowledge domains (Programming, Physics, Gaming, Mathematics, etc.)
* Document metadata:

  * ID
  * Title
  * Text
  * Tags
  * Category
* Training queries with relevance judgments
* Test queries for leaderboard submission

---

## Methodology

### 1. Exploratory Data Analysis

Before building the retrieval system, the dataset was analyzed to understand:

* Category distribution
* Tag statistics
* Missing values
* Empty documents
* Text length distribution

---

### 2. Sparse Retrieval

A classical lexical retrieval baseline was implemented using:

* TF-IDF Vectorization
* Cosine Similarity

This establishes a strong baseline for later comparisons.

---

### 3. Query Expansion

To improve lexical retrieval, Rocchio Query Expansion was applied.

The expanded query combines:

* Original query
* Top retrieved feedback documents

This improves recall by enriching the query representation.

---

### 4. Dense Retrieval

The core retrieval model relies on:

**BAAI/bge-base-en-v1.5**

Documents and queries are encoded into dense embeddings and indexed using **FAISS** for efficient nearest-neighbor search.

Pipeline:

1. Text preprocessing
2. Embedding generation
3. Embedding normalization
4. FAISS indexing
5. Top-K retrieval

---

### 5. Token-aware Chunking

Long documents exceeding the transformer's context window are split into overlapping chunks.

Characteristics:

* Maximum 400 tokens per chunk
* 80-token overlap
* Chunk-level embedding generation

This prevents information loss caused by truncation.

---

### 6. Embedding Fine-Tuning

Instead of relying solely on the pretrained model, **BGE-base-en-v1.5** was fine-tuned on the competition training data.

Training details:

* Sentence Transformers
* MultipleNegativesRankingLoss
* Positive (query, relevant document) pairs
* GPU training

The resulting embeddings better capture the semantic relationships specific to the competition dataset.

---

## Technologies

* Python
* Sentence Transformers
* Hugging Face Transformers
* FAISS
* Scikit-learn
* NumPy
* Pandas
* Matplotlib
* PyTorch

---

## Repository Structure

```text
.
├── notebooks/
│   └── Moteur-de-recherche-et-Classification-documentaire-RAG.ipynb
├── data/
├── outputs/
├── requirements.txt
└── README.md
```

---

## Results

The project evaluates several retrieval strategies and compares their performance throughout the notebook:

* TF-IDF baseline
* TF-IDF + Rocchio Query Expansion
* Dense Retrieval (BGE)
* Fine-tuned BGE embeddings

Performance is assessed using standard Information Retrieval metrics such as:

* Precision@K
* Recall@K
* Mean Reciprocal Rank (MRR)

---

## Future Improvements

Possible extensions include:

* Hybrid Retrieval (BM25 + Dense Retrieval)
* Cross-Encoder re-ranking
* Hard Negative Mining
* Ensemble of embedding models
* Multi-vector retrieval (ColBERT)
* Retrieval-Augmented Generation (RAG)

---

## Authors

Developed as part of the **Information Retrieval Challenge**, this project explores modern semantic search techniques by combining classical Information Retrieval methods with transformer-based dense retrieval.
