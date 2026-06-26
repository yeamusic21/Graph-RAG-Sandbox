# Graph-RAG-Sandbox
Sandbox for learning Graph RAG

# Learning Resources
- https://docs.langchain.com/oss/python/integrations/retrievers/graph_rag

# What GraphRetriever Does

GraphRetriever adds another step.

Query
      │
      ▼
Embedding
      │
      ▼
Vector Search
      │
      ▼
Top-k documents
      │
      ▼
Graph Traversal
      │
      ▼
Expanded context

### Deeper Dive

Imagine the metadata graph looks like

Paper A
therapy = PD-1
year = 2025

        │
        │ same therapy
        ▼

Paper C
therapy = PD-1
year = 2024

        │
        │ same disease
        ▼

Paper D
lung cancer

        │
        │ same year
        ▼

Paper E
2025

Even if Papers C, D, and E weren't among the top semantic matches, GraphRetriever can follow those metadata links and include them in the retrieved context. This helps answer questions that require connecting related documents rather than relying solely on semantic similarity.

# Graph RAG General Architecture

In fact, this is one of the most common GraphRAG architectures

The pipeline typically looks like:

PubMed
   │
   ▼
Load abstracts
   │
   ▼
LLM Entity Extraction
   │
   ▼
Document(
    page_content=abstract,
    metadata={
        disease,
        genes,
        drugs,
        biomarkers,
        pathways,
        year,
        ...
    }
)
   │
   ▼
Embeddings
   │
   ▼
Vector Store
   │
   ▼
GraphRetriever