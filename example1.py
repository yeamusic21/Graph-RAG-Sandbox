# -----------------------------------
# Create In Memory Vector DB using PubMed Data
# Enrich Data
# Create GraphRetriever
# -----------------------------------

from langchain_community.document_loaders import PubMedLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
import re
from langchain_graph_retriever import GraphRetriever

### Step 1 - Load PubMed

# Search PubMed
loader = PubMedLoader(
    query="lung cancer immunotherapy",
    load_max_docs=10
)

# Load documents
documents = loader.load()

### Step 2 - Enrich the metadata

GENES = [
    "PD-1",
    "PD-L1",
    "CTLA-4",
    "RUNX1",
    "NSUN2",
    "FABP5",
    "DLL3",
]

for doc in documents:

    text = doc.page_content.upper()

    # Disease
    if "NON-SMALL CELL LUNG CANCER" in text or "NSCLC" in text:
        disease = "NSCLC"

    elif "SMALL CELL LUNG CANCER" in text or "SCLC" in text:
        disease = "SCLC"

    else:
        disease = "Unknown"

    # Therapy
    if "PD-1" in text:
        therapy = "PD-1"

    elif "PD-L1" in text:
        therapy = "PD-L1"

    elif "CTLA-4" in text:
        therapy = "CTLA-4"

    else:
        therapy = "Other"

    # Gene extraction
    genes = []

    for gene in GENES:
        if re.search(rf"\b{re.escape(gene)}\b", text):
            genes.append(gene)

    doc.metadata["disease"] = disease
    doc.metadata["therapy"] = therapy
    doc.metadata["genes"] = genes
    doc.metadata["year"] = doc.metadata["Published"][:4]

print("Check out new metadata...")
print(documents[0].metadata)

### Step 3 - Build the vector store

# Create local embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# Create vector store
vector_store = InMemoryVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
)

### Step 4 - Add GraphRetriever

retriever = GraphRetriever(
    store=vector_store,
    edges=[
        ("disease", "disease"),
        ("therapy", "therapy"),
        ("genes", "genes"),
        ("year", "year"),
    ]
)

### Step 5 - Query

# Test similarity search
results = vector_store.similarity_search(
    "PD-1 checkpoint inhibitors",
    k=3,
)

print("=" * 80)
print("Similarity search results...")
print("=" * 80)
for r in results:
    print(r.metadata)
    print(r.page_content[:500])
    print("=" * 80)

# Test GraphRetriever
results = retriever.invoke(
    "Why are some NSCLC patients resistant to PD-1 therapy?"
)

print("=" * 80)
print("GraphRetriever search results...")
print("=" * 80)
for doc in results:
    print(doc.metadata)
    print(doc.page_content[:300])
    print("=" * 80)

#########################################################
# What happens behind the scenes?

# Suppose your query embedding is closest to Paper A:

# Paper A
# therapy = PD-1
# disease = NSCLC
# genes = [PD-1, PD-L1]

# A normal vector search would return:

# Paper A
# Paper B
# Paper C

# GraphRetriever can then expand the results by following shared metadata such as:

# Paper A
# │
# ├── same therapy (PD-1)
# │      ↓
# │   Paper D
# │
# ├── same disease (NSCLC)
# │      ↓
# │   Paper E
# │
# └── same gene (PD-L1)
#        ↓
#     Paper F

# This gives your language model additional, related context beyond the initial nearest-neighbor search.
#########################################################