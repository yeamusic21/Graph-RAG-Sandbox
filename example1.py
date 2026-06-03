# -----------------------------------
# STEP 1: Create In Memory Vector DB using PubMed Data
# -----------------------------------

from langchain_community.document_loaders import PubMedLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# Search PubMed
loader = PubMedLoader(
    query="lung cancer immunotherapy",
    load_max_docs=10
)

# Load documents
documents = loader.load()

# Create local embeddings
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5"
)

# Create vector store
vector_store = InMemoryVectorStore.from_documents(
    documents=documents,
    embedding=embeddings,
)

# Test similarity search
results = vector_store.similarity_search(
    "PD-1 checkpoint inhibitors",
    k=3,
)

for r in results:
    print(r.metadata)
    print(r.page_content[:500])
    print("=" * 80)