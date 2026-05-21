# -----------------------------------
# STEP 1: Create In Memory Vector DB using PubMed Data
# -----------------------------------

from langchain_community.document_loaders import PubMedLoader
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

# Search PubMed
loader = PubMedLoader(query="lung cancer immunotherapy", load_max_docs=10)

# Load documents
documents = loader.load()

# Create embeddings
embeddings = OpenAIEmbeddings()

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