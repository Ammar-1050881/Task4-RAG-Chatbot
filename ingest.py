from langchain_community.document_loaders import PyPDFLoader
# --- UPDATED IMPORT BELOW ---
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# 1. LOAD: Read the PDF
print("📖 Loading PDF...")
loader = PyPDFLoader("document.pdf")
data = loader.load()

# 2. CHUNK: Chop the text into smaller pieces
print("✂️ Splitting text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(data)

# 3. EMBED: Turn text into numbers (Vectors)
print("🔢 Creating embeddings (this might take a minute)...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# 4. STORE: Save these numbers into a database (ChromaDB)
print("💾 Saving to Vector Database...")
vector_db = Chroma.from_documents(
    documents=chunks, 
    embedding=embeddings, 
    persist_directory="./chroma_db"
)

print("✅ SUCCESS! Your document is now a searchable Vector Database.")