from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DB_FAISS_PATH = 'vectorstore/db_faiss'

def create_vector_db():
    try:
        # Load the PDF
        loader = PyPDFLoader("C:\\Users\\mk594\\OneDrive\\Desktop\\Llama2-Medical-Chatbot\\data\\swami-sada-shiva-tirtha-the-ayurveda-encyclopedia.pdf")
        documents = loader.load()
        print(f"Loaded {len(documents)} documents from the PDF.")

        # Split documents into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,  # Adjusted chunk size for better retrieval
            chunk_overlap=200,  # Slightly larger overlap for context continuity
            separators=["\n\n", "\n", ".", " "]
        )
        texts = text_splitter.split_documents(documents)
        print(f"Split documents into {len(texts)} chunks.")

        # Filter out chunks that are too short (e.g., less than 50 characters)
        texts = [chunk for chunk in texts if len(chunk.page_content.strip()) > 50]
        print(f"Filtered chunks; remaining: {len(texts)}.")

        # Initialize embeddings
        device = 'cpu'
        print(f"Using device: {device}")

        embeddings = HuggingFaceEmbeddings(
            model_name='sentence-transformers/all-MiniLM-L6-v2',
            model_kwargs={'device': device}
        )

        # Create FAISS vector store
        db = FAISS.from_documents(texts, embeddings)

        # Save the FAISS vector store locally
        db.save_local(DB_FAISS_PATH)
        print(f"FAISS vector store successfully saved at {DB_FAISS_PATH}.")

    except Exception as e:
        print(f"Error creating vector DB: {e}")

if __name__ == "__main__":
    create_vector_db()






