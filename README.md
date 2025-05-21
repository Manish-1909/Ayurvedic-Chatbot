AYUR-BOT: Ayurvedic Medicine Recommendation Chatbot

AYUCARE is an intelligent chatbot designed to recommend Ayurvedic remedies based on user queries. It leverages the power of Llama 2, FAISS (CPU) for vector similarity search, and Chainlit for a user-friendly interface. The knowledge base is built from a comprehensive Ayurvedic encyclopedia, making it a reliable assistant for traditional health solutions.

----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🧠 Features

- 🔍 Natural Query Handling: Ask health-related questions in natural language.
- 📚 Ayurvedic Remedy Suggestions: Provides relevant treatments based on traditional texts.
- ⚡ Fast Response Time: Efficient retrieval using FAISS vector store.
- 💬 Interactive Chat Interface: Built using Chainlit for an intuitive frontend.
- 🔐 Runs Locally: Fully private and offline-ready with a local LLM (Llama 2).

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

🛠️ Tech Stack

| Component        | Technology          |
|------------------|---------------------|
| Language Model   | [LLaMA 2](https://huggingface.co/meta-llama) (7B, quantized `.bin`) |
| Embeddings       | all-MiniLM-L6-v2 from Hugging Face |
| Vector DB        | FAISS (CPU)         |
| Interface        | Chainlit            |
| Language         | Python              |

------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


