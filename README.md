# Agentic RAG Pipeline
This repository contains two Python scripts that implement a Retrieval‑Augmented Generation (RAG) workflow:
## Script Details:
### 🔄 Insert into Vector Store

- **CSVLoader**  
  Reads transcripts and metadata from a CSV file.

- **Token Counting**  
  Uses `tiktoken` to count tokens and characters.

- **Embeddings**  
  Generates 384‑dim vectors with the `sentence-transformers/all-MiniLM-L6-v2` model.

- **Pinecone Index Creation**  
  Creates a new Pinecone index (cosine metric).

- **Batch Upsert**  
  Splits documents into chunks, assigns each a `UUID`, and upserts embeddings in batches.

- **Similarity Search**  
  Demonstrates both simple similarity search and search with score retrieval.

- **User‑Specific Queries**  
  Shows how to query different namespaces per user.

---

### 🧠 Retrieval & Agent Workflow

- **LLM Initialization**  
  Instantiates a `ChatLiteLLM` (Mistral-7B).

- **Memory**  
  Uses `ConversationBufferWindowMemory` to track conversation history.

- **RetrievalQA Chain**  
  Wraps the Pinecone vector store as a retriever in a `RetrievalQA` chain.

- **Tools**  
  - `pinecone_tool`: Vector DB search  
  - `tavily_tool`: Live web search via Tavily  

- **Agents & Crew**  
  - **Research Analyst**: Performs retrieval from Pinecone and Tavily  
  - **Content Verifier**: Validates and summarizes retrieved content  

- **Task Orchestration**  
  Leverages Crew AI to assign tasks and produce a final structured answer.








