# 📚 Project 8: Personal RAG Knowledge Base

A personal AI assistant powered by Retrieval-Augmented Generation (RAG) that has complete knowledge of my 24-week AI automation learning journey — able to answer questions accurately using my own documents, without training a custom model.

---

## 📌 What This Project Does

This project builds an AI assistant that "knows" everything about my learning roadmap by retrieving relevant information from my own notes and documents, then using an LLM to generate accurate, grounded answers — instead of relying on the model's general training data.

The knowledge base includes:
- Personal background and goals
- All 5 phases of the roadmap
- Completed topics per phase
- Projects built, with descriptions
- Key concepts mastered
- Upcoming learning topics

---

## ⚙️ How RAG Works (Step by Step)

My Documents (notes, roadmap, project details)
↓
Loaded into Python
↓
Split into 400-character chunks
↓
Converted to vectors using HuggingFace Embeddings
↓
Stored in FAISS Vector Database
↓
User asks a question
↓
Relevant chunks retrieved from FAISS
↓
Groq AI (openai/gpt-oss-20b) reads chunks → generates answer


**Why this matters:** RAG gives an AI model access to specific, private, up-to-date information (like my personal learning journey) without needing to train or fine-tune a new model. It's fast, accurate, and completely free.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **LangChain** | Orchestrates the RAG pipeline |
| **HuggingFace Embeddings** | Converts text chunks into vectors |
| **FAISS** | Vector database for fast similarity search |
| **Groq API (openai/gpt-oss-20b)** | Generates answers from retrieved chunks |
| **ConversationalRetrievalChain** | Combines retrieval with conversational context |
| **ConversationBufferWindowMemory** | Maintains recent conversation history |

---

## 📂 Knowledge Base Contents

- Personal background and learning goals
- All 5 phases of the AI automation roadmap
- Completed topics, broken down per phase
- Full list of projects built, with descriptions
- Key concepts mastered along the way
- Upcoming/future learning topics
  
---

## 🎯 Key Learning

RAG (Retrieval-Augmented Generation) gives an AI model access to your own specific documents and data without needing to train a new model. It's fast, accurate, and — using free tools like HuggingFace Embeddings, FAISS, and Groq — completely free to build.

---

## 👤 Author

Built by Abdullah as part of a self-directed AI Automation learning program.
