# 🕉️ Sanskrit Tutor - RAG-powered Learning System

A Retrieval-Augmented Generation (RAG) chatbot system for Sanskrit language learning.

## 🌟 Features

1. Data Sources: Mixed bag but functional
•  5,000 passages: Mix of legitimate Bhagavad Gita verses + random Kartik corpus chunks  
•  2,853 QA pairs: All valid Bhagavad Gita Q&A with proper citations
•  Sources: Kaggle datasets (legit: BG API database, questionable: random Sanskrit corpus)
•  Format: Proper JSONL with text_devanagari, text_iast, citations working
2. Embedding & Search: Actually solid
•  Model: all-mpnet-base-v2 (NOT the multilingual one from config - bug!)
•  FAISS Index: 5,000 vectors, HNSW algorithm working 
•  Search: Semantic search returning relevant results (your test proved it)
•  Composite embeddings: IAST + Devanagari concatenated - clever approach

## Techstack and Architecture

•  Local GGUF Model: mistral-7b-instruct-v0.2.Q4_K_M.gguf - 4-bit quantized, CPU-only
•  llama-cpp-python: Working, 2048 context length  
•  RAG Pipeline: Complete with citation validation
•  Multi-domain system: Sanskrit domains implemented
•  Text Processing: Devanagari/IAST normalization working

## For testing

1. Basic RAG: python src/rag.py --config user_assets/config.yaml --interactive
2. LLM Backend Test: python src/llm_backends.py --config user_assets/config.yaml 
3. Embedding Index: python src/embed_index.py --config user_assets/config.yaml --test-search "query" 
4. Sanskrit Chat UI: python src/sanskrit_chat_ui.py --config user_assets/config.yaml  
5. Main Gradio UI: python src/ui_gradio.py --config user_assets/config.yaml
6. Multi-domain UI: python src/multi_domain_ui.py --config user_assets/config.yaml


