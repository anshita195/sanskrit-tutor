# 🕉️ Sanskrit Tutor - RAG-powered Learning System

A Retrieval-Augmented Generation (RAG) chatbot system for Sanskrit language learning using authentic Bhagavad Gita texts.

## 🌟 Current System Status

**✅ WORKING SYSTEM:**
- **719 passages**: Authentic Bhagavad Gita verses with Devanagari + IAST
- **2,853 QA pairs**: Comprehensive Q&A covering all 18 chapters
- **FAISS search**: Semantic search with citation-backed responses
- **Local GGUF model**: mistral-7b-instruct-v0.2.Q4_K_M.gguf (4-bit quantized)
- **Multi-script support**: Devanagari, IAST, and English input/output

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download GGUF Model Only

**Note**: The repository includes all source data and processed files. You only need to download the model.

**Download GGUF Model:**
```bash
# Create models directory
mkdir -p user_assets/models

# Download Mistral 7B Instruct v0.2 (4-bit quantized, ~4.4GB)
# Recommended: Q4_K_M for best quality/size balance
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf -O user_assets/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

**Alternative Model Options:**
- **Q4_0** (4.1GB): Faster, slightly lower quality
- **Q5_K_M** (5.1GB): Better quality, larger size
- **Q6_K** (5.9GB): High quality, largest size

### 3. Validate Setup
```bash
python src/utils/config_validator.py
```

### 4. Build Search Index
```bash
python src/embed_index.py --config user_assets/config.yaml
```

### 5. Launch UI
```bash
# Main learning interface
python src/ui_gradio.py --config user_assets/config.yaml

# Sanskrit conversation interface
python src/sanskrit_chat_ui.py --config user_assets/config.yaml
```

## 🧪 Test the System

### Test 1: Search Functionality
```bash
python src/embed_index.py --config user_assets/config.yaml --test-search "dharma"
```

### Test 2: RAG System (Command Line)
```bash
python src/rag.py --config user_assets/config.yaml --interactive
```

### Test 3: LLM Backend
```bash
python src/llm_backends.py --config user_assets/config.yaml
```

## 📚 Sample Queries to Try

**English:**
- "What is dharma?"
- "Explain karma yoga"
- "What does the Bhagavad Gita say about meditation?"

**Sanskrit (IAST):**
- "dharmaḥ kim?"
- "karma yogaḥ katham?"
- "dhyānaṃ katham?"

**Sanskrit (Devanagari):**
- "योगः किम्?"
- "ध्यानं कथम्?"
- "धर्मः किम्?"

## 🎯 System Capabilities

- **Citation-backed responses**: All answers include exact verse references [BG1.1]
- **Multi-script input**: Accepts Devanagari, IAST, or English
- **Semantic search**: Finds relevant verses even with different wording
- **Interactive learning**: Q&A mode with difficulty levels
- **Local inference**: Runs completely offline with GGUF model

## 📋 Prerequisites

### Required Downloads (~4.4GB):
- **GGUF Model Only**: [Hugging Face - Mistral 7B Instruct v0.2](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)

### What's Included in Repository:
- ✅ **Source Data**: Bhagavad Gita dataset (raw_data/)
- ✅ **Sanskrit Corpus**: Text corpus (sanskrit_corpus_kaggle/)
- ✅ **Processed Data**: passages.jsonl, qa_pairs.jsonl, config.yaml
- ✅ **Application Code**: Complete RAG system (src/)

### System Requirements:
- **Python 3.8+**
- **RAM**: 8GB+ (for Q4_K_M model)
- **Storage**: 5GB+ free space (for model download)

## 🔧 Troubleshooting

### Common Issues:

**"No user assets found"**
```bash
# All data is included in the repository
# Just make sure you're in the right directory
ls -la user_assets/
# Should show: config.yaml, passages.jsonl, qa_pairs.jsonl
```

**"Model not found"**
```bash
# Check model file exists
ls -la user_assets/models/
# Should show: mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

**"Out of memory"**
```bash
# Use smaller model (Q4_0 instead of Q4_K_M)
# Or increase system RAM
```

