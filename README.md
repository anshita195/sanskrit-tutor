# 🕉️ Sanskrit Tutor - RAG-powered Learning System

A Retrieval-Augmented Generation (RAG) chatbot system for Sanskrit language learning using authentic Bhagavad Gita texts.

## 🎯 Getting Started (TL;DR)

**For immediate use:**
1. `git clone` this repository
2. `pip install -r requirements.txt`
3. Download the GGUF model (4.4GB) - see links below
4. `python src/ui_gradio.py --config user_assets/config.yaml`
5. Start learning Sanskrit! 🕉️

**Everything else is already included** - no additional data downloads needed!

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

### 2. Download Required Files

**Note**: The repository includes all processed data files. You only need to download the GGUF model to run the system.

**Download GGUF Model:**
```bash
# Create models directory
mkdir -p user_assets/models

# Download Mistral 7B Instruct v0.2 (4-bit quantized, ~4.4GB)
# Recommended: Q4_K_M for best quality/size balance
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf -O user_assets/models/mistral-7b-instruct-v0.2.Q4_K_M.gguf
```

**Optional: Download Source Datasets (for development/extending the system):**
```bash
# Download Sanskrit corpus (616MB) - used to create additional passages
wget https://www.kaggle.com/datasets/preetsojitra/sanskrit-text-corpus/download -O sanskrit_corpus_kaggle/train.txt

# Download Bhagavad Gita dataset (for reference)
wget https://www.kaggle.com/datasets/ptprashanttripathi/bhagavad-gita-api-database/download -O bhagavad_gita_dataset.zip
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

## 📖 Data Sources

**Primary Data (Included in Repository):**
- **Bhagavad Gita**: 18 chapters, 700+ authentic Sanskrit verses
- **Source**: [Kaggle Bhagavad Gita API Database](https://www.kaggle.com/datasets/ptprashanttripathi/bhagavad-gita-api-database)
- **Status**: ✅ **Already processed and included** - no download needed
- **Processing**: Converted to structured JSONL format with Devanagari + IAST

**Additional Data (Optional for Development):**
- **Sanskrit Corpus**: [Kaggle Sanskrit Text Corpus](https://www.kaggle.com/datasets/preetsojitra/sanskrit-text-corpus) (616MB)
- **Status**: ❌ **Not included** - optional download for extending the system
- **Usage**: Can be processed to add more passages to the system

## 📋 Prerequisites

### Required Downloads (~4.4GB):
- **GGUF Model Only**: [Hugging Face - Mistral 7B Instruct v0.2](https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF)

### What's Included in Repository:
- ✅ **Source Data**: Bhagavad Gita dataset (raw_data/) - 18 chapters, 700+ verses
- ✅ **Processed Data**: passages.jsonl (719 verses), qa_pairs.jsonl (2,853 Q&A pairs), config.yaml
- ✅ **Application Code**: Complete RAG system (src/)
- ✅ **Embeddings**: Pre-built FAISS index and embeddings
- ✅ **Ready to Run**: Just download the model and start!

### What You Need to Download:
- ❌ **GGUF Model**: mistral-7b-instruct-v0.2.Q4_K_M.gguf (4.4GB) - **REQUIRED**

### Optional Downloads (for development/extending the system):
- ❌ **Sanskrit Corpus**: [Kaggle Sanskrit Text Corpus](https://www.kaggle.com/datasets/preetsojitra/sanskrit-text-corpus) (616MB) - to add more passages
- ❌ **Original Bhagavad Gita Dataset**: [Kaggle Bhagavad Gita API Database](https://www.kaggle.com/datasets/ptprashanttripathi/bhagavad-gita-api-database) - original source files (already processed and included in repo)

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

