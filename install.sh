#!/bin/bash
# Sanskrit Tutor Installation Script
# Installs dependencies and sets up the environment

set -e  # Exit on any error

echo "🕉️ Sanskrit Tutor Installation Script"
echo "=================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1,2)
required_version="3.8"

if [ "$(echo "$python_version >= $required_version" | bc)" -eq 0 ]; then
    echo "❌ Error: Python $required_version or higher is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv .venv
fi

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "⬆️ Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📥 Installing core dependencies..."
pip install -r requirements.txt

# Check for additional installation options
echo ""
echo "🔧 Additional Installation Options:"
echo "1. Local GGUF inference: pip install llama-cpp-python"
echo "2. GPU acceleration: pip install faiss-gpu torch"
echo "3. Audio features: pip install librosa soundfile torchaudio"
echo "4. Fine-tuning: pip install transformers peft bitsandbytes accelerate datasets"
echo "5. Development tools: pip install pytest black flake8"
echo ""

read -p "Install additional features? (1-5, or 'n' for none): " choice

case $choice in
    1)
        echo "Installing local GGUF support..."
        pip install llama-cpp-python
        ;;
    2)
        echo "Installing GPU acceleration..."
        pip install faiss-gpu torch
        ;;
    3)
        echo "Installing audio features..."
        pip install librosa soundfile torchaudio
        ;;
    4)
        echo "Installing fine-tuning dependencies..."
        pip install transformers peft bitsandbytes accelerate datasets
        ;;
    5)
        echo "Installing development tools..."
        pip install pytest pytest-cov black flake8
        ;;
    n|N)
        echo "Skipping additional features."
        ;;
    *)
        echo "Unknown option. Skipping additional features."
        ;;
esac

# Create user assets directory structure if it doesn't exist
echo ""
echo "📁 Setting up user assets directory..."
mkdir -p user_assets/{models,adapters,audio_samples,embeddings}

# Create sample configuration if it doesn't exist
if [ ! -f "user_assets/config.yaml" ]; then
    echo "📝 Creating sample configuration..."
    cat > user_assets/config.yaml << 'EOF'
# Sanskrit Tutor Configuration
# Please customize this file according to your setup

# Model Configuration
model_path: null  # Path to local GGUF model, or null for hosted inference
gguf_local: false  # Set to true to use local GGUF model
n_ctx: 4096  # Context length for local models
n_gpu_layers: 0  # Number of GPU layers (0 for CPU-only)

# Embedding Configuration
embeddings_model: "sentence-transformers/all-mpnet-base-v2"

# Data Configuration
passages_file: "user_assets/passages.jsonl"
qa_file: "user_assets/qa_pairs.jsonl"
faiss_index_path: "data/faiss.index"

# Optional Features
audio_folder: "user_assets/audio_samples"

# RAG Parameters
retrieval_k: 5
max_tokens: 500
temperature: 0.7

# Hosted API Fallback Models (requires API keys)
hf_model: "mistralai/Mistral-7B-Instruct-v0.1"
openai_model: "gpt-3.5-turbo"
EOF
fi

echo ""
echo "🎉 Installation completed successfully!"
echo ""
echo "📋 Next Steps:"
echo "1. Place your Sanskrit texts in user_assets/passages.jsonl"
echo "2. Place your QA pairs in user_assets/qa_pairs.jsonl"
echo "3. (Optional) Place GGUF model in user_assets/models/"
echo "4. (Optional) Set API keys: export HF_API_KEY=... or export OPENAI_API_KEY=..."
echo "5. Validate setup: python src/utils/config_validator.py"
echo "6. Build index: python src/embed_index.py --config user_assets/config.yaml"
echo "7. Launch UI: python src/ui_gradio.py --config user_assets/config.yaml"
echo ""
echo "📚 For detailed instructions, see README.md"
echo ""
echo "🔍 Quick commands:"
echo "  Validate assets: python -m pytest tests/ -v"
echo "  Test RAG system: python src/rag.py --config user_assets/config.yaml --interactive"
echo "  Check LLM backends: python src/llm_backends.py --config user_assets/config.yaml"
