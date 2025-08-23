# 🚀 Quick Start Guide - Sanskrit Tutor

**Get your Sanskrit Tutor running in 15 minutes!**

## ⚡ **Option 1: Use Template Data (Fastest)**

This gets you running immediately with sample Sanskrit texts:

```bash
# 1. Copy template files to active locations
copy user_assets\passages_template.jsonl user_assets\passages.jsonl
copy user_assets\qa_pairs_template.jsonl user_assets\qa_pairs.jsonl  
copy user_assets\config_template.yaml user_assets\config.yaml

# 2. Edit config.yaml for API mode (no model download needed)
# Change these lines in user_assets/config.yaml:
# gguf_local: false
# model_path: null

# 3. Set API key (choose one)
set HF_API_KEY=hf_your_key_here
# OR
set OPENAI_API_KEY=sk_your_key_here

# 4. Install dependencies
pip install -r requirements.txt

# 5. Validate setup
python src\utils\config_validator.py

# 6. Build search index
python src\embed_index.py --config user_assets\config.yaml

# 7. Launch UI
python src\ui_gradio.py --config user_assets\config.yaml
```

**You'll have a working Sanskrit Tutor with 25+ passages from Bhagavad Gita, Upanishads, and more!**

## ⚡ **Option 2: Local Model (More Private)**

If you want to run completely offline:

```bash
# 1. Copy templates (same as above)
copy user_assets\passages_template.jsonl user_assets\passages.jsonl
copy user_assets\qa_pairs_template.jsonl user_assets\qa_pairs.jsonl  
copy user_assets\config_template.yaml user_assets\config.yaml

# 2. Download a GGUF model (4GB download)
mkdir user_assets\models
# Visit: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF
# Download: mistral-7b-instruct-v0.1.q4_k_m.gguf
# Save as: user_assets\models\mistral-7b-instruct.gguf

# 3. Install local inference
pip install llama-cpp-python

# 4. Make sure config.yaml has:
# gguf_local: true
# model_path: "user_assets/models/mistral-7b-instruct.gguf"

# 5. Continue with steps 4-7 from Option 1
```

## 🎯 **Exact File Requirements**

### **passages.jsonl Format** (Your Sanskrit Texts)
```json
{"id":"bg_001_001","text_devanagari":"धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः","text_iast":"dharmakṣetre kurukṣetre samavetā yuyutsavaḥ","work":"Bhagavadgita","chapter":"1","verse":"1a","language":"sanskrit","source_url":"https://www.sacred-texts.com/hin/sbe08/sbe08003.htm","notes":"Opening verse"}
```

**Required Fields:**
- `id` - Unique identifier (used in citations)
- `text_devanagari` - Sanskrit in Devanagari script  
- `text_iast` - Sanskrit in IAST transliteration
- `work` - Name of the text/book
- `chapter` - Chapter/section number
- `verse` - Verse/line number  
- `language` - Always "sanskrit"
- `source_url` - Where you got the text
- `notes` - Translation or explanation

### **qa_pairs.jsonl Format** (Your Q&A Data)
```json
{"id":"qa_001","question":"What is the opening verse of the Bhagavad Gita?","answer":"The Gita opens with 'dharmakṣetre kurukṣetre samavetā yuyutsavaḥ' meaning 'On the field of dharma, on the field of Kuru, assembled and eager to fight'. [bg_001_001]","difficulty":"easy","related_passage_ids":["bg_001_001"]}
```

**Required Fields:**
- `id` - Unique question identifier
- `question` - The learning question
- `answer` - Answer with citations like [passage_id]
- `difficulty` - "easy", "medium", or "hard" 
- `related_passage_ids` - Array of relevant passage IDs

### **config.yaml Format** (Your Settings)
```yaml
# Choose one inference method:

# Option A: Hosted API (easier)
model_path: null
gguf_local: false

# Option B: Local model (private)  
model_path: "user_assets/models/mistral-7b-instruct.gguf"
gguf_local: true

# Always required:
embeddings_model: "sentence-transformers/all-mpnet-base-v2"
passages_file: "user_assets/passages.jsonl"
qa_file: "user_assets/qa_pairs.jsonl"
faiss_index_path: "data/faiss.index"
retrieval_k: 5
max_tokens: 500
temperature: 0.7
```

## 🔑 **API Keys (If Using Hosted Inference)**

### **Hugging Face (Recommended)**
1. Go to: https://huggingface.co/settings/tokens
2. Create new token
3. Set: `set HF_API_KEY=hf_your_token_here`

### **OpenAI (Alternative)**
1. Go to: https://platform.openai.com/api-keys
2. Create new secret key  
3. Set: `set OPENAI_API_KEY=sk_your_key_here`

## 🔧 **GGUF Model Downloads (If Using Local Inference)**

### **Best for Most Users: Mistral-7B-Instruct**
- **Size**: 4GB
- **RAM needed**: 8GB+
- **Download**: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.q4_k_m.gguf
- **Save as**: `user_assets\models\mistral-7b-instruct.gguf`

### **Smaller Option: CodeLlama-7B**  
- **Size**: 3.8GB
- **RAM needed**: 6GB+
- **Download**: https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF/resolve/main/codellama-7b-instruct.q4_k_m.gguf
- **Save as**: `user_assets\models\codellama-7b-instruct.gguf`

## ✅ **Success Checklist**

After setup, you should see:

1. **Validation passes**: `python src\utils\config_validator.py` shows green checkmarks
2. **Index builds**: `python src\embed_index.py` creates `data\faiss.index` 
3. **UI launches**: `python src\ui_gradio.py` opens web interface at http://localhost:7860
4. **Chat works**: You can ask "What is dharma?" and get cited answers
5. **Exercise works**: You can practice with Q&A pairs
6. **Citations show**: Answers include [passage_id] references

## 🆘 **Common Issues**

### **"No user assets found"**
```bash
# Make sure files exist:
dir user_assets
# Should show: passages.jsonl, qa_pairs.jsonl, config.yaml
```

### **"No LLM backend available"** 
```bash
# For API mode:
echo %HF_API_KEY%
# Should show: hf_xxxxx

# For local mode:
dir user_assets\models
# Should show your .gguf file
```

### **"Embedding model not available"**
```bash
pip install sentence-transformers
```

### **"FAISS index not found"**
```bash
# Run index builder:
python src\embed_index.py --config user_assets\config.yaml
```

## 📚 **Next Steps**

Once you have the basic system working:

1. **Add your own Sanskrit texts** - Replace template passages with your texts
2. **Create better Q&A pairs** - Add questions specific to your learning goals  
3. **Fine-tune a model** - Use the Colab notebook for better Sanskrit responses
4. **Add audio practice** - Upload pronunciation samples for feedback

## 💡 **Tips**

- **Start small**: Use the template data first, then expand
- **Test frequently**: Run validation after each change
- **Check citations**: Make sure [passage_id] references are exact
- **Monitor memory**: Local models need 6-16GB RAM depending on size
- **Use GPU**: Set `n_gpu_layers: 35` in config for faster local inference (if you have GPU)

**Ready to start learning Sanskrit? Let's go! 🕉️**
