# 📚 Sanskrit Data Sources Guide

This guide helps you find and prepare high-quality Sanskrit texts and Q&A data for the Sanskrit Tutor system.

## 🎯 **What You Need**

1. **Sanskrit Passages** (passages.jsonl) - 50+ verses/passages minimum
2. **Q&A Pairs** (qa_pairs.jsonl) - 20+ question-answer pairs minimum  
3. **Configuration** (config.yaml) - Settings for your setup

## 📖 **Best Sanskrit Text Sources**

### **Free & Legal Sources**

#### 1. **GRETIL (Göttingen Register of Electronic Texts in Indian Languages)**
- **URL**: http://gretil.sub.uni-goettingen.de/
- **Content**: Extensive collection of Sanskrit texts
- **Format**: Plain text, needs conversion to JSONL
- **Best For**: Vedas, Upanishads, classical literature
- **License**: Academic use, check individual texts

#### 2. **Sacred Texts Archive**  
- **URL**: https://www.sacred-texts.com/hin/
- **Content**: English translations with some Sanskrit
- **Format**: HTML, needs extraction
- **Best For**: Bhagavad Gita, Upanishads, Puranas
- **License**: Public domain

#### 3. **Wikisource Sanskrit**
- **URL**: https://sa.wikisource.org/
- **Content**: Various Sanskrit texts in Devanagari
- **Format**: Wiki markup, needs conversion
- **Best For**: Verified texts with good formatting
- **License**: CC-BY-SA

#### 4. **Digital Corpus of Sanskrit (DCS)**
- **URL**: http://www.sanskrit-linguistics.org/dcs/
- **Content**: Tagged and analyzed Sanskrit texts
- **Format**: XML/TEI, complex but high quality
- **Best For**: Linguistically analyzed texts
- **License**: Academic research

#### 5. **Vedabase**
- **URL**: https://vedabase.com/
- **Content**: Bhaktivedanta translations with Sanskrit
- **Format**: HTML with Sanskrit verses
- **Best For**: Bhagavad Gita, Srimad Bhagavatam
- **License**: Check terms of use

### **Recommended Starting Texts**

#### **Beginner Level (Easy to Process)**
1. **Bhagavad Gita** - 700 verses, well-documented
2. **Ishavasya Upanishad** - 18 verses, fundamental concepts
3. **Gayatri Mantra & Basic Mantras** - Essential prayers
4. **Hitopadesha selections** - Moral teachings
5. **Subhashitas** - Sanskrit sayings and wisdom

#### **Intermediate Level**
1. **Major Upanishads** - Katha, Kena, Mandukya
2. **Yoga Sutras of Patanjali** - 196 sutras
3. **Mahabharata selections** - Key episodes and teachings
4. **Ramayana selections** - Important verses and passages

#### **Advanced Level**
1. **Vedic Hymns** - Rig Veda selections
2. **Advaita Vedanta texts** - Shankaracharya works
3. **Classical poetry** - Kalidasa, Bhartrhari
4. **Philosophical treatises** - Various darshanas

## 🛠️ **How to Convert Texts to Required Format**

### **Step 1: Extract Sanskrit Text**

Use this Python script to extract from various sources:

```python
import re
import json
import uuid

def extract_verse(text_devanagari, text_iast, work, chapter, verse, source_url, notes=""):
    """Convert a verse to required JSONL format"""
    return {
        "id": f"{work.lower()}_{chapter}_{verse}_{str(uuid.uuid4())[:8]}",
        "text_devanagari": text_devanagari.strip(),
        "text_iast": text_iast.strip(),
        "work": work,
        "chapter": str(chapter),
        "verse": str(verse),
        "language": "sanskrit",
        "source_url": source_url,
        "notes": notes
    }

# Example usage:
verse = extract_verse(
    text_devanagari="धर्मक्षेत्रे कुरुक्षेत्रे समवेता युयुत्सवः",
    text_iast="dharmakṣetre kurukṣetre samavetā yuyutsavaḥ",
    work="Bhagavadgita",
    chapter="1",
    verse="1a",
    source_url="https://www.sacred-texts.com/hin/sbe08/sbe08003.htm",
    notes="Opening verse - Dhritarashtra speaks"
)

# Save to JSONL
with open("passages.jsonl", "a", encoding="utf-8") as f:
    f.write(json.dumps(verse, ensure_ascii=False) + "\n")
```

### **Step 2: Create Q&A Pairs**

Generate questions based on your passages:

```python
def create_qa_pair(question, answer, difficulty, related_passage_ids):
    """Create a Q&A pair with proper citations"""
    return {
        "id": f"qa_{str(uuid.uuid4())[:8]}",
        "question": question,
        "answer": answer,  # Must include [passage_id] citations
        "difficulty": difficulty,  # "easy", "medium", "hard"
        "related_passage_ids": related_passage_ids
    }

# Example:
qa = create_qa_pair(
    question="What is the opening verse of the Bhagavad Gita?",
    answer="The Gita opens with 'dharmakṣetre kurukṣetre samavetā yuyutsavaḥ' - On the field of dharma, on the field of Kuru, assembled and eager to fight. [bg_001_001]",
    difficulty="easy",
    related_passage_ids=["bg_001_001"]
)
```

## 🔧 **GGUF Model Sources**

### **Recommended Models for Sanskrit**

#### **Best Overall: Mistral-7B-Instruct**
- **Download**: https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF
- **File**: `mistral-7b-instruct-v0.1.q4_k_m.gguf` (4GB)
- **Why**: Good instruction following, reasonable size
- **Place at**: `user_assets/models/mistral-7b-instruct.gguf`

#### **Smaller Option: CodeLlama-7B**
- **Download**: https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF  
- **File**: `codellama-7b-instruct.q4_k_m.gguf`
- **Why**: Good at structured responses
- **Place at**: `user_assets/models/codellama-7b-instruct.gguf`

#### **Larger Option: Llama-2-13B** (if you have 16GB+ RAM)
- **Download**: https://huggingface.co/TheBloke/Llama-2-13B-chat-GGUF
- **File**: `llama-2-13b-chat.q4_k_m.gguf` (8GB)
- **Why**: Better quality responses
- **Place at**: `user_assets/models/llama-2-13b-chat.gguf`

### **How to Download GGUF Models**

```bash
# Method 1: Direct download with wget/curl
wget -O user_assets/models/mistral-7b-instruct.gguf \
  "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.q4_k_m.gguf"

# Method 2: Using huggingface-hub
pip install huggingface-hub
huggingface-cli download TheBloke/Mistral-7B-Instruct-v0.1-GGUF \
  mistral-7b-instruct-v0.1.q4_k_m.gguf \
  --local-dir user_assets/models/ \
  --local-dir-use-symlinks False
```

## 🎯 **Quick Start Datasets**

### **Option 1: Bhagavad Gita Focus** (Recommended for beginners)
- **~100 key verses** from Bhagavad Gita
- **~50 Q&A pairs** covering main concepts
- **Themes**: Karma yoga, dharma, devotion, knowledge
- **Time**: ~4-6 hours to prepare

### **Option 2: Upanishads Collection**
- **~50 verses** from major Upanishads
- **~30 Q&A pairs** on philosophical concepts  
- **Themes**: Atman, Brahman, meditation, liberation
- **Time**: ~6-8 hours to prepare

### **Option 3: Mixed Classical** (Most comprehensive)
- **~200 passages** from various sources
- **~100 Q&A pairs** covering diverse topics
- **Themes**: Philosophy, ethics, spirituality, culture
- **Time**: ~15-20 hours to prepare

## ⚡ **Quick Setup Commands**

Once you have your data files ready:

```bash
# 1. Copy template files
cp user_assets/passages_template.jsonl user_assets/passages.jsonl
cp user_assets/qa_pairs_template.jsonl user_assets/qa_pairs.jsonl  
cp user_assets/config_template.yaml user_assets/config.yaml

# 2. Edit the files with your data
# (Use any text editor to add your Sanskrit passages and questions)

# 3. Download a GGUF model (example)
mkdir -p user_assets/models
wget -O user_assets/models/mistral-7b-instruct.gguf \
  "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.q4_k_m.gguf"

# 4. Install and run
pip install llama-cpp-python
python src/utils/config_validator.py
python src/embed_index.py --config user_assets/config.yaml
python src/ui_gradio.py --config user_assets/config.yaml
```

## 🚨 **Legal & Ethical Guidelines**

### **✅ Safe to Use**
- Public domain texts (pre-1923 in US)
- Creative Commons licensed materials
- Academic fair use excerpts
- Traditional texts without modern copyright

### **⚠️ Check First**
- Modern translations and commentaries
- Published editions with editorial work
- Copyrighted scholarly editions
- Commercial textbooks

### **❌ Avoid**
- Copyrighted modern works
- Proprietary databases
- Commercial software content
- Materials without clear licensing

## 💡 **Tips for High-Quality Data**

### **Text Quality**
- Use authentic, scholarly sources
- Verify Sanskrit spellings and IAST
- Include proper diacritical marks
- Cross-check multiple sources

### **Q&A Quality**  
- Make questions pedagogical and clear
- Include proper citations in answers
- Cover different difficulty levels
- Test questions with real learners

### **Technical Quality**
- Validate JSON format before use
- Use consistent ID schemes
- Include meaningful source URLs
- Add helpful notes and context

## 🔍 **Validation Checklist**

Before using your data:

- [ ] passages.jsonl has proper JSONL format
- [ ] All required fields present in each passage
- [ ] qa_pairs.jsonl references valid passage IDs
- [ ] Citations in answers match passage IDs exactly
- [ ] config.yaml has all required settings
- [ ] GGUF model downloaded and placed correctly
- [ ] Text encoding is UTF-8
- [ ] No duplicate IDs in passages or QA pairs

## 🆘 **Common Issues & Solutions**

### **"Invalid JSON" errors**
- Check for unescaped quotes in text
- Ensure proper UTF-8 encoding
- Validate each line separately

### **"Citation not found" warnings**  
- Check passage ID spelling in Q&A answers
- Ensure passage IDs are unique
- Use exact ID format: [passage_id]

### **"Model not loading" errors**
- Verify GGUF file integrity
- Check available RAM (need 8GB+ for 7B models)
- Try smaller quantized versions (q4_k_s vs q4_k_m)

### **"Embedding model download fails"**
- Check internet connection
- Try different embedding model
- Use smaller models like all-MiniLM-L6-v2

---

**Need help?** Open an issue on GitHub with your specific data source questions!
