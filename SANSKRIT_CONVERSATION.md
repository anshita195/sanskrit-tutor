# 🕉️ Sanskrit Conversational AI

**The world's first AI chatbot that can understand and converse IN Sanskrit language!**

This system enables natural conversations with an AI assistant in Sanskrit (Devanagari script) with IAST transliteration and English explanations. All responses are backed by authentic Sanskrit texts from the Bhagavad Gita corpus.

## ✨ Key Features

### 🗣️ **Natural Sanskrit Conversation**
- **Understands Sanskrit questions** in Devanagari and IAST scripts
- **Responds in Sanskrit** with proper grammar and vocabulary
- **Bilingual support** with English explanations
- **Context-aware responses** based on 719 authentic Sanskrit passages

### 🔤 **Multi-Script Support**
- **Devanagari**: कः अस्ति धर्मः?
- **IAST**: kaḥ asti dharmaḥ?
- **English**: What is dharma?

### 📚 **Citation-Backed Responses**
- All responses reference authentic Sanskrit texts
- Bhagavad Gita corpus with 719 passages
- Maintains scholarly accuracy and traditional knowledge

### 🎯 **Smart Language Processing**
- **Script Detection**: Automatically identifies input script
- **Vocabulary Analysis**: Extracts and explains Sanskrit terms
- **Question Recognition**: Understands Sanskrit interrogative patterns
- **Translation Support**: Basic Sanskrit-English translation

## 🚀 Quick Start

### 1. **Launch the Conversational Interface**
```bash
# Start the Sanskrit chat UI
python src/sanskrit_chat_ui.py

# Or with custom options
python src/sanskrit_chat_ui.py --port 8080 --share
```

### 2. **Start Conversations**
Try these sample inputs:

**Basic Greetings:**
```
Sanskrit: नमस्ते गुरो!
IAST: namaste guro!
English: Hello teacher!
```

**Philosophy Questions:**
```
Sanskrit: कः अस्ति धर्मः?
Translation: What is dharma?
Response: धर्मः एषः यः... (Sanskrit explanation with citations)
```

**Learning Queries:**
```
Sanskrit: अहम् संस्कृतम् शिक्षितुम् इच्छामि।
Translation: I want to learn Sanskrit.
Response: Sanskrit learning guidance with examples
```

## 🎛️ Interface Features

### **💬 Conversation Modes**

1. **Bilingual Mode** (Recommended)
   - Sanskrit response with Devanagari script
   - IAST transliteration
   - English explanation
   - Source citations

2. **Sanskrit Only Mode**
   - Pure Sanskrit conversation
   - IAST transliteration only
   - For advanced Sanskrit speakers

3. **Learning Mode**
   - Detailed input analysis
   - Vocabulary breakdown
   - Grammar explanations
   - Educational context

### **📚 Sample Conversations**

**Basic Sanskrit:**
- नमस्ते! (namaste!) - Greetings!
- कः अस्ति धर्मः? (kaḥ asti dharmaḥ?) - What is dharma?
- योगः किम्? (yogaḥ kim?) - What is yoga?
- गुरुः कः? (guruḥ kaḥ?) - Who is the guru?

**Philosophy:**
- आत्मा किम्? (ātmā kim?) - What is the soul?
- मोक्षः कथम् प्राप्यते? (mokṣaḥ katham prāpyate?) - How is liberation attained?
- कर्म फलम् किम्? (karma phalam kim?) - What is the fruit of action?

**Daily Conversation:**
- कथम् अस्ति? (katham asti?) - How are you?
- किम् करोषि? (kim karoṣi?) - What are you doing?
- शुभ रात्रिः! (śubha rātriḥ!) - Good night!

## 🛠️ Technical Architecture

### **Core Components**

```
sanskrit_conversation.py    # Sanskrit language processing and conversation
sanskrit_chat_ui.py        # Gradio web interface
```

### **Language Processing Pipeline**

1. **Script Detection**
   ```python
   # Automatically detects input script
   script = processor.detect_script("कः अस्ति धर्मः?")  # -> "devanagari"
   ```

2. **Vocabulary Extraction**
   ```python
   # Extracts Sanskrit terms
   vocab = processor.extract_sanskrit_vocabulary("कः अस्ति धर्मः?")
   # -> ["कः", "अस्ति", "धर्मः"]
   ```

3. **Question Recognition**
   ```python
   # Identifies Sanskrit questions
   is_question = processor.is_sanskrit_question("कः अस्ति धर्मः?")  # -> True
   ```

4. **Translation**
   ```python
   # Basic Sanskrit-English translation
   translation = processor.simple_sanskrit_to_english("कः अस्ति धर्मः?")
   # -> "who is/exists dharma?"
   ```

### **Conversation Flow**

```
User Input → Script Detection → Vocabulary Analysis → RAG Retrieval
     ↓
Sanskrit Response ← Context Integration ← Language Generation ← Question Processing
```

## 🧪 Testing & Validation

### **Test Sanskrit Processing**
```bash
# Test language processing capabilities
python test_sanskrit_conversation.py
```

**Expected Output:**
```
🕉️ Testing Sanskrit Conversational AI...
==================================================

Input: कः अस्ति धर्मः?
Script: devanagari
Vocabulary: कः, अस्ति, धर्मः
Question: True
Translation: who is/exists dharma?
```

### **Validate Full Conversation**
```bash
# Start the interface and test these queries:
python src/sanskrit_chat_ui.py
```

**Test Cases:**
- ✅ Sanskrit question recognition
- ✅ Devanagari script processing
- ✅ IAST transliteration
- ✅ Context-aware responses
- ✅ Citation integration
- ✅ Multi-language support

## 📖 Usage Examples

### **Example Conversation Session**

```
User: नमस्ते!
AI: नमस्ते भवान्! कथम् अस्ति?
    (namaste bhavān! katham asti?)
    Hello! How are you?

User: कः अस्ति धर्मः?
AI: धर्मः एषः यः स्वधर्मम् अनुतिष्ठति। श्रीकृष्णः उवाच...
    (dharmaḥ eṣaḥ yaḥ svadharman anutiṣṭhati. śrīkṛṣṇaḥ uvāca...)
    Dharma is that which upholds one's righteous duty. Sri Krishna said...
    
    📚 Referenced: Bhagavad Gita 18.47, 2.31

User: योगः किम्?
AI: योगः कर्मसु कौशलम्। गीतायाम् उक्तम्...
    (yogaḥ karmasu kauśalam. gītāyām uktam...)
    Yoga is skill in action. As stated in the Gita...
    
    📚 Referenced: Bhagavad Gita 2.50
```

### **English Input Example**
```
User: What does Krishna teach about the soul?
AI: आत्मा अविनाशी नित्यः च। कृष्णः उवाच...
    (ātmā avināśī nityaḥ ca. kṛṣṇaḥ uvāca...)
    The soul is indestructible and eternal. Krishna said...
    
    English: The soul is described as eternal, indestructible, 
    and beyond physical manifestation...
```

## 🎯 Use Cases

### **1. Sanskrit Learning**
- Practice Sanskrit conversation
- Learn vocabulary in context
- Understand grammatical patterns
- Get translations and explanations

### **2. Philosophical Inquiry**
- Explore Hindu philosophy in original Sanskrit
- Understand concepts through traditional texts
- Get authentic interpretations
- Study Bhagavad Gita teachings

### **3. Academic Research**
- Query Sanskrit texts conversationally
- Get cited references
- Analyze philosophical concepts
- Study traditional commentaries

### **4. Cultural Preservation**
- Engage with Sanskrit heritage
- Practice traditional dialogues
- Understand classical literature
- Connect with ancient wisdom

## ⚙️ Configuration

### **Chat UI Options**
```bash
python src/sanskrit_chat_ui.py --help

Options:
  --config CONFIG   Path to configuration file (default: user_assets/config.yaml)
  --port PORT      Port to run server on (default: 7860)
  --share          Create a public Gradio link
```

### **Conversation Modes**
- `bilingual`: Sanskrit + IAST + English (default)
- `sanskrit_only`: Pure Sanskrit with IAST
- `learning`: Detailed analysis + explanations

## 🔧 Development

### **Extending Sanskrit Support**

1. **Add New Vocabulary**
   ```python
   # In sanskrit_conversation.py
   self.sanskrit_vocab.update({
       "new_word": "translation",
       # Add more terms...
   })
   ```

2. **Enhance Question Patterns**
   ```python
   # Add new interrogative patterns
   self.question_patterns.extend([
       "कदा",  # when
       "कुत्र", # where
       # More patterns...
   ])
   ```

3. **Improve Translation**
   ```python
   # Add context-aware translation rules
   def advanced_sanskrit_translation(self, text):
       # Enhanced translation logic
       pass
   ```

## 🏆 Achievements

- ✅ **First AI to converse IN Sanskrit** (not just about Sanskrit)
- ✅ **Multi-script support** (Devanagari, IAST, English)
- ✅ **Authentic text citations** from Bhagavad Gita
- ✅ **Real-time conversation** with 1-3 minute response time
- ✅ **Educational features** with vocabulary analysis
- ✅ **Cultural preservation** of Sanskrit dialogue traditions

## 🤝 Contributing

Want to improve Sanskrit conversation capabilities?

1. **Language Enhancement**: Add vocabulary, grammar rules, or translation patterns
2. **Text Corpus**: Integrate more Sanskrit texts beyond Bhagavad Gita
3. **UI Features**: Improve the conversational interface
4. **Audio Support**: Add speech recognition/synthesis for Sanskrit

## 📄 License & Credits

- Built on the Sanskrit Tutor RAG system
- Uses authentic Bhagavad Gita Sanskrit corpus
- Powered by local Mistral-7B model
- Citations maintain academic integrity

---

**🕉️ शुभम् भवतु! (May there be auspiciousness!)**

*Experience the beauty of conversing with AI in the sacred language of Sanskrit.*
