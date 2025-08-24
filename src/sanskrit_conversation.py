#!/usr/bin/env python3
"""
Sanskrit Conversational Chatbot Module.
Enables natural conversation in Sanskrit language with understanding and response generation.
"""

import re
import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

from rag import SanskritRAG, RAGResponse, RetrievalResult


@dataclass
class SanskritConversation:
    """Represents a Sanskrit conversation turn."""
    user_input: str          # User's Sanskrit input
    user_input_iast: str     # IAST transliteration
    translation: str         # English translation
    bot_response: str        # Bot's Sanskrit response
    bot_response_iast: str   # Bot's IAST response
    bot_translation: str     # English translation of bot response
    context_passages: List[str]  # Supporting passages
    conversation_id: str
    timestamp: float


class SanskritLanguageProcessor:
    """Handles Sanskrit language processing, translation, and generation."""
    
    def __init__(self):
        self.devanagari_pattern = re.compile(r'[\u0900-\u097F]+')
        self.iast_pattern = re.compile(r'[aāiīuūṛṝḷḹeēoōaṃḥkgṅcjñṭḍṇtdnpbmyrlvśṣshḵḡṅ̇ĉĵñ̇ṭ̇ḍ̇ṇ̇t̄d̄n̄]+')
    
    def detect_script(self, text: str) -> str:
        """Detect if text is in Devanagari, IAST, or English."""
        text = text.strip()
        
        if self.devanagari_pattern.search(text):
            return "devanagari"
        elif self.iast_pattern.search(text):
            return "iast"
        else:
            return "english"
    
    def is_sanskrit_question(self, text: str) -> bool:
        """Check if the input appears to be a Sanskrit question."""
        script = self.detect_script(text)
        
        if script == "devanagari":
            # Check for question patterns in Devanagari
            question_markers = ['कः', 'का', 'किम्', 'कुत्र', 'कदा', 'कथम्', 'के', 'काः']
            return any(marker in text for marker in question_markers)
        
        elif script == "iast":
            # Check for question patterns in IAST
            question_markers = ['kaḥ', 'kā', 'kim', 'kutra', 'kadā', 'katham', 'ke', 'kāḥ']
            return any(marker in text.lower() for marker in question_markers)
        
        return False
    
    def extract_sanskrit_vocabulary(self, text: str) -> List[str]:
        """Extract Sanskrit words for vocabulary learning."""
        script = self.detect_script(text)
        
        if script == "devanagari":
            # Extract Devanagari words
            words = re.findall(r'[\u0900-\u097F]+', text)
            return [word for word in words if len(word) > 1]
        
        elif script == "iast":
            # Extract IAST words
            words = re.findall(r'\b[aāiīuūṛṝḷḹeēoōaṃḥkgṅcjñṭḍṇtdnpbmyrlvśṣsh]+\b', text)
            return words
        
        return []
    
    def simple_sanskrit_to_english(self, sanskrit_text: str) -> str:
        """Basic Sanskrit to English translation using common patterns."""
        # This is a simplified approach - in a full system you'd use proper translation models
        
        common_translations = {
            # Questions
            'कः': 'who', 'का': 'who (fem)', 'किम्': 'what', 'कुत्र': 'where', 
            'कदा': 'when', 'कथम्': 'how', 'केन': 'by whom', 'कस्मात्': 'why',
            
            # Common words
            'धर्मः': 'dharma/righteousness', 'कर्म': 'action/karma', 'योगः': 'yoga',
            'ज्ञानम्': 'knowledge', 'भक्तिः': 'devotion', 'मोक्षः': 'liberation',
            'आत्मा': 'soul/self', 'ब्रह्म': 'Brahman/ultimate reality',
            'गुरुः': 'teacher', 'शिष्यः': 'student', 'मन्त्रम्': 'mantra',
            
            # Verbs
            'अस्ति': 'is/exists', 'भवति': 'becomes', 'करोति': 'does', 
            'गच्छति': 'goes', 'आगच्छति': 'comes', 'तिष्ठति': 'stands',
            
            # Common phrases
            'नमस्ते': 'greetings/salutations', 'धन्यवादः': 'thank you',
            'कथम् अस्ति': 'how are you', 'अहम् जानामि': 'I know'
        }
        
        translation_parts = []
        words = sanskrit_text.split()
        
        for word in words:
            clean_word = re.sub(r'[।॥\s]', '', word)
            if clean_word in common_translations:
                translation_parts.append(common_translations[clean_word])
            else:
                translation_parts.append(f"({clean_word})")
        
        return " ".join(translation_parts) if translation_parts else "Sanskrit text"


class SanskritConversationalBot:
    """
    Main Sanskrit conversational chatbot that can understand and respond in Sanskrit.
    """
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.rag_system = None
        self.language_processor = SanskritLanguageProcessor()
        self.conversation_history = []
        
        # Sanskrit conversation templates
        self.sanskrit_templates = {
            'greeting': [
                'नमस्ते! अहम् संस्कृत-शिक्षकः अस्मि।',
                'स्वागतम्! किम् प्रश्नः अस्ति?',
                'आगतोस्मि सहायार्थम्। किम् इच्छसि ज्ञातुम्?'
            ],
            'acknowledgment': [
                'आम्, एतत् उत्तमं प्रश्नः।',
                'अहम् अवगच्छामि।',
                'साधु! एतत् महत्वपूर्णं विषयः।'
            ],
            'explanation_intro': [
                'श्रृणु, अहम् व्याख्यास्यामि।',
                'अत्र उत्तरम् अस्ति।',
                'गीतायाम् एतत् प्रोक्तम्।'
            ],
            'citation_intro': [
                'यथा उक्तम् ग्रन्थे:',
                'शास्त्रे दृश्यते:',
                'आचार्यैः प्रोक्तम्:'
            ],
            'closing': [
                'एतत् तव प्रश्नस्य उत्तरम्।',
                'किम् अन्यत् जिज्ञासा अस्ति?',
                'इति मम विचारः।'
            ]
        }
    
    def initialize(self) -> bool:
        """Initialize the conversational system."""
        try:
            print("Initializing Sanskrit Conversational Bot...")
            
            # Initialize the underlying RAG system
            self.rag_system = SanskritRAG(self.config_path)
            if not self.rag_system.initialize():
                return False
            
            print("Sanskrit Conversational Bot initialized successfully!")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to initialize conversational bot: {str(e)}")
            return False
    
    def create_sanskrit_prompt(self, user_input: str, retrieved_passages: List[RetrievalResult]) -> str:
        """Create a prompt that encourages Sanskrit response."""
        
        # Detect input language
        script = self.language_processor.detect_script(user_input)
        
        # Create context from passages
        context_parts = []
        for result in retrieved_passages:
            passage = result.passage
            context_parts.append(f"""
Reference [{passage['id']}]:
Sanskrit: {passage['text_devanagari']}
IAST: {passage['text_iast']}
Work: {passage['work']} {passage['chapter']}.{passage['verse']}
""")
        
        context = "\n".join(context_parts)
        
        # Create Sanskrit conversation prompt
        sanskrit_prompt = f"""You are a Sanskrit conversation tutor who responds primarily in Sanskrit with supporting explanations. 

CONVERSATION RULES:
1. If user asks in Sanskrit (Devanagari or IAST), respond primarily in Sanskrit
2. If user asks in English, you may respond in Sanskrit with English explanation
3. Always include proper Devanagari script in your response
4. Use simple, clear Sanskrit appropriate for learners
5. Include IAST transliteration when helpful
6. Provide English translation/explanation after Sanskrit response

SANSKRIT RESPONSE STRUCTURE:
- Start with appropriate Sanskrit greeting/acknowledgment
- Give main answer in Sanskrit
- Include relevant citations from provided texts
- End with English explanation if needed

AVAILABLE CONTEXT:
{context}

USER INPUT: {user_input}

Please respond with:
1. Sanskrit response (Devanagari)
2. IAST transliteration 
3. English explanation
4. Relevant citations from the provided passages

Example format:
Sanskrit: नमस्ते! धर्मः एषः अस्ति... [citation]
IAST: namaste! dharmaḥ eṣaḥ asti...
English: Greetings! Dharma is this...
"""
        
        return sanskrit_prompt
    
    def process_sanskrit_conversation(self, user_input: str) -> SanskritConversation:
        """Process a complete Sanskrit conversation turn."""
        
        start_time = time.time()
        conversation_id = f"conv_{int(start_time)}"
        
        try:
            # Detect script and translate if needed
            script = self.language_processor.detect_script(user_input)
            
            if script == "devanagari":
                user_input_iast = user_input  # Would need proper transliteration
                translation = self.language_processor.simple_sanskrit_to_english(user_input)
            else:
                user_input_iast = user_input
                translation = user_input  # Assume English input
            
            # Get relevant passages
            search_query = translation if script == "devanagari" else user_input
            retrieved_passages = self.rag_system.retrieve_passages(search_query, k=3)
            
            # Create Sanskrit conversation prompt
            prompt = self.create_sanskrit_prompt(user_input, retrieved_passages)
            
            # Generate Sanskrit response
            try:
                response = self.rag_system.llm_manager.generate(
                    prompt=prompt,
                    max_tokens=300,
                    temperature=0.7,
                    stop_sequences=["USER INPUT:", "AVAILABLE CONTEXT:"]
                )
                
                # Parse the response to extract Sanskrit, IAST, and English parts
                bot_response, bot_response_iast, bot_translation = self.parse_sanskrit_response(response)
                
            except Exception as e:
                # Fallback to simple Sanskrit response
                bot_response = "क्षम्यताम्, अहम् न अवगच्छामि।"
                bot_response_iast = "kṣamyatām, aham na avagacchāmi"
                bot_translation = "Sorry, I don't understand."
            
            # Create conversation object
            conversation = SanskritConversation(
                user_input=user_input,
                user_input_iast=user_input_iast,
                translation=translation,
                bot_response=bot_response,
                bot_response_iast=bot_response_iast,
                bot_translation=bot_translation,
                context_passages=[p.passage_id for p in retrieved_passages],
                conversation_id=conversation_id,
                timestamp=time.time()
            )
            
            self.conversation_history.append(conversation)
            return conversation
            
        except Exception as e:
            # Error handling with Sanskrit response
            return SanskritConversation(
                user_input=user_input,
                user_input_iast="",
                translation="",
                bot_response="क्षम्यताम्, त्रुटिः जातः।",
                bot_response_iast="kṣamyatām, truṭiḥ jātaḥ",
                bot_translation=f"Sorry, an error occurred: {str(e)}",
                context_passages=[],
                conversation_id=conversation_id,
                timestamp=time.time()
            )
    
    def parse_sanskrit_response(self, response: str) -> Tuple[str, str, str]:
        """Parse LLM response to extract Sanskrit, IAST, and English parts."""
        
        # Try to extract structured response
        sanskrit_match = re.search(r'Sanskrit:\s*(.+?)(?:\n|IAST:|$)', response, re.IGNORECASE)
        iast_match = re.search(r'IAST:\s*(.+?)(?:\n|English:|$)', response, re.IGNORECASE)
        english_match = re.search(r'English:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        
        sanskrit_text = sanskrit_match.group(1).strip() if sanskrit_match else ""
        iast_text = iast_match.group(1).strip() if iast_match else ""
        english_text = english_match.group(1).strip() if english_match else ""
        
        # Fallback: extract first Devanagari text found
        if not sanskrit_text:
            devanagari_matches = self.devanagari_pattern.findall(response)
            if devanagari_matches:
                sanskrit_text = devanagari_matches[0]
        
        # Fallback: use entire response if no structure found
        if not sanskrit_text and not english_text:
            english_text = response.strip()
            sanskrit_text = "धन्यवादः।"  # Thank you
            iast_text = "dhanyavādaḥ"
        
        return sanskrit_text, iast_text, english_text
    
    def generate_simple_sanskrit_response(self, topic: str, passages: List[RetrievalResult]) -> Tuple[str, str, str]:
        """Generate a simple Sanskrit response based on topic and passages."""
        
        if not passages:
            return (
                "अहम् न जानामि। क्षम्यताम्।",
                "aham na jānāmi. kṣamyatām.",
                "I don't know. Please forgive me."
            )
        
        # Use the first passage for response
        passage = passages[0].passage
        
        # Create simple Sanskrit explanation
        sanskrit_response = f"गीतायाम् उक्तम् - {passage['text_devanagari']} [{passage['id']}]"
        iast_response = f"gītāyām uktam - {passage['text_iast']}"
        english_response = f"In the Gita it is said - {passage.get('notes', 'Sanskrit text')} [{passage['id']}]"
        
        return sanskrit_response, iast_response, english_response
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get summary of conversation history."""
        if not self.conversation_history:
            return {"total_turns": 0, "sanskrit_inputs": 0, "avg_response_time": 0}
        
        sanskrit_inputs = sum(1 for conv in self.conversation_history 
                            if self.language_processor.detect_script(conv.user_input) in ['devanagari', 'iast'])
        
        avg_time = sum(conv.timestamp for conv in self.conversation_history) / len(self.conversation_history)
        
        return {
            "total_turns": len(self.conversation_history),
            "sanskrit_inputs": sanskrit_inputs,
            "english_inputs": len(self.conversation_history) - sanskrit_inputs,
            "avg_response_time": avg_time,
            "last_conversation": self.conversation_history[-1].conversation_id if self.conversation_history else None
        }


class SanskritConversationUI:
    """Enhanced UI for Sanskrit conversations."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.sanskrit_bot = SanskritConversationalBot(config_path)
        self.conversation_mode = "bilingual"  # "sanskrit_only", "bilingual", "learning"
    
    def process_conversation(self, user_input: str, history: List[List[str]]) -> Tuple[str, List[List[str]]]:
        """Process a Sanskrit conversation turn."""
        
        if not user_input.strip():
            return "", history
        
        try:
            # Process the conversation
            conversation = self.sanskrit_bot.process_sanskrit_conversation(user_input)
            
            # Format response based on mode
            if self.conversation_mode == "sanskrit_only":
                response_text = f"{conversation.bot_response}\n\n*{conversation.bot_response_iast}*"
            
            elif self.conversation_mode == "bilingual":
                response_text = f"""**Sanskrit:** {conversation.bot_response}

**IAST:** {conversation.bot_response_iast}

**English:** {conversation.bot_translation}

📚 Context: {len(conversation.context_passages)} passages referenced"""
            
            else:  # learning mode
                script = self.sanskrit_bot.language_processor.detect_script(user_input)
                response_text = f"""**Your input:** {conversation.user_input}
**Script detected:** {script}
**Translation:** {conversation.translation}

**Sanskrit response:** {conversation.bot_response}
**IAST:** {conversation.bot_response_iast}
**English:** {conversation.bot_translation}

📚 **Learning notes:** Citations from {len(conversation.context_passages)} relevant passages"""
            
            # Update history
            new_history = history + [[user_input, response_text]]
            return "", new_history
            
        except Exception as e:
            error_response = f"क्षम्यताम्! (kṣamyatām!) Sorry, error: {str(e)}"
            new_history = history + [[user_input, error_response]]
            return "", new_history


# Quick test functions
def test_sanskrit_processing():
    """Test Sanskrit language processing capabilities."""
    processor = SanskritLanguageProcessor()
    
    test_inputs = [
        "कः अस्ति धर्मः?",  # Who/What is dharma?
        "kaḥ asti dharmaḥ?",  # IAST version
        "What is dharma?",     # English
        "नमस्ते गुरो!",       # Greetings teacher!
        "किम् भवान् शिक्षयति?" # What do you teach?
    ]
    
    print("Testing Sanskrit Language Processing:")
    print("=" * 50)
    
    for text in test_inputs:
        script = processor.detect_script(text)
        is_question = processor.is_sanskrit_question(text)
        translation = processor.simple_sanskrit_to_english(text)
        vocab = processor.extract_sanskrit_vocabulary(text)
        
        print(f"Input: {text}")
        print(f"Script: {script}")
        print(f"Is Question: {is_question}")
        print(f"Translation: {translation}")
        print(f"Vocabulary: {vocab}")
        print("-" * 30)


if __name__ == "__main__":
    # Test the Sanskrit processing
    test_sanskrit_processing()
    
    # Test with actual system if available
    try:
        print("\nTesting with actual system...")
        bot = SanskritConversationalBot("user_assets/config.yaml")
        if bot.initialize():
            # Test conversation
            test_question = "कः अस्ति धर्मः?"
            conversation = bot.process_sanskrit_conversation(test_question)
            
            print(f"\nSanskrit Conversation Test:")
            print(f"User: {conversation.user_input}")
            print(f"Translation: {conversation.translation}")
            print(f"Bot Sanskrit: {conversation.bot_response}")
            print(f"Bot IAST: {conversation.bot_response_iast}")
            print(f"Bot English: {conversation.bot_translation}")
            
    except Exception as e:
        print(f"Could not test with actual system: {e}")
