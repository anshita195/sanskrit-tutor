#!/usr/bin/env python3
"""
Test Sanskrit conversational functionality.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from sanskrit_conversation import SanskritConversationalBot, SanskritLanguageProcessor

def test_sanskrit_conversation():
    """Test Sanskrit conversation capabilities."""
    
    print("🕉️ Testing Sanskrit Conversational AI...")
    print("=" * 50)
    
    # Initialize language processor
    processor = SanskritLanguageProcessor()
    
    # Test Sanskrit questions
    test_inputs = [
        "नमस्ते!",
        "कः अस्ति धर्मः?",
        "What is dharma?",
        "योगः किम्?",
        "How does one achieve moksha?"
    ]
    
    print("Testing Sanskrit Language Processing:")
    print("-" * 35)
    
    for text in test_inputs:
        print(f"\nInput: {text}")
        
        # Detect script
        script = processor.detect_script(text)
        print(f"Script: {script}")
        
        # Extract vocabulary (if Sanskrit)
        if script in ["devanagari", "iast"]:
            vocab = processor.extract_sanskrit_vocabulary(text)
            print(f"Vocabulary: {', '.join(vocab)}")
        
        # Check if it's a question
        is_question = processor.is_sanskrit_question(text)
        print(f"Question: {is_question}")
        
        # Basic translation attempt
        if script in ["devanagari", "iast"]:
            translation = processor.simple_sanskrit_to_english(text)
            print(f"Translation: {translation}")
    
    print("\n" + "=" * 50)
    print("✅ Sanskrit Language Processing Test Complete!")
    print("\nTo test full conversation:")
    print("python src/sanskrit_chat_ui.py")
    print("Then try: कः अस्ति धर्मः?")


if __name__ == "__main__":
    test_sanskrit_conversation()
