#!/usr/bin/env python3
"""
Sanskrit Query Normalizer - Proper transliteration and normalization for Sanskrit text.
This addresses the limitation of basic pattern matching in the previous system.
"""

import re
import unicodedata
from typing import Dict, List, Tuple
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate


class SanskritNormalizer:
    """Proper Sanskrit text normalization and transliteration."""
    
    def __init__(self):
        self.devanagari_pattern = re.compile(r'[\u0900-\u097F]+')
        self.iast_pattern = re.compile(r'[aāiīuūṛṝḷḹeēoōṃḥkgṅcjñṭḍṇtdnpbmyrlvśṣsh]+')
        
    def normalize_unicode(self, text: str) -> str:
        """Normalize Unicode text using NFKC form."""
        return unicodedata.normalize('NFKC', text.strip())
    
    def detect_script(self, text: str) -> str:
        """
        Detect script type with better accuracy.
        Returns: 'devanagari', 'iast', 'english'
        """
        text = self.normalize_unicode(text)
        
        # Check for Devanagari characters
        if self.devanagari_pattern.search(text):
            return 'devanagari'
        
        # Check for IAST diacritics (more specific than before)
        iast_chars = set('āīūṛṝḷḹēōṃḥṅñṭḍṇśṣ')
        if any(char in text for char in iast_chars):
            return 'iast'
        
        # Check if it looks like romanized Sanskrit
        sanskrit_words = ['karma', 'dharma', 'yoga', 'atma', 'brahma', 'moksa', 'guru']
        text_lower = text.lower()
        if any(word in text_lower for word in sanskrit_words):
            return 'iast'
        
        return 'english'
    
    def transliterate_to_iast(self, text: str, source_script: str = None) -> str:
        """Convert text to IAST transliteration."""
        if not source_script:
            source_script = self.detect_script(text)
        
        try:
            if source_script == 'devanagari':
                return transliterate(text, sanscript.DEVANAGARI, sanscript.IAST)
            elif source_script == 'iast':
                return text  # Already IAST
            else:
                return text  # English or unknown
        except Exception as e:
            print(f"Transliteration error: {e}")
            return text
    
    def transliterate_to_devanagari(self, text: str, source_script: str = None) -> str:
        """Convert text to Devanagari script."""
        if not source_script:
            source_script = self.detect_script(text)
        
        try:
            if source_script == 'iast':
                return transliterate(text, sanscript.IAST, sanscript.DEVANAGARI)
            elif source_script == 'devanagari':
                return text  # Already Devanagari
            else:
                # Try to interpret as basic romanization
                basic_mappings = {
                    'karma': 'कर्म', 'dharma': 'धर्म', 'yoga': 'योग',
                    'atma': 'आत्मा', 'brahma': 'ब्रह्म', 'moksa': 'मोक्ष'
                }
                for rom, dev in basic_mappings.items():
                    text = re.sub(r'\b' + rom + r'\b', dev, text, flags=re.IGNORECASE)
                return text
        except Exception as e:
            print(f"Transliteration error: {e}")
            return text
    
    def normalize_query(self, query: str) -> Dict[str, str]:
        """
        Normalize a query into multiple forms for better retrieval.
        Returns dict with 'original', 'iast', 'devanagari', 'normalized' keys.
        """
        original = self.normalize_unicode(query)
        script = self.detect_script(original)
        
        # Generate IAST form
        iast = self.transliterate_to_iast(original, script)
        
        # Generate Devanagari form
        devanagari = self.transliterate_to_devanagari(original, script)
        
        # Create normalized form (lowercase IAST, remove punctuation)
        normalized = re.sub(r'[^\w\s]', '', iast.lower())
        
        return {
            'original': original,
            'script': script,
            'iast': iast,
            'devanagari': devanagari,
            'normalized': normalized
        }
    
    def create_composite_text(self, iast: str, devanagari: str) -> str:
        """Create composite text for embedding (IAST + Devanagari)."""
        return f"{iast} ||| {devanagari}"


def test_normalizer():
    """Test the Sanskrit normalizer."""
    normalizer = SanskritNormalizer()
    
    test_cases = [
        "कः अस्ति धर्मः?",  # Devanagari
        "kaḥ asti dharmaḥ?",  # IAST
        "What is dharma?",    # English
        "karma yoga",         # Romanized
    ]
    
    print("🔤 Testing Sanskrit Normalizer:")
    print("=" * 50)
    
    for query in test_cases:
        result = normalizer.normalize_query(query)
        print(f"\nInput: {query}")
        print(f"Script: {result['script']}")
        print(f"IAST: {result['iast']}")
        print(f"Devanagari: {result['devanagari']}")
        print(f"Normalized: {result['normalized']}")
        print(f"Composite: {normalizer.create_composite_text(result['iast'], result['devanagari'])}")


if __name__ == "__main__":
    test_normalizer()
