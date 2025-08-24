#!/usr/bin/env python3
"""
Multi-Domain Manager for Sanskrit Tutor.
Handles different knowledge domains: Grammar, Literature, Ayurveda, Mathematics, etc.
"""

from enum import Enum
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import yaml


class SanskritDomain(Enum):
    """Sanskrit knowledge domains for specialized AI assistants."""
    PHILOSOPHY = "philosophy"      # Bhagavad Gita, Upanishads, etc.
    GRAMMAR = "grammar"           # Panini, Ashtadhyayi
    LITERATURE = "literature"     # Kavya, Natya, etc.  
    AYURVEDA = "ayurveda"        # Charak Samhita, Susruta
    MATHEMATICS = "mathematics"   # Lilavati, Bijaganita
    YOGA = "yoga"                # Patanjali Yoga Sutras
    SANKHYA = "sankhya"          # Sankhya Karika
    VEDANTA = "vedanta"          # Brahma Sutras
    DHARMA = "dharma"            # Manu Smriti, Dharma Shastras
    MUSIC = "music"              # Sangita Ratnakara
    ASTRONOMY = "astronomy"       # Surya Siddhanta
    GENERAL = "general"          # Cross-domain queries


@dataclass
class DomainConfig:
    """Configuration for a specific Sanskrit domain."""
    name: str
    display_name: str
    description: str
    expert_name: str  # e.g., "Panini AI", "Charak AI"
    system_prompt: str
    primary_sources: List[str]
    specialized_features: List[str]
    ui_color: str
    icon: str


class DomainExpertPrompts:
    """System prompts for different domain experts."""
    
    PANINI_AI = """You are Panini AI, a Sanskrit grammar expert based on Ashtadhyayi. Your expertise includes:
- Sanskrit grammar rules (sutras) and their applications
- Morphology, phonetics, and syntax analysis
- Word formation (prakriya) explanations
- Sandhi rules and transformations
- Root analysis and dhatu forms

CITATION REQUIREMENTS: Use exact passage IDs in [id] format from provided grammar texts.
TEACHING STYLE: Break down complex grammar rules step-by-step, show derivations clearly."""

    CHARAK_AI = """You are Charak AI, an Ayurveda expert based on Charak Samhita. Your expertise includes:
- Classical Ayurvedic principles and treatments
- Dosha analysis and constitutional medicine
- Medicinal plants and their properties
- Disease diagnosis and therapeutic approaches
- Preventive healthcare and lifestyle guidance

CITATION REQUIREMENTS: Use exact passage IDs in [id] format from provided Ayurvedic texts.
TEACHING STYLE: Explain concepts clearly, relate to modern understanding when helpful."""

    YOGA_AI = """You are Yoga AI, a specialist in Patanjali's Yoga Sutras and yoga philosophy. Your expertise includes:
- Eight limbs of yoga (Ashtanga Yoga)
- Meditation techniques and mental disciplines
- Philosophical foundations of yoga practice  
- Practical guidance for spiritual development
- Integration of yoga with daily life

CITATION REQUIREMENTS: Use exact passage IDs in [id] format from provided yoga texts.
TEACHING STYLE: Practical wisdom combined with philosophical depth."""

    LILAVATI_AI = """You are Lilavati AI, a Sanskrit mathematics expert based on Bhaskaracharya's works. Your expertise includes:
- Classical Indian mathematical concepts
- Arithmetic, algebra, and geometry
- Problem-solving techniques from ancient texts
- Historical context of mathematical developments
- Connections between mathematics and Sanskrit literature

CITATION REQUIREMENTS: Use exact passage IDs in [id] format from provided mathematical texts.
TEACHING STYLE: Show step-by-step solutions, explain underlying principles."""

    GENERAL_AI = """You are a comprehensive Sanskrit scholar with knowledge across multiple domains. Your role is to:
- Help users navigate different Sanskrit knowledge areas
- Provide cross-domain connections and insights
- Direct users to appropriate domain experts when needed
- Offer general Sanskrit language and cultural guidance

CITATION REQUIREMENTS: Use exact passage IDs in [id] format from provided texts.
TEACHING STYLE: Broad perspective with ability to drill down into specifics."""


class MultiDomainManager:
    """Manages multiple Sanskrit knowledge domains and their specialized AI assistants."""
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.domains: Dict[SanskritDomain, DomainConfig] = {}
        self.current_domain = SanskritDomain.GENERAL
        self.domain_data: Dict[SanskritDomain, Dict] = {}
        
        self._initialize_domains()
    
    def _initialize_domains(self):
        """Initialize all supported domains with their configurations."""
        
        self.domains[SanskritDomain.PHILOSOPHY] = DomainConfig(
            name="philosophy",
            display_name="Sanskrit Philosophy", 
            description="Bhagavad Gita, Upanishads, and philosophical texts",
            expert_name="Philosophy AI",
            system_prompt=DomainExpertPrompts.GENERAL_AI,  # Using existing system for now
            primary_sources=["Bhagavad Gita", "Upanishads", "Brahma Sutras"],
            specialized_features=["Verse analysis", "Philosophical discussions", "Commentary comparison"],
            ui_color="#FF6B35",
            icon="🕉️"
        )
        
        self.domains[SanskritDomain.GRAMMAR] = DomainConfig(
            name="grammar",
            display_name="Sanskrit Grammar",
            description="Panini's grammar rules and linguistic analysis", 
            expert_name="Panini AI",
            system_prompt=DomainExpertPrompts.PANINI_AI,
            primary_sources=["Ashtadhyayi", "Mahabhasya", "Kasika Vritti"],
            specialized_features=["Grammar checking", "Sandhi analysis", "Word derivation", "Prakriya display"],
            ui_color="#2E8B57",
            icon="📝"
        )
        
        self.domains[SanskritDomain.AYURVEDA] = DomainConfig(
            name="ayurveda", 
            display_name="Ayurveda",
            description="Traditional Indian medicine and wellness",
            expert_name="Charak AI",
            system_prompt=DomainExpertPrompts.CHARAK_AI,
            primary_sources=["Charak Samhita", "Susruta Samhita", "Ashtanga Hridaya"],
            specialized_features=["Herb identification", "Dosha analysis", "Treatment suggestions"],
            ui_color="#228B22",
            icon="🌿"
        )
        
        self.domains[SanskritDomain.MATHEMATICS] = DomainConfig(
            name="mathematics",
            display_name="Sanskrit Mathematics", 
            description="Ancient Indian mathematical treatises",
            expert_name="Lilavati AI", 
            system_prompt=DomainExpertPrompts.LILAVATI_AI,
            primary_sources=["Lilavati", "Bijaganita", "Surya Siddhanta"],
            specialized_features=["Problem solving", "Calculation methods", "Geometric constructions"],
            ui_color="#4169E1",
            icon="🔢"
        )
        
        self.domains[SanskritDomain.YOGA] = DomainConfig(
            name="yoga",
            display_name="Yoga Philosophy",
            description="Patanjali's Yoga Sutras and practice guidance",
            expert_name="Yoga AI",
            system_prompt=DomainExpertPrompts.YOGA_AI,
            primary_sources=["Yoga Sutras", "Hatha Yoga Pradipika", "Gherand Samhita"],
            specialized_features=["Meditation guidance", "Practice sequences", "Philosophy explanation"],
            ui_color="#9370DB",
            icon="🧘"
        )
        
        self.domains[SanskritDomain.GENERAL] = DomainConfig(
            name="general",
            display_name="General Sanskrit",
            description="Cross-domain Sanskrit knowledge and guidance",
            expert_name="Sanskrit Scholar AI",
            system_prompt=DomainExpertPrompts.GENERAL_AI,
            primary_sources=["Multiple Sources"],
            specialized_features=["Domain routing", "General guidance", "Language help"],
            ui_color="#708090",
            icon="🎓"
        )
    
    def get_domain_config(self, domain: SanskritDomain) -> DomainConfig:
        """Get configuration for a specific domain."""
        return self.domains.get(domain, self.domains[SanskritDomain.GENERAL])
    
    def set_active_domain(self, domain: SanskritDomain):
        """Set the currently active domain."""
        self.current_domain = domain
    
    def get_active_domain(self) -> SanskritDomain:
        """Get the currently active domain."""
        return self.current_domain
    
    def get_system_prompt(self, domain: SanskritDomain = None) -> str:
        """Get system prompt for specified or current domain."""
        if domain is None:
            domain = self.current_domain
        
        domain_config = self.get_domain_config(domain)
        return domain_config.system_prompt
    
    def auto_detect_domain(self, question: str) -> SanskritDomain:
        """
        Auto-detect the most appropriate domain for a question.
        Uses keyword matching and context analysis.
        """
        question_lower = question.lower()
        
        # Grammar keywords
        if any(keyword in question_lower for keyword in [
            'grammar', 'sandhi', 'dhatu', 'pratyaya', 'vibhakti', 
            'panini', 'sutra', 'prakriya', 'rule'
        ]):
            return SanskritDomain.GRAMMAR
        
        # Ayurveda keywords  
        if any(keyword in question_lower for keyword in [
            'ayurveda', 'dosha', 'vata', 'pitta', 'kapha', 'herb', 
            'medicine', 'charak', 'susruta', 'treatment'
        ]):
            return SanskritDomain.AYURVEDA
            
        # Mathematics keywords
        if any(keyword in question_lower for keyword in [
            'mathematics', 'calculate', 'number', 'geometry', 
            'lilavati', 'bhaskara', 'arithmetic', 'algebra'
        ]):
            return SanskritDomain.MATHEMATICS
            
        # Yoga keywords
        if any(keyword in question_lower for keyword in [
            'yoga', 'meditation', 'asana', 'pranayama', 'patanjali',
            'samadhi', 'dharana', 'yama', 'niyama'
        ]):
            return SanskritDomain.YOGA
            
        # Philosophy keywords (default for philosophical content)
        if any(keyword in question_lower for keyword in [
            'dharma', 'karma', 'moksha', 'atman', 'brahman',
            'gita', 'upanishad', 'philosophy', 'spiritual'
        ]):
            return SanskritDomain.PHILOSOPHY
        
        # Default to general for ambiguous queries
        return SanskritDomain.GENERAL
    
    def get_available_domains(self) -> List[Tuple[SanskritDomain, DomainConfig]]:
        """Get list of all available domains with their configurations."""
        return [(domain, config) for domain, config in self.domains.items()]
    
    def load_domain_data(self, domain: SanskritDomain, data_path: str):
        """Load domain-specific data (passages, QA pairs, etc.)."""
        try:
            # This would load domain-specific data files
            # For now, we'll use the existing structure
            self.domain_data[domain] = {
                'data_path': data_path,
                'loaded': True
            }
        except Exception as e:
            print(f"Warning: Could not load data for domain {domain.value}: {e}")
    
    def get_domain_features(self, domain: SanskritDomain) -> List[str]:
        """Get specialized features available for a domain."""
        config = self.get_domain_config(domain)
        return config.specialized_features
    
    def format_domain_response(self, response: str, domain: SanskritDomain) -> str:
        """Format response with domain-specific styling and information."""
        config = self.get_domain_config(domain)
        
        header = f"{config.icon} **{config.expert_name}** responds:\n\n"
        footer = f"\n\n*Specialized in: {config.description}*"
        
        return header + response + footer


# Example usage and testing
if __name__ == "__main__":
    manager = MultiDomainManager("user_assets/config.yaml")
    
    # Test domain detection
    test_questions = [
        "What is the sandhi rule for vowel combination?",
        "How do I treat a vata imbalance?", 
        "Solve this arithmetic problem from Lilavati",
        "Explain the eight limbs of yoga",
        "What does Krishna teach about dharma?"
    ]
    
    for question in test_questions:
        detected = manager.auto_detect_domain(question)
        config = manager.get_domain_config(detected)
        print(f"Q: {question}")
        print(f"Domain: {config.display_name} ({config.expert_name})")
        print()
