#!/usr/bin/env python3
"""
RAG Citation Post-Checker - Ensures strict citation accuracy for Sanskrit Tutor.
Validates that all citations in generated responses refer to actual retrieved passages.
"""

import re
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass


@dataclass
class CitationValidationResult:
    """Result of citation validation."""
    is_valid: bool
    valid_citations: List[str]
    invalid_citations: List[str]
    missing_citations: List[str]
    total_claims: int
    citation_coverage: float
    suggestion: str


class RAGPostChecker:
    """
    Post-generation citation validator with re-prompt failover.
    Implements the exact specification requirements for citation safety.
    """
    
    # Regex pattern to extract citations [id] or [id1, id2]
    CITATION_RE = re.compile(r'\[([^\]]+)\]')
    
    # Patterns that typically require citations
    FACTUAL_CLAIM_PATTERNS = [
        r'According to',
        r'The text states',
        r'As mentioned in',
        r'The verse says',
        r'This means',
        r'The Sanskrit word \w+ means',
        r'In the Bhagavad Gita',
        r'The Upanishads teach',
        r'Dharma is defined as',
        r'The root \w+ means',
    ]
    
    def __init__(self):
        self.factual_pattern = re.compile('|'.join(self.FACTUAL_CLAIM_PATTERNS), re.IGNORECASE)
    
    def extract_citations(self, text: str) -> List[str]:
        """Extract all citation IDs from response text."""
        citations = []
        matches = self.CITATION_RE.findall(text)
        
        for match in matches:
            # Split by comma and clean up
            cite_ids = [cite.strip() for cite in match.split(',')]
            citations.extend(cite_ids)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_citations = []
        for cite in citations:
            if cite not in seen:
                seen.add(cite)
                unique_citations.append(cite)
                
        return unique_citations
    
    def count_factual_claims(self, text: str) -> int:
        """Count sentences that appear to make factual claims."""
        sentences = re.split(r'[.!?]+', text)
        factual_count = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 10:  # Skip very short fragments
                continue
            
            # Check if sentence contains factual claim indicators
            if self.factual_pattern.search(sentence):
                factual_count += 1
            elif any(word in sentence.lower() for word in ['means', 'is defined', 'teaches', 'states']):
                factual_count += 1
        
        return max(factual_count, 1)  # At least 1 to avoid division by zero
    
    def validate_citations(self, response_text: str, 
                         retrieved_passage_ids: Set[str]) -> CitationValidationResult:
        """
        Validate citations in response against retrieved passages.
        
        Args:
            response_text: Generated response text
            retrieved_passage_ids: Set of valid passage IDs from retrieval
            
        Returns:
            CitationValidationResult with validation details
        """
        citations = self.extract_citations(response_text)
        
        # Classify citations
        valid_citations = []
        invalid_citations = []
        
        for cite_id in citations:
            if cite_id in retrieved_passage_ids:
                valid_citations.append(cite_id)
            else:
                invalid_citations.append(cite_id)
        
        # Count factual claims
        total_claims = self.count_factual_claims(response_text)
        
        # Calculate citation coverage
        citation_coverage = len(valid_citations) / total_claims if total_claims > 0 else 0
        
        # Determine if response is valid
        is_valid = (
            len(invalid_citations) == 0 and  # No invalid citations
            citation_coverage >= 0.3  # At least 30% of claims are cited
        )
        
        # Generate suggestion
        if invalid_citations:
            suggestion = f"Remove invalid citations: {invalid_citations}"
        elif citation_coverage < 0.3:
            suggestion = f"Add citations for factual claims (coverage: {citation_coverage:.1%})"
        else:
            suggestion = "Citations are valid"
        
        return CitationValidationResult(
            is_valid=is_valid,
            valid_citations=valid_citations,
            invalid_citations=invalid_citations,
            missing_citations=[],  # Could be computed if needed
            total_claims=total_claims,
            citation_coverage=citation_coverage,
            suggestion=suggestion
        )
    
    def create_re_prompt(self, original_query: str, 
                        response_text: str,
                        validation_result: CitationValidationResult,
                        retrieved_passages: List[Dict]) -> str:
        """
        Create a re-prompt to fix citation issues.
        
        Args:
            original_query: Original user query
            response_text: Generated response with citation issues
            validation_result: Validation result details
            retrieved_passages: Available passages for citation
            
        Returns:
            Re-prompt text for fixing citations
        """
        available_ids = [p.get('id', p.get('passage_id', '')) for p in retrieved_passages]
        
        re_prompt = f"""CITATION CORRECTION REQUIRED

Original Query: {original_query}

Previous Response Had Issues:
{validation_result.suggestion}

Available Passage IDs for Citation: {available_ids}

Please rewrite the response following these STRICT rules:
1. Use ONLY these passage IDs: {available_ids}
2. Add [passage_id] after EVERY factual claim
3. If you cannot cite a claim from available passages, say "I am unsure"
4. Format: "Statement about X [passage_id]. Another fact [passage_id2]."

Retrieved Passages:
"""
        
        # Add passage context
        for passage in retrieved_passages:
            passage_id = passage.get('id', passage.get('passage_id', ''))
            text_iast = passage.get('text_iast', '')
            work_info = f"{passage.get('work', '')} {passage.get('chapter', '')}.{passage.get('verse', '')}"
            
            re_prompt += f"\n[{passage_id}] {work_info}\n{text_iast}\n"
        
        re_prompt += f"\nNow rewrite your response to: {original_query}"
        
        return re_prompt
    
    def create_fallback_response(self, original_query: str, 
                                retrieved_passages: List[Dict]) -> str:
        """
        Create a safe fallback response when citation validation fails repeatedly.
        
        Args:
            original_query: Original user query
            retrieved_passages: Retrieved passages
            
        Returns:
            Safe response with proper citations
        """
        if not retrieved_passages:
            return "I am unsure. I don't have enough relevant passages to answer this question accurately."
        
        # Create a basic response with citations
        available_ids = [p.get('id', p.get('passage_id', '')) for p in retrieved_passages[:3]]
        
        response = f"I am unsure about the complete answer to your question, but I found these relevant passages:\n\n"
        
        for passage in retrieved_passages[:3]:
            passage_id = passage.get('id', passage.get('passage_id', ''))
            text_iast = passage.get('text_iast', '')
            work_info = f"{passage.get('work', '')} {passage.get('chapter', '')}.{passage.get('verse', '')}"
            
            response += f"• {work_info}: {text_iast} [{passage_id}]\n"
        
        response += "\nPlease ask a more specific question or provide more context for a detailed answer."
        
        return response


class StrictRAGPromptTemplate:
    """
    Strict RAG prompt template that enforces citation requirements.
    Implements the exact system instruction from the specification.
    """
    
    SYSTEM_INSTRUCTION = """You are SanskritTutor. Use ONLY the passages listed in RETRIEVED_PASSAGES to answer factual claims.

CRITICAL CITATION REQUIREMENTS:
1) Give a SHORT ANSWER (1-3 sentences) in the requested language (Sanskrit/English).
2) Provide a short EXPLANATION (1-3 sentences).
3) Append CITATIONS: list the exact passage ids used in square brackets, and include a 1-line excerpt (IAST).
4) Use ONLY passage IDs provided below - never invent citations.
5) Place [passage_id] immediately after each factual claim.
6) If you cannot answer purely from the retrieved passages, say "I am unsure" and list candidate passages.

EXAMPLES:
✓ GOOD: "Dharma means righteous duty [bg_002_047]. The Gita teaches karma yoga [bg_002_048]."
✗ BAD: "Dharma means duty. This is taught in Hindu philosophy."
✗ BAD: "Dharma means duty [fake_citation]."

If you cannot find supporting passages for any part of your answer, respond with "I am unsure" followed by the relevant passages you found."""

    def format_context(self, retrieved_passages: List[Dict]) -> str:
        """Format retrieved passages for the prompt context."""
        if not retrieved_passages:
            return "No passages retrieved."
        
        context_parts = []
        
        for i, passage in enumerate(retrieved_passages, 1):
            passage_id = passage.get('id', passage.get('passage_id', f'passage_{i}'))
            text_iast = passage.get('text_iast', '')
            text_devanagari = passage.get('text_devanagari', '')
            work = passage.get('work', '')
            chapter = passage.get('chapter', '')
            verse = passage.get('verse', '')
            source_url = passage.get('source_url', '')
            
            context_part = f"""{i}. [{passage_id}] {work} {chapter}.{verse}
IAST: {text_iast}
Devanagari: {text_devanagari}
Source: {source_url}"""
            
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)
    
    def create_prompt(self, query: str, retrieved_passages: List[Dict]) -> str:
        """Create complete RAG prompt with strict citation requirements."""
        context = self.format_context(retrieved_passages)
        
        prompt = f"""{self.SYSTEM_INSTRUCTION}

RETRIEVED_PASSAGES:
{context}

USER QUERY: {query}

INSTRUCTIONS:
- Respond using ONLY the RETRIEVED_PASSAGES above
- Add [passage_id] after every factual claim
- If unsure, say "I am unsure" + list candidate passages
- Use exact passage IDs shown above: never make up citations

Response:"""
        
        return prompt


def test_citation_validator():
    """Test the citation validation system."""
    checker = RAGPostChecker()
    
    # Test cases
    test_cases = [
        {
            "response": "Dharma means righteous duty [bg_002_047]. The Gita teaches karma yoga [bg_002_048].",
            "retrieved_ids": {"bg_002_047", "bg_002_048", "bg_001_001"},
            "expected_valid": True
        },
        {
            "response": "Dharma means duty [fake_citation]. This is important.",
            "retrieved_ids": {"bg_002_047", "bg_002_048"},
            "expected_valid": False
        },
        {
            "response": "Dharma is complex. It means many things. This is philosophical.",
            "retrieved_ids": {"bg_002_047"},
            "expected_valid": False  # No citations for factual claims
        }
    ]
    
    print("🔍 Testing Citation Validator:")
    print("=" * 50)
    
    for i, case in enumerate(test_cases, 1):
        result = checker.validate_citations(case["response"], case["retrieved_ids"])
        
        print(f"\nTest {i}:")
        print(f"Response: {case['response'][:60]}...")
        print(f"Valid: {result.is_valid} (expected: {case['expected_valid']})")
        print(f"Citations: {result.valid_citations}")
        print(f"Invalid: {result.invalid_citations}")
        print(f"Coverage: {result.citation_coverage:.1%}")
        print(f"Suggestion: {result.suggestion}")
        
        status = "✓ PASS" if result.is_valid == case["expected_valid"] else "✗ FAIL"
        print(f"Status: {status}")


if __name__ == "__main__":
    test_citation_validator()
