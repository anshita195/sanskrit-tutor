#!/usr/bin/env python3
"""
Retrieval-Augmented Generation (RAG) system for Sanskrit Tutor.
Retrieves relevant passages and formats responses with exact ID citations.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

try:
    from .embed_index import EmbeddingIndexer
    from .llm_backends import LLMManager, create_llm_manager
    from .ingest import DataIngester, Passage, QAPair
    from .domain_manager import MultiDomainManager, SanskritDomain
    from .rag_postcheck import RAGPostChecker, StrictRAGPromptTemplate
except ImportError:
    from embed_index import EmbeddingIndexer
    from llm_backends import LLMManager, create_llm_manager
    from ingest import DataIngester, Passage, QAPair
    from domain_manager import MultiDomainManager, SanskritDomain
    from rag_postcheck import RAGPostChecker, StrictRAGPromptTemplate


@dataclass
class RetrievalResult:
    """Result from passage retrieval."""
    passage_id: str
    passage: Dict[str, Any]
    score: float
    relevance_rank: int
    
    
@dataclass
class RAGResponse:
    """Complete RAG response with context and citations."""
    query: str
    answer: str
    retrieved_passages: List[RetrievalResult]
    citations: List[str]
    model_info: Dict[str, str]
    processing_time: float


class SanskritPromptTemplate:
    """
    Prompt template for Sanskrit tutoring with proper citation format.
    Uses exact passage IDs for citations as required by the specification.
    """
    
    SYSTEM_INSTRUCTION = """You are a Sanskrit tutor and scholar. Your role is to help students learn Sanskrit language, literature, and philosophy by providing accurate, educational responses.

CRITICAL CITATION REQUIREMENTS:
- You MUST cite sources using the exact passage IDs provided in square brackets: [passage_id]
- Every factual claim or translation must include a citation
- Use multiple citations when drawing from multiple sources: [id1, id2]
- Place citations at the end of each claim, not at the end of the response
- Only use passage IDs that are explicitly provided in the context below

RESPONSE STYLE:
- Be pedagogical and educational
- Explain Sanskrit grammar, etymology, and cultural context when relevant
- Break down complex concepts into digestible parts
- Use both Devanagari and IAST transliteration when helpful
- Provide pronunciation guidance when appropriate

ACCURACY:
- Only make claims supported by the provided passages
- If unsure, acknowledge uncertainty
- Distinguish between different philosophical schools or textual traditions
- Note variant readings or interpretations when relevant"""

    USER_TEMPLATE = """Context passages:
{context}

Question: {question}

Please provide a comprehensive answer with proper citations using the passage IDs shown above."""

    def format_context(self, retrieved_passages: List[RetrievalResult]) -> str:
        """Format retrieved passages for the prompt context."""
        context_parts = []
        
        for result in retrieved_passages:
            passage = result.passage
            context_part = f"""[{passage['id']}] {passage['work']} {passage['chapter']}.{passage['verse']}
Devanagari: {passage['text_devanagari']}
IAST: {passage['text_iast']}
Source: {passage['source_url']}"""
            
            if passage.get('notes'):
                context_part += f"\nNotes: {passage['notes']}"
                
            context_parts.append(context_part)
        
        return "\n\n".join(context_parts)
    
    def create_prompt(self, question: str, retrieved_passages: List[RetrievalResult]) -> str:
        """Create complete prompt with system instruction, context, and question."""
        context = self.format_context(retrieved_passages)
        
        user_prompt = self.USER_TEMPLATE.format(
            context=context,
            question=question
        )
        
        # Combine system instruction and user prompt
        full_prompt = f"{self.SYSTEM_INSTRUCTION}\n\n{user_prompt}"
        return full_prompt


class CitationValidator:
    """Validates and extracts citations from generated responses."""
    
    def __init__(self, valid_passage_ids: set):
        self.valid_passage_ids = valid_passage_ids
    
    def extract_citations(self, text: str) -> List[str]:
        """
        Extract citation IDs from response text.
        Citations should be in format [passage_id] or [id1, id2].
        """
        citations = []
        
        # Pattern to match [citation] format
        pattern = r'\[([^\]]+)\]'
        matches = re.findall(pattern, text)
        
        for match in matches:
            # Split by comma and clean up
            cite_ids = [cite.strip() for cite in match.split(',')]
            citations.extend(cite_ids)
            
        return list(set(citations))  # Remove duplicates
    
    def validate_citations(self, citations: List[str]) -> Tuple[List[str], List[str]]:
        """
        Validate citations against known passage IDs.
        
        Returns:
            Tuple of (valid_citations, invalid_citations)
        """
        valid = []
        invalid = []
        
        for cite_id in citations:
            if cite_id in self.valid_passage_ids:
                valid.append(cite_id)
            else:
                invalid.append(cite_id)
                
        return valid, invalid
    
    def validate_response(self, response_text: str) -> Dict[str, Any]:
        """
        Validate a complete response for proper citations.
        
        Returns:
            Dictionary with validation results
        """
        citations = self.extract_citations(response_text)
        valid_citations, invalid_citations = self.validate_citations(citations)
        
        return {
            'total_citations': len(citations),
            'valid_citations': valid_citations,
            'invalid_citations': invalid_citations,
            'citation_rate': len(valid_citations) / max(1, len(citations.split('. '))) if citations else 0
        }


class SanskritRAG:
    """
    Main RAG system for Sanskrit tutoring.
    Combines retrieval, generation, and citation validation with multi-domain support.
    """
    
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.base_path = Path(".")
        
        # Components
        self.indexer = None
        self.llm_manager = None
        self.index = None
        self.metadata = None
        self.prompt_template = SanskritPromptTemplate()
        self.citation_validator = None
        
        # Multi-domain support
        self.domain_manager = MultiDomainManager(config_path)
        self.current_domain = SanskritDomain.GENERAL
        
        # Configuration
        self.config = None
        self.retrieval_k = 5
        self.max_tokens = 500
        self.temperature = 0.7
        
    def initialize(self) -> bool:
        """
        Initialize all RAG components.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            print("Initializing Sanskrit RAG system...")
            
            # Load configuration
            import yaml
            with open(self.config_path, 'r', encoding='utf-8') as f:
                self.config = yaml.safe_load(f)
            
            # Initialize LLM manager
            print("Loading LLM backend...")
            self.llm_manager = create_llm_manager(self.config_path)
            if not self.llm_manager:
                print("ERROR: Failed to initialize LLM backend")
                return False
            
            # Initialize embedding indexer
            print("Loading embedding model and index...")
            embeddings_model = self.config.get('embeddings_model', 'sentence-transformers/all-mpnet-base-v2')
            self.indexer = EmbeddingIndexer(embeddings_model, self.base_path)
            
            if not self.indexer.load_embedding_model():
                print("ERROR: Failed to load embedding model")
                return False
            
            # Load FAISS index
            index_path = self.config['faiss_index_path']
            self.index, self.metadata = self.indexer.load_faiss_index(index_path)
            
            # Initialize citation validator
            passage_ids = set(self.metadata['passage_ids'])
            self.citation_validator = CitationValidator(passage_ids)
            
            # Set parameters from config
            self.retrieval_k = self.config.get('retrieval_k', 5)
            self.max_tokens = self.config.get('max_tokens', 500)
            self.temperature = self.config.get('temperature', 0.7)
            
            print("Sanskrit RAG system initialized successfully!")
            print(f"- Index contains {self.index.ntotal} passages")
            print(f"- LLM backend: {self.llm_manager.get_current_backend_info()['backend']}")
            print(f"- Retrieval top-k: {self.retrieval_k}")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to initialize RAG system: {str(e)}")
            return False
    
    def retrieve_passages(self, query: str, k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Retrieve relevant passages for a query.
        
        Args:
            query: User query
            k: Number of passages to retrieve (uses config default if None)
            
        Returns:
            List of retrieval results sorted by relevance
        """
        k = k or self.retrieval_k
        
        try:
            # Search the index
            search_results = self.indexer.search(self.index, self.metadata, query, k)
            
            # Convert to RetrievalResult objects
            results = []
            for i, result in enumerate(search_results):
                results.append(RetrievalResult(
                    passage_id=result['passage_id'],
                    passage=result['passage'],
                    score=result['score'],
                    relevance_rank=i + 1
                ))
            
            return results
            
        except Exception as e:
            print(f"WARNING: Retrieval failed: {str(e)}")
            return []
    
    def generate_response(self, query: str, retrieved_passages: List[RetrievalResult]) -> str:
        """
        Generate response using retrieved passages and LLM.
        
        Args:
            query: User query
            retrieved_passages: Retrieved context passages
            
        Returns:
            Generated response text
        """
        if not retrieved_passages:
            # Fallback response when no passages retrieved
            return "I don't have enough context to answer this question accurately. Please try rephrasing your question or check if the relevant passages are available."
        
        # Create prompt with context
        prompt = self.prompt_template.create_prompt(query, retrieved_passages)
        
        # Generate response
        try:
            response = self.llm_manager.generate(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                stop_sequences=["Question:", "Context:"]
            )
            
            return response.strip()
            
        except Exception as e:
            print(f"WARNING: Generation failed: {str(e)}")
            return f"I encountered an error while generating the response: {str(e)}"
    
    def answer_question(self, question: str, retrieval_k: Optional[int] = None) -> RAGResponse:
        """
        Complete RAG pipeline: retrieve, generate, and validate.
        
        Args:
            question: User question
            retrieval_k: Number of passages to retrieve (optional)
            
        Returns:
            Complete RAG response with metadata
        """
        import time
        start_time = time.time()
        
        try:
            # Retrieve relevant passages
            retrieved_passages = self.retrieve_passages(question, retrieval_k)
            
            # Generate response
            answer = self.generate_response(question, retrieved_passages)
            
            # Validate citations
            citations = self.citation_validator.extract_citations(answer)
            
            # Create response object
            processing_time = time.time() - start_time
            
            response = RAGResponse(
                query=question,
                answer=answer,
                retrieved_passages=retrieved_passages,
                citations=citations,
                model_info=self.llm_manager.get_current_backend_info(),
                processing_time=processing_time
            )
            
            return response
            
        except Exception as e:
            # Error response
            processing_time = time.time() - start_time
            
            return RAGResponse(
                query=question,
                answer=f"ERROR: Failed to process question: {str(e)}",
                retrieved_passages=[],
                citations=[],
                model_info=self.llm_manager.get_current_backend_info() if self.llm_manager else {},
                processing_time=processing_time
            )
    
    def get_passage_by_id(self, passage_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific passage by its ID.
        
        Args:
            passage_id: Passage ID to retrieve
            
        Returns:
            Passage dictionary or None if not found
        """
        if not self.metadata:
            return None
            
        try:
            passage_ids = self.metadata['passage_ids']
            if passage_id in passage_ids:
                idx = passage_ids.index(passage_id)
                return self.metadata['passages'][idx]
        except Exception:
            pass
            
        return None
    
    def validate_response_quality(self, response: RAGResponse) -> Dict[str, Any]:
        """
        Validate the quality of a RAG response.
        
        Args:
            response: RAG response to validate
            
        Returns:
            Validation metrics and feedback
        """
        citation_validation = self.citation_validator.validate_response(response.answer)
        
        quality_metrics = {
            'retrieval_success': len(response.retrieved_passages) > 0,
            'has_citations': len(response.citations) > 0,
            'citation_accuracy': len(citation_validation['valid_citations']) / max(1, len(response.citations)),
            'response_length': len(response.answer.split()),
            'processing_time': response.processing_time,
            'citation_details': citation_validation
        }
        
        return quality_metrics
    
    # ========================================
    # Multi-Domain Methods
    # ========================================
    
    def set_domain(self, domain: SanskritDomain):
        """Set the current domain for specialized responses."""
        self.current_domain = domain
        self.domain_manager.set_active_domain(domain)
    
    def auto_detect_domain(self, question: str) -> SanskritDomain:
        """Auto-detect the appropriate domain for a question."""
        return self.domain_manager.auto_detect_domain(question)
    
    def answer_with_domain_detection(self, question: str, retrieval_k: Optional[int] = None) -> Tuple[RAGResponse, SanskritDomain]:
        """Answer question with automatic domain detection."""
        detected_domain = self.auto_detect_domain(question)
        self.set_domain(detected_domain)
        
        # Use domain-specific system prompt
        original_system_prompt = self.prompt_template.SYSTEM_INSTRUCTION
        domain_prompt = self.domain_manager.get_system_prompt(detected_domain)
        self.prompt_template.SYSTEM_INSTRUCTION = domain_prompt
        
        try:
            response = self.answer_question(question, retrieval_k)
            
            # Format response with domain styling
            formatted_answer = self.domain_manager.format_domain_response(
                response.answer, detected_domain
            )
            response.answer = formatted_answer
            
            return response, detected_domain
        finally:
            # Restore original prompt
            self.prompt_template.SYSTEM_INSTRUCTION = original_system_prompt
    
    def get_available_domains(self) -> List[Tuple[SanskritDomain, str]]:
        """Get list of available domains with their display names."""
        domains = self.domain_manager.get_available_domains()
        return [(domain, config.display_name) for domain, config in domains]
    
    def get_domain_config(self, domain: SanskritDomain) -> dict:
        """Get configuration for a specific domain."""
        config = self.domain_manager.get_domain_config(domain)
        return {
            'name': config.name,
            'display_name': config.display_name,
            'description': config.description,
            'expert_name': config.expert_name,
            'features': config.specialized_features,
            'icon': config.icon,
            'color': config.ui_color
        }


def main():
    """Command-line interface for testing the RAG system."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test Sanskrit RAG system"
    )
    parser.add_argument(
        "--config",
        default="user_assets/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--question",
        default="What does 'dharma' mean in Sanskrit philosophy?",
        help="Question to ask the system"
    )
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=3,
        help="Number of passages to retrieve"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize RAG system
        rag = SanskritRAG(args.config)
        
        if not rag.initialize():
            print("Failed to initialize RAG system.")
            exit(1)
        
        if args.interactive:
            # Interactive mode
            print("Sanskrit RAG Interactive Mode")
            print("Type 'quit' to exit\n")
            
            while True:
                question = input("Question: ").strip()
                if question.lower() in ['quit', 'exit', 'q']:
                    break
                    
                if not question:
                    continue
                
                print("\nProcessing...")
                response = rag.answer_question(question, args.retrieval_k)
                
                print(f"\nAnswer:")
                print("-" * 50)
                print(response.answer)
                print("-" * 50)
                
                # Show retrieved passages
                if response.retrieved_passages:
                    print(f"\nRetrieved passages ({len(response.retrieved_passages)}):")
                    for result in response.retrieved_passages:
                        passage = result.passage
                        print(f"  [{passage['id']}] {passage['work']} - Score: {result.score:.3f}")
                
                # Show citations
                if response.citations:
                    print(f"\nCitations: {response.citations}")
                
                print(f"Processing time: {response.processing_time:.2f}s")
                print("="*60)
        
        else:
            # Single question mode
            print(f"Question: {args.question}")
            print()
            
            response = rag.answer_question(args.question, args.retrieval_k)
            
            print("Answer:")
            print("-" * 50)
            print(response.answer)
            print("-" * 50)
            
            # Show details
            print(f"\nRetrieved {len(response.retrieved_passages)} passages:")
            for result in response.retrieved_passages:
                passage = result.passage
                print(f"  [{passage['id']}] {passage['work']} {passage['chapter']}.{passage['verse']} (Score: {result.score:.3f})")
            
            if response.citations:
                print(f"\nCitations found: {response.citations}")
            
            # Show quality metrics
            quality = rag.validate_response_quality(response)
            print(f"\nQuality metrics:")
            print(f"  Citation accuracy: {quality['citation_accuracy']:.2f}")
            print(f"  Response length: {quality['response_length']} words")
            print(f"  Processing time: {response.processing_time:.2f}s")
    
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"ERROR: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
