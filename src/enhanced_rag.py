#!/usr/bin/env python3
"""
Enhanced Sanskrit RAG System with strict citation safety.
Implements post-generation validation and re-prompting for citation accuracy.
"""

import time
from typing import Dict, List, Any, Optional, Tuple, Set
from pathlib import Path

try:
    from .rag import SanskritRAG, RAGResponse, RetrievalResult
    from .rag_postcheck import RAGPostChecker, StrictRAGPromptTemplate, CitationValidationResult
except ImportError:
    from rag import SanskritRAG, RAGResponse, RetrievalResult
    from rag_postcheck import RAGPostChecker, StrictRAGPromptTemplate, CitationValidationResult


class EnhancedSanskritRAG(SanskritRAG):
    """
    Enhanced Sanskrit RAG with strict citation safety.
    Extends the base RAG system with post-generation validation and re-prompting.
    """
    
    def __init__(self, config_path: str):
        super().__init__(config_path)
        
        # Enhanced components
        self.post_checker = RAGPostChecker()
        self.strict_prompt_template = StrictRAGPromptTemplate()
        
        # Re-prompt settings
        self.max_reprompt_attempts = 2
        self.enable_citation_safety = True
        
    def generate_response_with_validation(self, query: str, 
                                        retrieved_passages: List[RetrievalResult],
                                        attempt: int = 1) -> Tuple[str, CitationValidationResult]:
        """
        Generate response with citation validation and re-prompting.
        
        Args:
            query: User query
            retrieved_passages: Retrieved passages
            attempt: Current attempt number (for re-prompting)
            
        Returns:
            Tuple of (response_text, validation_result)
        """
        if not retrieved_passages:
            fallback_response = self.post_checker.create_fallback_response(query, [])
            # Create a valid validation result for fallback
            validation_result = CitationValidationResult(
                is_valid=True,
                valid_citations=[],
                invalid_citations=[],
                missing_citations=[],
                total_claims=0,
                citation_coverage=0.0,
                suggestion="No passages available"
            )
            return fallback_response, validation_result
        
        # Convert retrieved passages to dict format for post-checker
        passages_dict = []
        retrieved_ids = set()
        for result in retrieved_passages:
            passage = result.passage.copy()
            passages_dict.append(passage)
            retrieved_ids.add(passage['id'])
        
        # Use strict prompt template for better citation compliance
        if attempt == 1:
            # First attempt: use strict template
            prompt = self.strict_prompt_template.create_prompt(query, passages_dict)
        else:
            # Re-prompt: create correction prompt
            # We need the previous response for this, but we don't have it in this method
            # So we'll use the strict template again but with more emphasis
            prompt = self.strict_prompt_template.create_prompt(query, passages_dict)
            prompt += f"\n\nIMPORTANT: This is attempt {attempt}. Previous attempts had citation issues. Please be extra careful to cite EVERY factual claim with the exact passage IDs provided above."
        
        # Generate response
        try:
            response = self.llm_manager.generate(
                prompt=prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature * 0.8,  # Reduce temperature for more focused responses
                stop_sequences=["Question:", "Context:", "RETRIEVED_PASSAGES:"]
            )
            
            response = response.strip()
            
            # Validate citations
            validation_result = self.post_checker.validate_citations(response, retrieved_ids)
            
            return response, validation_result
            
        except Exception as e:
            print(f"WARNING: Generation failed on attempt {attempt}: {str(e)}")
            fallback_response = self.post_checker.create_fallback_response(query, passages_dict)
            validation_result = CitationValidationResult(
                is_valid=True,
                valid_citations=[],
                invalid_citations=[],
                missing_citations=[],
                total_claims=0,
                citation_coverage=0.0,
                suggestion="Generation failed, using fallback"
            )
            return fallback_response, validation_result
    
    def answer_question_with_safety(self, question: str, 
                                   retrieval_k: Optional[int] = None) -> Tuple[RAGResponse, CitationValidationResult]:
        """
        Complete RAG pipeline with citation safety validation and re-prompting.
        
        Args:
            question: User question
            retrieval_k: Number of passages to retrieve (optional)
            
        Returns:
            Tuple of (RAG response, final validation result)
        """
        start_time = time.time()
        
        try:
            # Retrieve relevant passages
            retrieved_passages = self.retrieve_passages(question, retrieval_k)
            
            if not self.enable_citation_safety:
                # Use standard generation if safety is disabled
                answer = self.generate_response(question, retrieved_passages)
                citations = self.citation_validator.extract_citations(answer)
                processing_time = time.time() - start_time
                
                response = RAGResponse(
                    query=question,
                    answer=answer,
                    retrieved_passages=retrieved_passages,
                    citations=citations,
                    model_info=self.llm_manager.get_current_backend_info(),
                    processing_time=processing_time
                )
                
                # Create a basic validation result
                validation_result = CitationValidationResult(
                    is_valid=True,
                    valid_citations=citations,
                    invalid_citations=[],
                    missing_citations=[],
                    total_claims=1,
                    citation_coverage=1.0 if citations else 0.0,
                    suggestion="Citation safety disabled"
                )
                
                return response, validation_result
            
            # Multi-attempt generation with validation
            final_response = None
            final_validation = None
            passages_dict = [result.passage for result in retrieved_passages]
            
            for attempt in range(1, self.max_reprompt_attempts + 2):  # +1 for initial attempt
                print(f"Generation attempt {attempt}...")
                
                response_text, validation_result = self.generate_response_with_validation(
                    question, retrieved_passages, attempt
                )
                
                if validation_result.is_valid:
                    print(f"✓ Valid response generated on attempt {attempt}")
                    final_response = response_text
                    final_validation = validation_result
                    break
                else:
                    print(f"✗ Attempt {attempt} failed validation: {validation_result.suggestion}")
                    
                    if attempt <= self.max_reprompt_attempts:
                        # Create re-prompt for next attempt
                        print(f"Preparing re-prompt for attempt {attempt + 1}...")
                        # Here we could create a more sophisticated re-prompt
                        # For now, the next iteration will use enhanced prompt
                        continue
                    else:
                        # Final attempt failed, use fallback
                        print("All attempts failed, using safe fallback response")
                        final_response = self.post_checker.create_fallback_response(question, passages_dict)
                        final_validation = CitationValidationResult(
                            is_valid=True,
                            valid_citations=[],
                            invalid_citations=[],
                            missing_citations=[],
                            total_claims=0,
                            citation_coverage=0.0,
                            suggestion="Used safe fallback after validation failures"
                        )
                        break
            
            # Extract citations from final response
            citations = self.post_checker.extract_citations(final_response)
            
            processing_time = time.time() - start_time
            
            response = RAGResponse(
                query=question,
                answer=final_response,
                retrieved_passages=retrieved_passages,
                citations=citations,
                model_info=self.llm_manager.get_current_backend_info(),
                processing_time=processing_time
            )
            
            return response, final_validation
            
        except Exception as e:
            # Error response
            processing_time = time.time() - start_time
            
            error_response = RAGResponse(
                query=question,
                answer=f"ERROR: Failed to process question: {str(e)}",
                retrieved_passages=[],
                citations=[],
                model_info=self.llm_manager.get_current_backend_info() if self.llm_manager else {},
                processing_time=processing_time
            )
            
            error_validation = CitationValidationResult(
                is_valid=False,
                valid_citations=[],
                invalid_citations=[],
                missing_citations=[],
                total_claims=0,
                citation_coverage=0.0,
                suggestion=f"System error: {str(e)}"
            )
            
            return error_response, error_validation
    
    def configure_citation_safety(self, enabled: bool = True, max_attempts: int = 2):
        """Configure citation safety parameters."""
        self.enable_citation_safety = enabled
        self.max_reprompt_attempts = max_attempts
        print(f"Citation safety: {'enabled' if enabled else 'disabled'}")
        if enabled:
            print(f"Max re-prompt attempts: {max_attempts}")
    
    def validate_existing_response(self, response_text: str, 
                                 retrieved_passages: List[RetrievalResult]) -> CitationValidationResult:
        """Validate an existing response for citation accuracy."""
        retrieved_ids = {result.passage['id'] for result in retrieved_passages}
        return self.post_checker.validate_citations(response_text, retrieved_ids)
    
    def get_citation_statistics(self) -> Dict[str, Any]:
        """Get statistics about citation usage in the current session."""
        # This could be enhanced to track citations across multiple queries
        return {
            "safety_enabled": self.enable_citation_safety,
            "max_attempts": self.max_reprompt_attempts,
            "available_passages": self.index.ntotal if self.index else 0,
            "post_checker_enabled": True
        }


def main():
    """Test the enhanced RAG system."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Test Enhanced Sanskrit RAG system with citation safety"
    )
    parser.add_argument(
        "--config",
        default="user_assets/config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--question",
        default="What does dharma mean?",
        help="Question to ask the system"
    )
    parser.add_argument(
        "--retrieval-k",
        type=int,
        default=3,
        help="Number of passages to retrieve"
    )
    parser.add_argument(
        "--disable-safety",
        action="store_true",
        help="Disable citation safety validation"
    )
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=2,
        help="Maximum re-prompt attempts"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize enhanced RAG system
        print("Initializing Enhanced Sanskrit RAG with citation safety...")
        rag = EnhancedSanskritRAG(args.config)
        
        if not rag.initialize():
            print("Failed to initialize RAG system.")
            exit(1)
        
        # Configure citation safety
        rag.configure_citation_safety(
            enabled=not args.disable_safety,
            max_attempts=args.max_attempts
        )
        
        print(f"\nTesting question: {args.question}")
        print("=" * 60)
        
        # Generate response with safety validation
        response, validation = rag.answer_question_with_safety(
            args.question, 
            args.retrieval_k
        )
        
        # Display results
        print(f"\nFinal Answer:")
        print("-" * 50)
        print(response.answer)
        print("-" * 50)
        
        print(f"\nCitation Validation:")
        print(f"  Valid: {validation.is_valid}")
        print(f"  Valid citations: {validation.valid_citations}")
        print(f"  Invalid citations: {validation.invalid_citations}")
        print(f"  Coverage: {validation.citation_coverage:.1%}")
        print(f"  Suggestion: {validation.suggestion}")
        
        print(f"\nRetrieved Passages:")
        for result in response.retrieved_passages:
            passage = result.passage
            print(f"  [{passage['id']}] {passage['work']} {passage['chapter']}.{passage['verse']} (Score: {result.score:.3f})")
        
        print(f"\nProcessing time: {response.processing_time:.2f}s")
        
        # Show citation statistics
        stats = rag.get_citation_statistics()
        print(f"\nSystem Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
