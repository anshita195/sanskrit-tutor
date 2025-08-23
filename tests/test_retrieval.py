#!/usr/bin/env python3
"""
Test suite for Sanskrit Tutor RAG retrieval and citation behavior.
Tests both fixtures and user asset modes.
"""

import os
import sys
import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ingest import DataIngester, Passage, QAPair
from embed_index import EmbeddingIndexer
from rag import SanskritRAG, CitationValidator, SanskritPromptTemplate
from utils.config_validator import ConfigValidator


class TestDataIngestion:
    """Test data ingestion and validation."""
    
    def test_load_fixture_passages(self):
        """Test loading passages from test fixtures."""
        fixtures_path = Path(__file__).parent / "fixtures"
        ingester = DataIngester(fixtures_path.parent.parent)
        
        passages = ingester.load_passages("tests/fixtures/passages.jsonl")
        
        assert len(passages) == 5
        assert passages[0].id == "test_gita_001"
        assert "dharmakṣetre" in passages[0].text_iast
        assert passages[0].work == "TEST:Bhagavadgita"
        
    def test_load_fixture_qa_pairs(self):
        """Test loading QA pairs from test fixtures."""
        fixtures_path = Path(__file__).parent / "fixtures"
        ingester = DataIngester(fixtures_path.parent.parent)
        
        # First load passages to get valid IDs
        passages = ingester.load_passages("tests/fixtures/passages.jsonl")
        passage_ids = {p.id for p in passages}
        
        qa_pairs = ingester.load_qa_pairs("tests/fixtures/qa_pairs.jsonl", passage_ids)
        
        assert len(qa_pairs) == 5
        assert qa_pairs[0].id == "test_qa_001"
        assert "Bhagavad Gita" in qa_pairs[0].question
        assert qa_pairs[0].difficulty == "easy"
        assert "test_gita_001" in qa_pairs[0].related_passage_ids
        
    def test_passage_validation(self):
        """Test passage data validation."""
        # Test valid passage
        valid_data = {
            "id": "test_001",
            "text_devanagari": "धर्म",
            "text_iast": "dharma",
            "work": "Test",
            "chapter": "1",
            "verse": "1",
            "language": "sanskrit",
            "source_url": "https://test.com",
            "notes": "test note"
        }
        
        passage = Passage.from_dict(valid_data)
        assert passage.id == "test_001"
        
        # Test missing field
        invalid_data = valid_data.copy()
        del invalid_data["id"]
        
        with pytest.raises(ValueError, match="Missing required field: id"):
            Passage.from_dict(invalid_data)
    
    def test_cross_reference_validation(self):
        """Test cross-reference validation between passages and QA pairs."""
        fixtures_path = Path(__file__).parent / "fixtures"
        ingester = DataIngester(fixtures_path.parent.parent)
        
        passages = ingester.load_passages("tests/fixtures/passages.jsonl")
        qa_pairs = ingester.load_qa_pairs("tests/fixtures/qa_pairs.jsonl")
        
        stats = ingester.validate_cross_references(passages, qa_pairs)
        
        assert stats['total_passages'] == 5
        assert stats['total_qa_pairs'] == 5
        assert stats['qa_with_valid_refs'] > 0
        # All our test QA pairs have valid references
        assert len(stats['invalid_references']) == 0


class TestEmbeddingAndIndexing:
    """Test embedding generation and FAISS indexing."""
    
    @pytest.fixture
    def sample_passages(self):
        """Create sample passages for testing."""
        return [
            Passage(
                id="test_1",
                text_devanagari="धर्म",
                text_iast="dharma",
                work="Test",
                chapter="1",
                verse="1",
                language="sanskrit",
                source_url="https://test.com",
                notes="righteousness"
            ),
            Passage(
                id="test_2",
                text_devanagari="कर्म",
                text_iast="karma",
                work="Test",
                chapter="1",
                verse="2",
                language="sanskrit",
                source_url="https://test.com",
                notes="action"
            )
        ]
    
    @pytest.mark.skipif(
        os.getenv("SKIP_EMBEDDING_TESTS") == "1",
        reason="Embedding tests skipped (requires sentence-transformers)"
    )
    def test_embedding_generation(self, sample_passages):
        """Test embedding generation."""
        indexer = EmbeddingIndexer("sentence-transformers/all-MiniLM-L6-v2")
        
        if not indexer.load_embedding_model():
            pytest.skip("Embedding model not available")
        
        embeddings, passage_ids = indexer.generate_embeddings(sample_passages)
        
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] > 0  # Should have some dimensions
        assert passage_ids == ["test_1", "test_2"]
    
    @pytest.mark.skipif(
        os.getenv("SKIP_EMBEDDING_TESTS") == "1",
        reason="Embedding tests skipped"
    )
    def test_faiss_index_creation(self, sample_passages):
        """Test FAISS index creation."""
        indexer = EmbeddingIndexer("sentence-transformers/all-MiniLM-L6-v2")
        
        if not indexer.load_embedding_model():
            pytest.skip("Embedding model not available")
        
        embeddings, passage_ids = indexer.generate_embeddings(sample_passages)
        index = indexer.create_faiss_index(embeddings)
        
        assert index.ntotal == 2
        
        # Test search
        results = indexer.search(index, {"passages": [p.to_dict() for p in sample_passages]}, "dharma", k=1)
        assert len(results) == 1
        assert "dharma" in results[0]['passage']['text_iast']


class TestCitationValidation:
    """Test citation extraction and validation."""
    
    def test_citation_extraction(self):
        """Test extracting citations from text."""
        valid_passage_ids = {"test_gita_001", "test_dharma_001", "test_karma_001"}
        validator = CitationValidator(valid_passage_ids)
        
        # Test single citation
        text1 = "This is about dharma [test_dharma_001] in Sanskrit."
        citations1 = validator.extract_citations(text1)
        assert citations1 == ["test_dharma_001"]
        
        # Test multiple citations
        text2 = "References [test_gita_001, test_karma_001] show different concepts."
        citations2 = validator.extract_citations(text2)
        assert set(citations2) == {"test_gita_001", "test_karma_001"}
        
        # Test no citations
        text3 = "No citations in this text."
        citations3 = validator.extract_citations(text3)
        assert citations3 == []
    
    def test_citation_validation(self):
        """Test validating citations against known IDs."""
        valid_passage_ids = {"test_gita_001", "test_dharma_001"}
        validator = CitationValidator(valid_passage_ids)
        
        citations = ["test_gita_001", "test_invalid_001", "test_dharma_001"]
        valid, invalid = validator.validate_citations(citations)
        
        assert set(valid) == {"test_gita_001", "test_dharma_001"}
        assert invalid == ["test_invalid_001"]
    
    def test_response_validation(self):
        """Test complete response validation."""
        valid_passage_ids = {"test_gita_001", "test_dharma_001"}
        validator = CitationValidator(valid_passage_ids)
        
        response = "Dharma is important [test_dharma_001]. The Gita teaches [test_gita_001] about duty. Invalid reference [test_invalid_001]."
        
        validation = validator.validate_response(response)
        
        assert validation['total_citations'] == 3
        assert len(validation['valid_citations']) == 2
        assert len(validation['invalid_citations']) == 1
        assert "test_invalid_001" in validation['invalid_citations']


class TestPromptTemplate:
    """Test Sanskrit prompt template formatting."""
    
    def test_context_formatting(self):
        """Test formatting of retrieved passages for context."""
        from rag import RetrievalResult
        
        template = SanskritPromptTemplate()
        
        # Create mock retrieval results
        results = [
            RetrievalResult(
                passage_id="test_gita_001",
                passage={
                    "id": "test_gita_001",
                    "text_devanagari": "धर्मक्षेत्रे कुरुक्षेत्रे",
                    "text_iast": "dharmakṣetre kurukṣetre",
                    "work": "Bhagavad Gita",
                    "chapter": "1",
                    "verse": "1",
                    "source_url": "https://test.com",
                    "notes": "Opening verse"
                },
                score=0.95,
                relevance_rank=1
            )
        ]
        
        context = template.format_context(results)
        
        assert "[test_gita_001]" in context
        assert "dharmakṣetre" in context
        assert "Bhagavad Gita" in context
        assert "Opening verse" in context
    
    def test_prompt_creation(self):
        """Test complete prompt creation."""
        from rag import RetrievalResult
        
        template = SanskritPromptTemplate()
        
        results = [RetrievalResult(
            passage_id="test_001",
            passage={
                "id": "test_001",
                "text_devanagari": "धर्म",
                "text_iast": "dharma",
                "work": "Test",
                "chapter": "1",
                "verse": "1",
                "source_url": "https://test.com",
                "notes": ""
            },
            score=0.8,
            relevance_rank=1
        )]
        
        prompt = template.create_prompt("What is dharma?", results)
        
        assert "You are a Sanskrit tutor" in prompt
        assert "[test_001]" in prompt
        assert "What is dharma?" in prompt
        assert "CRITICAL CITATION REQUIREMENTS" in prompt


@pytest.mark.integration
class TestRAGIntegration:
    """Integration tests for the complete RAG system."""
    
    @pytest.fixture
    def temp_config_path(self):
        """Create a temporary config file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("""
model_path: null
gguf_local: false
embeddings_model: "sentence-transformers/all-MiniLM-L6-v2"
faiss_index_path: "tests/fixtures/test.index"
passages_file: "tests/fixtures/passages.jsonl"
qa_file: "tests/fixtures/qa_pairs.jsonl"
audio_folder: "tests/fixtures/audio_samples"
retrieval_k: 2
max_tokens: 100
temperature: 0.7
""")
            f.flush()
            yield f.name
        os.unlink(f.name)
    
    @pytest.mark.skipif(
        os.getenv("SKIP_INTEGRATION_TESTS") == "1", 
        reason="Integration tests skipped"
    )
    def test_rag_initialization(self, temp_config_path):
        """Test RAG system initialization with fixtures."""
        rag = SanskritRAG(temp_config_path)
        
        # Mock LLM manager to avoid requiring actual model
        with patch('rag.create_llm_manager') as mock_llm:
            mock_manager = Mock()
            mock_manager.get_current_backend_info.return_value = {"backend": "Mock"}
            mock_llm.return_value = mock_manager
            rag.llm_manager = mock_manager
            
            # Mock embedding model loading
            with patch.object(rag, 'indexer') as mock_indexer:
                mock_indexer.load_embedding_model.return_value = True
                mock_indexer.load_faiss_index.return_value = (Mock(), {"passage_ids": ["test_1"]})
                
                success = rag.initialize()
                assert success
    
    def test_passage_retrieval_mock(self):
        """Test passage retrieval with mocked components."""
        # Create mock RAG system
        rag = SanskritRAG("dummy_config")
        rag.indexer = Mock()
        
        # Mock search results
        mock_results = [
            {
                'passage_id': 'test_gita_001',
                'passage': {'id': 'test_gita_001', 'text_iast': 'dharmakṣetre'},
                'score': 0.9
            }
        ]
        rag.indexer.search.return_value = mock_results
        rag.index = Mock()
        rag.metadata = Mock()
        rag.retrieval_k = 2
        
        results = rag.retrieve_passages("dharma")
        
        assert len(results) == 1
        assert results[0].passage_id == 'test_gita_001'
        assert results[0].score == 0.9


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_missing_user_assets(self):
        """Test validation with missing user assets."""
        # Create temporary directory without required files
        with tempfile.TemporaryDirectory() as temp_dir:
            validator = ConfigValidator(temp_dir)
            success = validator.validate_all()
            
            assert not success  # Should fail validation
    
    def test_fixture_validation(self):
        """Test validation with test fixtures."""
        fixtures_path = Path(__file__).parent / "fixtures"
        base_path = fixtures_path.parent.parent
        
        # Create a temporary config pointing to fixtures
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(f"""
embeddings_model: "sentence-transformers/all-MiniLM-L6-v2"
faiss_index_path: "tests/fixtures/test.index"  
passages_file: "tests/fixtures/passages.jsonl"
qa_file: "tests/fixtures/qa_pairs.jsonl"
""")
            f.flush()
            config_path = f.name
        
        try:
            # Copy config to user_assets for validation
            user_assets = base_path / "user_assets"
            user_assets.mkdir(exist_ok=True)
            
            shutil.copy(config_path, user_assets / "config.yaml")
            shutil.copy(fixtures_path / "passages.jsonl", user_assets / "passages.jsonl")
            shutil.copy(fixtures_path / "qa_pairs.jsonl", user_assets / "qa_pairs.jsonl")
            
            validator = ConfigValidator(base_path)
            success = validator.validate_all()
            
            # Clean up
            shutil.rmtree(user_assets, ignore_errors=True)
            
            assert success  # Should pass with fixtures
            
        finally:
            os.unlink(config_path)


def test_user_assets_mode():
    """Test with actual user assets if available."""
    user_assets_path = Path("user_assets")
    
    if not user_assets_path.exists():
        pytest.skip("No user_assets directory found")
    
    validator = ConfigValidator()
    if not validator.validate_all():
        pytest.skip("User assets validation failed")
    
    # If we get here, user has provided valid assets
    ingester = DataIngester()
    try:
        data = ingester.load_all_data("user_assets/config.yaml")
        
        print(f"\nUser asset summary:")
        print(f"  Passages: {len(data['passages'])}")
        print(f"  QA pairs: {len(data['qa_pairs'])}")
        print(f"  Validation stats: {data['validation_stats']}")
        
        assert len(data['passages']) > 0
        assert len(data['qa_pairs']) > 0
        
    except Exception as e:
        pytest.fail(f"Failed to load user assets: {str(e)}")


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
