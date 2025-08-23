#!/usr/bin/env python3
"""
Data ingestion and normalization for Sanskrit Tutor RAG system.
Processes user-supplied passages.jsonl and qa_pairs.jsonl files.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, asdict
from utils.config_validator import ConfigValidator


@dataclass
class Passage:
    """Normalized passage data structure."""
    id: str
    text_devanagari: str
    text_iast: str
    work: str
    chapter: str
    verse: str
    language: str
    source_url: str
    notes: str
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Passage':
        """Create Passage from dictionary with validation."""
        required_fields = [
            'id', 'text_devanagari', 'text_iast', 'work', 
            'chapter', 'verse', 'language', 'source_url', 'notes'
        ]
        
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
                
        return cls(**{field: data[field] for field in required_fields})
    
    def get_searchable_text(self) -> str:
        """Get combined text for embedding/search."""
        parts = [
            self.text_devanagari,
            self.text_iast,
            f"{self.work} {self.chapter}.{self.verse}",
            self.notes
        ]
        return " ".join(part for part in parts if part.strip())


@dataclass  
class QAPair:
    """Normalized QA pair data structure."""
    id: str
    question: str
    answer: str
    difficulty: str
    related_passage_ids: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QAPair':
        """Create QAPair from dictionary with validation."""
        required_fields = ['id', 'question', 'answer', 'difficulty', 'related_passage_ids']
        
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Missing required field: {field}")
                
        # Ensure related_passage_ids is a list
        related_ids = data['related_passage_ids']
        if not isinstance(related_ids, list):
            related_ids = [related_ids] if related_ids else []
            
        return cls(
            id=data['id'],
            question=data['question'],
            answer=data['answer'],
            difficulty=data['difficulty'],
            related_passage_ids=related_ids
        )


class DataIngester:
    """Handles ingestion and validation of user-supplied data files."""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.validator = ConfigValidator(base_path)
        
    def load_passages(self, passages_file: str) -> List[Passage]:
        """
        Load and validate passages from JSONL file.
        
        Args:
            passages_file: Path to passages.jsonl file
            
        Returns:
            List of validated Passage objects
            
        Raises:
            FileNotFoundError: If passages file doesn't exist
            ValueError: If passages have invalid format
        """
        passages_path = self.base_path / passages_file
        
        if not passages_path.exists():
            raise FileNotFoundError(
                f"Passages file not found: {passages_path.absolute()}\n"
                f"Please create this file according to the documentation."
            )
            
        passages = []
        passage_ids = set()
        
        try:
            with open(passages_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        data = json.loads(line)
                        passage = Passage.from_dict(data)
                        
                        # Check for duplicate IDs
                        if passage.id in passage_ids:
                            raise ValueError(f"Duplicate passage ID: {passage.id}")
                        passage_ids.add(passage.id)
                        
                        # Basic validation
                        if not passage.text_devanagari.strip() and not passage.text_iast.strip():
                            raise ValueError(f"Passage {passage.id} has empty text")
                            
                        passages.append(passage)
                        
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON on line {line_num}: {str(e)}")
                    except Exception as e:
                        raise ValueError(f"Error processing line {line_num}: {str(e)}")
                        
        except Exception as e:
            if isinstance(e, (FileNotFoundError, ValueError)):
                raise
            raise ValueError(f"Error reading passages file: {str(e)}")
            
        if not passages:
            raise ValueError("No valid passages found in file")
            
        print(f"Loaded {len(passages)} passages from {passages_file}")
        return passages
    
    def load_qa_pairs(self, qa_file: str, valid_passage_ids: Set[str] = None) -> List[QAPair]:
        """
        Load and validate QA pairs from JSONL file.
        
        Args:
            qa_file: Path to qa_pairs.jsonl file
            valid_passage_ids: Set of valid passage IDs for reference validation
            
        Returns:
            List of validated QAPair objects
            
        Raises:
            FileNotFoundError: If QA file doesn't exist
            ValueError: If QA pairs have invalid format
        """
        qa_path = self.base_path / qa_file
        
        if not qa_path.exists():
            raise FileNotFoundError(
                f"QA pairs file not found: {qa_path.absolute()}\n"
                f"Please create this file according to the documentation."
            )
            
        qa_pairs = []
        qa_ids = set()
        invalid_refs = []
        
        try:
            with open(qa_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        data = json.loads(line)
                        qa_pair = QAPair.from_dict(data)
                        
                        # Check for duplicate IDs
                        if qa_pair.id in qa_ids:
                            raise ValueError(f"Duplicate QA pair ID: {qa_pair.id}")
                        qa_ids.add(qa_pair.id)
                        
                        # Basic validation
                        if not qa_pair.question.strip():
                            raise ValueError(f"QA pair {qa_pair.id} has empty question")
                        if not qa_pair.answer.strip():
                            raise ValueError(f"QA pair {qa_pair.id} has empty answer")
                            
                        # Validate passage references if provided
                        if valid_passage_ids:
                            for ref_id in qa_pair.related_passage_ids:
                                if ref_id not in valid_passage_ids:
                                    invalid_refs.append(f"QA {qa_pair.id} references unknown passage: {ref_id}")
                                    
                        qa_pairs.append(qa_pair)
                        
                    except json.JSONDecodeError as e:
                        raise ValueError(f"Invalid JSON on line {line_num}: {str(e)}")
                    except Exception as e:
                        raise ValueError(f"Error processing line {line_num}: {str(e)}")
                        
        except Exception as e:
            if isinstance(e, (FileNotFoundError, ValueError)):
                raise
            raise ValueError(f"Error reading QA pairs file: {str(e)}")
            
        if not qa_pairs:
            raise ValueError("No valid QA pairs found in file")
            
        # Report invalid references but don't fail
        if invalid_refs:
            print("WARNING: Found invalid passage references:")
            for ref in invalid_refs[:5]:  # Limit output
                print(f"  {ref}")
            if len(invalid_refs) > 5:
                print(f"  ... and {len(invalid_refs) - 5} more")
            print("These QA pairs will still be loaded but citations may not work properly.")
            
        print(f"Loaded {len(qa_pairs)} QA pairs from {qa_file}")
        return qa_pairs
    
    def normalize_text(self, text: str) -> str:
        """
        Normalize Sanskrit text for consistent processing.
        
        Args:
            text: Input text to normalize
            
        Returns:
            Normalized text
        """
        if not text:
            return ""
            
        # Basic normalization
        text = text.strip()
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common formatting artifacts
        text = re.sub(r'[\u200c\u200d]', '', text)  # Zero-width joiners
        
        return text
    
    def extract_citations_from_answer(self, answer: str) -> List[str]:
        """
        Extract citation IDs from answer text.
        Citations should be in format [passage_id] or [passage_id1, passage_id2].
        
        Args:
            answer: Answer text that may contain citations
            
        Returns:
            List of citation IDs found
        """
        citations = []
        
        # Pattern to match [citation] format
        pattern = r'\[([^\]]+)\]'
        matches = re.findall(pattern, answer)
        
        for match in matches:
            # Split by comma and clean up
            cite_ids = [cite.strip() for cite in match.split(',')]
            citations.extend(cite_ids)
            
        return citations
    
    def validate_cross_references(self, passages: List[Passage], qa_pairs: List[QAPair]) -> Dict[str, Any]:
        """
        Validate cross-references between passages and QA pairs.
        
        Args:
            passages: List of loaded passages
            qa_pairs: List of loaded QA pairs
            
        Returns:
            Dictionary with validation statistics and warnings
        """
        passage_ids = {p.id for p in passages}
        
        stats = {
            'total_passages': len(passages),
            'total_qa_pairs': len(qa_pairs),
            'qa_with_valid_refs': 0,
            'qa_with_invalid_refs': 0,
            'unreferenced_passages': [],
            'invalid_references': []
        }
        
        referenced_passage_ids = set()
        
        for qa in qa_pairs:
            has_valid_refs = False
            
            # Check declared references
            for ref_id in qa.related_passage_ids:
                if ref_id in passage_ids:
                    referenced_passage_ids.add(ref_id)
                    has_valid_refs = True
                else:
                    stats['invalid_references'].append(f"QA {qa.id} -> {ref_id}")
            
            # Check citations in answer text
            answer_citations = self.extract_citations_from_answer(qa.answer)
            for cite_id in answer_citations:
                if cite_id in passage_ids:
                    referenced_passage_ids.add(cite_id)
                    has_valid_refs = True
                else:
                    stats['invalid_references'].append(f"QA {qa.id} answer cites unknown: {cite_id}")
            
            if has_valid_refs:
                stats['qa_with_valid_refs'] += 1
            else:
                stats['qa_with_invalid_refs'] += 1
        
        # Find unreferenced passages
        stats['unreferenced_passages'] = list(passage_ids - referenced_passage_ids)
        
        return stats
    
    def load_all_data(self, config_path: str) -> Dict[str, Any]:
        """
        Load all data files specified in configuration.
        
        Args:
            config_path: Path to config.yaml file
            
        Returns:
            Dictionary containing loaded passages, QA pairs, and validation stats
        """
        # Validate user assets first
        if not self.validator.validate_all():
            raise ValueError("User asset validation failed. Please fix the reported issues.")
        
        # Load config
        import yaml
        with open(self.base_path / config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        # Load passages first
        passages = self.load_passages(config['passages_file'])
        passage_ids = {p.id for p in passages}
        
        # Load QA pairs with passage validation
        qa_pairs = self.load_qa_pairs(config['qa_file'], passage_ids)
        
        # Validate cross-references
        validation_stats = self.validate_cross_references(passages, qa_pairs)
        
        # Print summary
        print(f"\nData ingestion summary:")
        print(f"  Passages: {len(passages)}")
        print(f"  QA pairs: {len(qa_pairs)}")
        print(f"  QA pairs with valid references: {validation_stats['qa_with_valid_refs']}")
        
        if validation_stats['invalid_references']:
            print(f"  Invalid references: {len(validation_stats['invalid_references'])} (see warnings above)")
            
        if validation_stats['unreferenced_passages']:
            print(f"  Unreferenced passages: {len(validation_stats['unreferenced_passages'])}")
        
        return {
            'passages': passages,
            'qa_pairs': qa_pairs,
            'validation_stats': validation_stats,
            'config': config
        }


def main():
    """Command-line interface for data ingestion."""
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser(
        description="Ingest and validate Sanskrit Tutor data files"
    )
    parser.add_argument(
        "--config",
        default="user_assets/config.yaml",
        help="Path to configuration file (default: user_assets/config.yaml)"
    )
    parser.add_argument(
        "--output",
        help="Output directory for processed data (optional)"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate data without processing"
    )
    
    args = parser.parse_args()
    
    try:
        ingester = DataIngester()
        
        if args.validate_only:
            success = ingester.validator.validate_all()
            exit(0 if success else 1)
        
        # Load all data
        result = ingester.load_all_data(args.config)
        
        # Optionally save processed data
        if args.output:
            output_path = Path(args.output)
            output_path.mkdir(exist_ok=True)
            
            # Save processed passages
            passages_out = output_path / "processed_passages.jsonl"
            with open(passages_out, 'w', encoding='utf-8') as f:
                for passage in result['passages']:
                    f.write(json.dumps(passage.to_dict(), ensure_ascii=False) + '\n')
            
            # Save processed QA pairs  
            qa_out = output_path / "processed_qa_pairs.jsonl"
            with open(qa_out, 'w', encoding='utf-8') as f:
                for qa in result['qa_pairs']:
                    f.write(json.dumps(qa.to_dict(), ensure_ascii=False) + '\n')
            
            # Save validation stats
            stats_out = output_path / "validation_stats.json"
            with open(stats_out, 'w', encoding='utf-8') as f:
                json.dump(result['validation_stats'], f, indent=2, ensure_ascii=False)
                
            print(f"\nProcessed data saved to: {output_path.absolute()}")
        
        print("\nData ingestion completed successfully!")
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
