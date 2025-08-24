#!/usr/bin/env python3
"""
Sanskrit Corpus Processor - Convert Kartik and GRETIL datasets to RAG-ready format.

This script processes two major Sanskrit corpora:
1. Kartik Corpus (sanskrit_corpus_kaggle) - Large clean Devanagari text
2. GRETIL - Canonical texts with scholarly citations

Output: Unified passages.jsonl file with proper source attribution.
"""

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import argparse
from bs4 import BeautifulSoup
import hashlib

# Import our normalizer
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))
from sanskrit_normalizer import SanskritNormalizer


@dataclass
class ProcessedPassage:
    """Standardized passage format for our corpus."""
    id: str
    text_devanagari: str
    text_iast: str
    work: str
    chapter: str
    verse: str
    language: str
    source_url: str
    notes: str
    source_type: str  # 'kartik' or 'gretil'


class KartikCorpusProcessor:
    """Process the large Kartik Sanskrit corpus (train.txt)."""
    
    def __init__(self, normalizer: SanskritNormalizer):
        self.normalizer = normalizer
        self.chunk_size = 150  # Characters per passage chunk
        self.overlap_size = 50  # Overlap between chunks
        
    def process_train_file(self, train_file: Path) -> List[ProcessedPassage]:
        """
        Process train.txt into chunked passages.
        
        Args:
            train_file: Path to train.txt
            
        Returns:
            List of processed passages
        """
        print(f"Processing Kartik corpus: {train_file}")
        
        # Read the entire file
        with open(train_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Basic cleanup
        content = unicodedata.normalize('NFKC', content)
        content = re.sub(r'\s+', ' ', content)  # Normalize whitespace
        
        # Split into lines first (respect natural text boundaries)
        lines = [line.strip() for line in content.split('\n') if line.strip()]
        
        passages = []
        passage_id = 0
        
        for line_idx, line in enumerate(lines):
            if len(line) < 30:  # Skip very short lines
                continue
                
            # Chunk long lines
            if len(line) > self.chunk_size:
                chunks = self._chunk_text(line)
                for chunk_idx, chunk in enumerate(chunks):
                    passage = self._create_kartik_passage(
                        chunk, passage_id, line_idx, chunk_idx
                    )
                    passages.append(passage)
                    passage_id += 1
            else:
                # Use whole line as passage
                passage = self._create_kartik_passage(
                    line, passage_id, line_idx, 0
                )
                passages.append(passage)
                passage_id += 1
                
            # Limit for testing (remove this for full processing)
            if len(passages) >= 1000:  # Process first 1000 for testing
                break
        
        print(f"Created {len(passages)} passages from Kartik corpus")
        return passages
    
    def _chunk_text(self, text: str) -> List[str]:
        """Chunk text at word boundaries with overlap."""
        words = text.split()
        chunks = []
        
        i = 0
        while i < len(words):
            # Calculate chunk end
            chunk_words = []
            char_count = 0
            
            while i < len(words) and char_count < self.chunk_size:
                word = words[i]
                if char_count + len(word) + 1 <= self.chunk_size:
                    chunk_words.append(word)
                    char_count += len(word) + 1
                    i += 1
                else:
                    break
            
            if chunk_words:
                chunks.append(' '.join(chunk_words))
                # Overlap: go back a few words
                overlap_words = min(len(chunk_words) // 3, 5)
                i = max(0, i - overlap_words)
            else:
                i += 1  # Skip if we can't fit even one word
        
        return chunks
    
    def _create_kartik_passage(self, text: str, passage_id: int, 
                              line_idx: int, chunk_idx: int) -> ProcessedPassage:
        """Create a standardized passage from Kartik corpus text."""
        
        # Generate ID
        id_str = f"kartik_{passage_id:06d}"
        
        # Transliterate to IAST
        text_iast = self.normalizer.transliterate_to_iast(text, 'devanagari')
        
        # Create hash for verification
        text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
        
        return ProcessedPassage(
            id=id_str,
            text_devanagari=text,
            text_iast=text_iast,
            work="Sanskrit_Corpus",
            chapter=str(line_idx),
            verse=str(chunk_idx),
            language="sanskrit",
            source_url=f"https://www.kaggle.com/datasets/disisbig/sanskrit-corpus#line_{line_idx}",
            notes=f"Kartik corpus chunk {chunk_idx}, hash:{text_hash}",
            source_type="kartik"
        )


class GretilCorpusProcessor:
    """Process GRETIL HTML files into structured passages."""
    
    def __init__(self, normalizer: SanskritNormalizer):
        self.normalizer = normalizer
        
    def process_gretil_directory(self, gretil_dir: Path) -> List[ProcessedPassage]:
        """
        Process all GRETIL HTML files.
        
        Args:
            gretil_dir: Root GRETIL directory
            
        Returns:
            List of processed passages
        """
        print(f"Processing GRETIL directory: {gretil_dir}")
        
        passages = []
        
        # Priority files to process
        priority_files = [
            # Upanishads
            "1_veda/4_upa/isaup_u.htm",
            "1_veda/4_upa/kathup_u.htm", 
            "1_veda/4_upa/mandup_u.htm",
            "1_veda/4_upa/chandogyu.htm",
            "1_veda/4_upa/brhadup_u.htm",
            "1_veda/4_upa/mundakau.htm",
            # Rig Veda samples
            "1_veda/1_sam/1_rv/rvpp_01u.htm",
            "1_veda/1_sam/1_rv/rvpp_03u.htm",
            # Mahabharata (contains Bhagavad Gita)
            "2_epic/mbh/mbh_06_u.htm",  # Book 6 contains Gita
        ]
        
        for file_path in priority_files:
            full_path = gretil_dir / file_path
            if full_path.exists():
                print(f"Processing: {file_path}")
                file_passages = self._process_gretil_file(full_path, file_path)
                passages.extend(file_passages)
            else:
                print(f"File not found: {file_path}")
        
        print(f"Created {len(passages)} passages from GRETIL corpus")
        return passages
    
    def _process_gretil_file(self, file_path: Path, relative_path: str) -> List[ProcessedPassage]:
        """Process a single GRETIL HTML file."""
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            return []
        
        # Parse HTML
        try:
            soup = BeautifulSoup(content, 'html.parser')
        except Exception as e:
            print(f"Error parsing HTML {file_path}: {e}")
            return []
        
        # Extract work name from path
        work_name = self._extract_work_name(relative_path)
        
        # Find text content (GRETIL files vary in structure)
        passages = []
        
        # Method 1: Look for <p> tags with Sanskrit text
        p_tags = soup.find_all('p')
        for i, p in enumerate(p_tags):
            text = p.get_text(strip=True)
            if self._is_sanskrit_text(text) and len(text.strip()) > 10:
                passage = self._create_gretil_passage(
                    text, work_name, relative_path, f"p_{i}", len(passages)
                )
                if passage:
                    passages.append(passage)
        
        # Method 2: Look for <pre> tags (common in GRETIL)
        pre_tags = soup.find_all('pre')
        for i, pre in enumerate(pre_tags):
            text = pre.get_text(strip=True)
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            
            for j, line in enumerate(lines):
                if self._is_sanskrit_text(line) and len(line.strip()) > 10:
                    passage = self._create_gretil_passage(
                        line, work_name, relative_path, f"pre_{i}_{j}", len(passages)
                    )
                    if passage:
                        passages.append(passage)
        
        # Limit passages per file for testing
        return passages[:50]  # Max 50 passages per file
    
    def _extract_work_name(self, relative_path: str) -> str:
        """Extract work name from GRETIL file path."""
        path_parts = relative_path.split('/')
        
        # Map common patterns
        if "isa" in relative_path.lower():
            return "Ishavasya_Upanishad"
        elif "katha" in relative_path.lower():
            return "Katha_Upanishad"
        elif "mand" in relative_path.lower():
            return "Mandukya_Upanishad"
        elif "chand" in relative_path.lower():
            return "Chandogya_Upanishad"
        elif "brha" in relative_path.lower():
            return "Brihadaranyaka_Upanishad"
        elif "mund" in relative_path.lower():
            return "Mundaka_Upanishad"
        elif "rv" in relative_path.lower():
            return "Rig_Veda"
        elif "mbh" in relative_path.lower():
            return "Mahabharata"
        else:
            return f"GRETIL_{path_parts[-1].replace('.htm', '')}"
    
    def _is_sanskrit_text(self, text: str) -> bool:
        """Check if text contains Sanskrit (Devanagari or IAST)."""
        if not text or len(text) < 5:
            return False
        
        # Check for Devanagari characters
        devanagari_count = sum(1 for char in text if '\u0900' <= char <= '\u097F')
        
        # Check for IAST diacritics
        iast_chars = set('āīūṛṝḷḹēōṃḥṅñṭḍṇśṣ')
        iast_count = sum(1 for char in text if char in iast_chars)
        
        # Must have significant Sanskrit content
        return (devanagari_count >= 3) or (iast_count >= 2)
    
    def _create_gretil_passage(self, text: str, work_name: str, 
                              relative_path: str, element_id: str, 
                              passage_idx: int) -> Optional[ProcessedPassage]:
        """Create a standardized passage from GRETIL text."""
        
        # Clean up text
        text = unicodedata.normalize('NFKC', text.strip())
        text = re.sub(r'\s+', ' ', text)
        
        if len(text) < 10:  # Skip very short texts
            return None
        
        # Determine if input is Devanagari or IAST
        script = self.normalizer.detect_script(text)
        
        if script == 'devanagari':
            text_devanagari = text
            text_iast = self.normalizer.transliterate_to_iast(text, 'devanagari')
        elif script == 'iast':
            text_iast = text
            text_devanagari = self.normalizer.transliterate_to_devanagari(text, 'iast')
        else:
            # Skip non-Sanskrit text
            return None
        
        # Generate ID
        file_hash = hashlib.md5(relative_path.encode()).hexdigest()[:6]
        id_str = f"gretil_{file_hash}_{passage_idx:03d}"
        
        # Extract chapter/verse info if possible
        chapter, verse = self._extract_verse_info(text)
        
        return ProcessedPassage(
            id=id_str,
            text_devanagari=text_devanagari,
            text_iast=text_iast,
            work=work_name,
            chapter=chapter,
            verse=verse,
            language="sanskrit",
            source_url=f"http://gretil.sub.uni-goettingen.de/gretil/{relative_path}",
            notes=f"GRETIL {element_id}",
            source_type="gretil"
        )
    
    def _extract_verse_info(self, text: str) -> Tuple[str, str]:
        """Try to extract chapter/verse information from text."""
        
        # Look for common patterns like "1.1", "2.47", etc.
        verse_pattern = r'(\d+)\.(\d+)'
        match = re.search(verse_pattern, text)
        
        if match:
            return match.group(1), match.group(2)
        
        # Look for Roman numerals or other patterns
        roman_pattern = r'([IVX]+)\.(\d+)'
        match = re.search(roman_pattern, text)
        
        if match:
            return match.group(1), match.group(2)
        
        # Default values
        return "0", "0"


class CorpusBuilder:
    """Main corpus builder that combines all sources."""
    
    def __init__(self):
        self.normalizer = SanskritNormalizer()
        self.kartik_processor = KartikCorpusProcessor(self.normalizer)
        self.gretil_processor = GretilCorpusProcessor(self.normalizer)
    
    def build_corpus(self, 
                     kartik_dir: Optional[Path] = None,
                     gretil_dir: Optional[Path] = None,
                     output_file: Path = None) -> List[ProcessedPassage]:
        """
        Build complete corpus from all sources.
        
        Args:
            kartik_dir: Directory containing sanskrit_corpus_kaggle
            gretil_dir: Directory containing gretil files
            output_file: Where to save the final corpus
            
        Returns:
            List of all processed passages
        """
        all_passages = []
        
        # Process Kartik corpus
        if kartik_dir:
            train_file = kartik_dir / "train.txt"
            if train_file.exists():
                kartik_passages = self.kartik_processor.process_train_file(train_file)
                all_passages.extend(kartik_passages)
                print(f"Added {len(kartik_passages)} Kartik passages")
        
        # Process GRETIL corpus
        if gretil_dir and gretil_dir.exists():
            gretil_passages = self.gretil_processor.process_gretil_directory(gretil_dir)
            all_passages.extend(gretil_passages)
            print(f"Added {len(gretil_passages)} GRETIL passages")
        
        # Save to file if specified
        if output_file:
            self.save_corpus(all_passages, output_file)
        
        print(f"Total corpus size: {len(all_passages)} passages")
        return all_passages
    
    def save_corpus(self, passages: List[ProcessedPassage], output_file: Path):
        """Save corpus to JSONL file."""
        
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for passage in passages:
                json_line = json.dumps(asdict(passage), ensure_ascii=False)
                f.write(json_line + '\n')
        
        print(f"Saved {len(passages)} passages to {output_file}")
    
    def merge_with_existing(self, new_passages: List[ProcessedPassage], 
                           existing_file: Path) -> List[ProcessedPassage]:
        """Merge new passages with existing corpus."""
        
        all_passages = []
        existing_ids = set()
        
        # Load existing passages
        if existing_file.exists():
            with open(existing_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            data = json.loads(line)
                            # Convert to ProcessedPassage if needed
                            if 'source_type' not in data:
                                data['source_type'] = 'existing'
                            
                            passage = ProcessedPassage(**data)
                            all_passages.append(passage)
                            existing_ids.add(passage.id)
                        except Exception as e:
                            print(f"Error loading existing passage: {e}")
        
        # Add new passages (avoid duplicates)
        added_count = 0
        for passage in new_passages:
            if passage.id not in existing_ids:
                all_passages.append(passage)
                existing_ids.add(passage.id)
                added_count += 1
        
        print(f"Merged {added_count} new passages with {len(all_passages) - added_count} existing")
        return all_passages


def main():
    """Command-line interface for corpus processing."""
    
    parser = argparse.ArgumentParser(description="Process Sanskrit corpora for RAG system")
    parser.add_argument("--kartik-dir", type=Path, help="Path to sanskrit_corpus_kaggle directory")
    parser.add_argument("--gretil-dir", type=Path, help="Path to gretil directory")
    parser.add_argument("--output", type=Path, default="user_assets/processed_passages.jsonl", 
                       help="Output file for processed corpus")
    parser.add_argument("--merge-existing", type=Path, help="Merge with existing passages file")
    parser.add_argument("--test-mode", action="store_true", help="Process only small samples for testing")
    
    args = parser.parse_args()
    
    # Initialize builder
    builder = CorpusBuilder()
    
    # Build corpus
    passages = builder.build_corpus(
        kartik_dir=args.kartik_dir,
        gretil_dir=args.gretil_dir,
        output_file=args.output
    )
    
    # Merge with existing if requested
    if args.merge_existing:
        passages = builder.merge_with_existing(passages, args.merge_existing)
        builder.save_corpus(passages, args.output)
    
    # Show statistics
    print(f"\nCorpus Statistics:")
    print(f"Total passages: {len(passages)}")
    
    source_counts = {}
    for passage in passages:
        source_counts[passage.source_type] = source_counts.get(passage.source_type, 0) + 1
    
    for source, count in source_counts.items():
        print(f"  {source}: {count}")
    
    print(f"\nSample passages:")
    for i, passage in enumerate(passages[:3]):
        print(f"{i+1}. [{passage.id}] {passage.work}")
        print(f"   Devanagari: {passage.text_devanagari[:60]}...")
        print(f"   IAST: {passage.text_iast[:60]}...")
        print()


if __name__ == "__main__":
    main()
