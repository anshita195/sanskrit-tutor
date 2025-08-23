#!/usr/bin/env python3
"""
Bhagavad Gita Data Conversion Script
Converts the Kaggle Bhagavad Gita dataset to our Sanskrit Tutor format.

This script:
1. Converts all slokas to passages.jsonl
2. Generates Q&A pairs from commentary and translations
3. Creates proper metadata and citations
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Any

def load_chapter_data(chapter_dir: Path) -> Dict[int, Dict]:
    """Load chapter metadata from JSON files."""
    chapters = {}
    for chapter_file in chapter_dir.glob("*.json"):
        if "chapter_" in chapter_file.name:
            with open(chapter_file, 'r', encoding='utf-8') as f:
                chapter_data = json.load(f)
                chapters[chapter_data['chapter_number']] = chapter_data
    return chapters

def load_sloka_data(sloka_dir: Path) -> List[Dict]:
    """Load all sloka data from JSON files."""
    slokas = []
    for sloka_file in sloka_dir.glob("*.json"):
        if "slok_" in sloka_file.name:
            with open(sloka_file, 'r', encoding='utf-8') as f:
                sloka_data = json.load(f)
                slokas.append(sloka_data)
    
    # Sort by chapter and verse
    slokas.sort(key=lambda x: (x['chapter'], x['verse']))
    return slokas

def clean_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove verse numbers like ||1-1|| at the end
    text = re.sub(r'\|\|[\d\-]+\|\|$', '', text).strip()
    return text

def create_passage_from_sloka(sloka: Dict, chapter_info: Dict) -> Dict:
    """Convert a sloka to our passage format matching the expected schema."""
    
    # Get English translation for notes
    english_translation = ""
    if 'siva' in sloka and 'et' in sloka['siva']:
        english_translation = clean_text(sloka['siva']['et'])
    elif 'prabhu' in sloka and 'et' in sloka['prabhu']:
        english_translation = clean_text(sloka['prabhu']['et'])
    elif 'purohit' in sloka and 'et' in sloka['purohit']:
        english_translation = clean_text(sloka['purohit']['et'])
    
    # Get commentary for notes
    commentary = ""
    if 'siva' in sloka and 'ec' in sloka['siva']:
        commentary = clean_text(sloka['siva']['ec'])
        if len(commentary) > 500:
            commentary = commentary[:500] + "..."
    
    # Create notes with translation and commentary
    notes_parts = []
    if english_translation:
        notes_parts.append(f"English: {english_translation}")
    if commentary:
        notes_parts.append(f"Commentary: {commentary}")
    
    passage = {
        "id": sloka['_id'],
        "text_devanagari": clean_text(sloka['slok']),
        "text_iast": clean_text(sloka['transliteration']),
        "work": "Bhagavad Gita",
        "chapter": str(sloka['chapter']),
        "verse": str(sloka['verse']),
        "language": "sanskrit",
        "source_url": f"https://www.kaggle.com/datasets/prashant-tripathi/bhagavad-gita-api-database/data?select=slok",
        "notes": " | ".join(notes_parts) if notes_parts else ""
    }
    
    return passage

def generate_qa_pairs_from_sloka(sloka: Dict, chapter_info: Dict) -> List[Dict]:
    """Generate Q&A pairs from a sloka matching the expected schema."""
    qa_pairs = []
    
    # Q1: Basic meaning question
    if 'siva' in sloka and 'et' in sloka['siva']:
        qa_pairs.append({
            "id": f"{sloka['_id']}_meaning",
            "question": f"What is the meaning of Bhagavad Gita verse {sloka['chapter']}.{sloka['verse']}?",
            "answer": f"{clean_text(sloka['siva']['et'])} [{sloka['_id']}]",
            "difficulty": "easy",
            "related_passage_ids": [sloka['_id']]
        })
    
    # Q2: Transliteration question
    qa_pairs.append({
        "id": f"{sloka['_id']}_transliteration",
        "question": f"How is the Sanskrit verse {sloka['chapter']}.{sloka['verse']} transliterated?",
        "answer": f"{clean_text(sloka['transliteration'])} [{sloka['_id']}]",
        "difficulty": "medium",
        "related_passage_ids": [sloka['_id']]
    })
    
    # Q3: Chapter context question
    if chapter_info:
        qa_pairs.append({
            "id": f"{sloka['_id']}_context",
            "question": f"Which chapter of the Bhagavad Gita contains the verse '{clean_text(sloka['slok'])[:50]}...'?",
            "answer": f"This verse is from Chapter {sloka['chapter']} - {chapter_info.get('meaning', {}).get('en', chapter_info.get('name', ''))}. {chapter_info.get('summary', {}).get('en', '')[:200]}... [{sloka['_id']}]",
            "difficulty": "medium",
            "related_passage_ids": [sloka['_id']]
        })
    
    # Q4: Commentary-based question (if available)
    if 'siva' in sloka and 'ec' in sloka['siva']:
        commentary = clean_text(sloka['siva']['ec'])
        if len(commentary) > 100:
            qa_pairs.append({
                "id": f"{sloka['_id']}_commentary",
                "question": f"What is the deeper philosophical significance of verse {sloka['chapter']}.{sloka['verse']}?",
                "answer": f"{commentary[:400] + '...' if len(commentary) > 400 else commentary} [{sloka['_id']}]",
                "difficulty": "hard",
                "related_passage_ids": [sloka['_id']]
            })
    
    return qa_pairs

def main():
    """Main conversion function."""
    print("🚀 Starting Bhagavad Gita data conversion...")
    
    # Setup paths
    raw_data_dir = Path("raw_data")
    chapter_dir = raw_data_dir / "chapter"
    sloka_dir = raw_data_dir / "slok"
    output_dir = Path("user_assets")
    
    # Ensure output directory exists
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    print("📖 Loading chapter metadata...")
    chapters = load_chapter_data(chapter_dir)
    print(f"   Loaded {len(chapters)} chapters")
    
    print("📜 Loading slokas...")
    slokas = load_sloka_data(sloka_dir)
    print(f"   Loaded {len(slokas)} slokas")
    
    # Convert to passages
    print("🔄 Converting slokas to passages...")
    passages = []
    all_qa_pairs = []
    
    for i, sloka in enumerate(slokas):
        if i % 100 == 0:
            print(f"   Processing sloka {i+1}/{len(slokas)}")
        
        chapter_info = chapters.get(sloka['chapter'], {})
        
        # Create passage
        passage = create_passage_from_sloka(sloka, chapter_info)
        passages.append(passage)
        
        # Generate Q&A pairs
        qa_pairs = generate_qa_pairs_from_sloka(sloka, chapter_info)
        all_qa_pairs.extend(qa_pairs)
    
    # Write passages.jsonl
    passages_file = output_dir / "passages.jsonl"
    print(f"💾 Writing {len(passages)} passages to {passages_file}")
    with open(passages_file, 'w', encoding='utf-8') as f:
        for passage in passages:
            f.write(json.dumps(passage, ensure_ascii=False) + '\n')
    
    # Write qa_pairs.jsonl
    qa_file = output_dir / "qa_pairs.jsonl"
    print(f"💾 Writing {len(all_qa_pairs)} Q&A pairs to {qa_file}")
    with open(qa_file, 'w', encoding='utf-8') as f:
        for qa in all_qa_pairs:
            f.write(json.dumps(qa, ensure_ascii=False) + '\n')
    
    # Create config.yaml matching the expected schema
    config_content = """# Sanskrit Tutor Configuration - Bhagavad Gita Dataset
# Generated from Kaggle Bhagavad Gita API Database conversion

# Model configuration
model_path: null  # Set to "user_assets/models/your-model.gguf" if using local GGUF
gguf_local: false  # Set to true if using local GGUF model

# Required embedding model
embeddings_model: "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# FAISS index path
faiss_index_path: "data/faiss.index"

# Required data files
passages_file: "user_assets/passages.jsonl"
qa_file: "user_assets/qa_pairs.jsonl"

# Optional audio folder
audio_folder: "user_assets/audio_samples"

# Generated dataset info (informational only)
# {} passages, {} Q&A pairs from complete Bhagavad Gita
""".format(len(passages), len(all_qa_pairs))
    
    config_file = output_dir / "config.yaml"
    if not config_file.exists():
        print(f"📝 Creating config file: {config_file}")
        with open(config_file, 'w', encoding='utf-8') as f:
            f.write(config_content)
    else:
        print(f"⏭️  Config file already exists: {config_file}")
    
    print("\n✅ Conversion completed successfully!")
    print(f"   📊 Statistics:")
    print(f"      - Passages: {len(passages)}")
    print(f"      - Q&A pairs: {len(all_qa_pairs)}")
    print(f"      - Chapters: {len(chapters)}")
    print(f"      - Output files: {passages_file}, {qa_file}, {config_file}")
    print("\n🎯 Next steps:")
    print("   1. Run: python src/utils/config_validator.py")
    print("   2. Run: python src/embed_index.py --config user_assets/config.yaml")
    print("   3. Run: python src/ui_gradio.py --config user_assets/config.yaml")

if __name__ == "__main__":
    main()
