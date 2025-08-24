#!/usr/bin/env python3
"""
Create a manageable sample from the massive Sanskrit corpus.
Samples intelligently to preserve diversity while keeping the corpus size reasonable.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any
from collections import defaultdict


def create_stratified_sample(input_file: Path, output_file: Path, 
                           target_size: int = 5000, preserve_original: bool = True):
    """
    Create a stratified sample from the massive corpus.
    
    Args:
        input_file: Input passages.jsonl file
        output_file: Output sample file
        target_size: Target number of passages
        preserve_original: Whether to preserve all original passages
    """
    
    print(f"Creating sample from {input_file}")
    print(f"Target size: {target_size} passages")
    
    # Categories for stratified sampling
    original_passages = []
    kartik_passages = []
    
    # Read and categorize all passages
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                try:
                    data = json.loads(line)
                    source_type = data.get('source_type', 'existing')
                    
                    if source_type in ['existing', 'kaggle']:
                        original_passages.append(data)
                    elif source_type == 'kartik':
                        kartik_passages.append(data)
                        
                except json.JSONDecodeError:
                    print(f"Skipping invalid JSON on line {line_num}")
                    
            # Progress indicator for large files
            if line_num % 100000 == 0:
                print(f"Processed {line_num} lines...")
    
    print(f"Found {len(original_passages)} original passages")
    print(f"Found {len(kartik_passages)} Kartik passages")
    
    # Calculate sampling strategy
    if preserve_original:
        # Keep all original passages
        selected_passages = original_passages.copy()
        remaining_slots = target_size - len(original_passages)
        
        if remaining_slots > 0 and kartik_passages:
            # Sample from Kartik corpus
            sample_size = min(remaining_slots, len(kartik_passages))
            sampled_kartik = random.sample(kartik_passages, sample_size)
            selected_passages.extend(sampled_kartik)
            print(f"Sampled {len(sampled_kartik)} passages from Kartik corpus")
    else:
        # Sample proportionally from both
        total_available = len(original_passages) + len(kartik_passages)
        original_ratio = len(original_passages) / total_available
        
        original_sample_size = int(target_size * original_ratio)
        kartik_sample_size = target_size - original_sample_size
        
        selected_passages = []
        
        if original_sample_size > 0:
            original_sample = random.sample(original_passages, 
                                          min(original_sample_size, len(original_passages)))
            selected_passages.extend(original_sample)
        
        if kartik_sample_size > 0 and kartik_passages:
            kartik_sample = random.sample(kartik_passages, 
                                        min(kartik_sample_size, len(kartik_passages)))
            selected_passages.extend(kartik_sample)
    
    # Shuffle the final selection
    random.shuffle(selected_passages)
    
    # Save sample
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for passage in selected_passages:
            f.write(json.dumps(passage, ensure_ascii=False) + '\n')
    
    print(f"Created sample with {len(selected_passages)} passages")
    print(f"Saved to: {output_file}")
    
    # Show sample statistics
    source_counts = defaultdict(int)
    work_counts = defaultdict(int)
    
    for passage in selected_passages:
        source_counts[passage.get('source_type', 'unknown')] += 1
        work_counts[passage.get('work', 'unknown')] += 1
    
    print(f"\nSample composition:")
    for source, count in source_counts.items():
        print(f"  {source}: {count}")
    
    print(f"\nTop works:")
    for work, count in sorted(work_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"  {work}: {count}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Create manageable sample from massive corpus")
    parser.add_argument("--input", type=Path, default="user_assets/passages.jsonl",
                       help="Input passages file")
    parser.add_argument("--output", type=Path, default="user_assets/passages_sample.jsonl",
                       help="Output sample file")
    parser.add_argument("--size", type=int, default=5000,
                       help="Target sample size")
    parser.add_argument("--preserve-original", action="store_true", default=True,
                       help="Preserve all original passages")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible sampling")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    create_stratified_sample(
        input_file=args.input,
        output_file=args.output,
        target_size=args.size,
        preserve_original=args.preserve_original
    )


if __name__ == "__main__":
    main()
