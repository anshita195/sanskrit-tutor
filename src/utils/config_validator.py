#!/usr/bin/env python3
"""
Config validator for Sanskrit Tutor RAG system.
Checks for required user-supplied assets and provides clear error messages.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Any


class UserAssetError(Exception):
    """Raised when required user assets are missing or invalid."""
    pass


class ConfigValidator:
    """Validates user-supplied assets and configuration."""
    
    def __init__(self, base_path: str = "."):
        self.base_path = Path(base_path)
        self.user_assets_path = self.base_path / "user_assets"
        self.errors = []
        
    def validate_all(self) -> bool:
        """
        Validate all required user assets.
        Returns True if all validations pass, False otherwise.
        Prints detailed error messages for any missing assets.
        """
        self.errors = []
        
        # Check basic structure
        self._check_user_assets_directory()
        
        # Check required files
        config = self._check_config_yaml()
        self._check_passages_jsonl()
        self._check_qa_pairs_jsonl()
        
        # Check model-related assets if specified
        if config:
            self._check_model_assets(config)
            
        if self.errors:
            self._print_errors()
            return False
            
        self._print_success()
        return True
        
    def _check_user_assets_directory(self):
        """Check if user_assets directory exists."""
        if not self.user_assets_path.exists():
            self.errors.append({
                "file": "user_assets/ directory",
                "error": "Directory does not exist",
                "fix": f"Create the user_assets directory at: {self.user_assets_path.absolute()}",
                "example": "mkdir user_assets"
            })
            
    def _check_config_yaml(self) -> Optional[Dict[str, Any]]:
        """Check and load config.yaml file."""
        config_path = self.user_assets_path / "config.yaml"
        
        if not config_path.exists():
            self.errors.append({
                "file": "user_assets/config.yaml",
                "error": "Configuration file is missing",
                "fix": f"Create config.yaml at: {config_path.absolute()}",
                "example": """model_path: "user_assets/models/mistral-7b-instruct.gguf"  # or null
gguf_local: true
embeddings_model: "sentence-transformers/all-mpnet-base-v2"
faiss_index_path: "data/faiss.index"
passages_file: "user_assets/passages.jsonl"
qa_file: "user_assets/qa_pairs.jsonl"
audio_folder: "user_assets/audio_samples\""""
            })
            return None
            
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            # Validate required config keys
            required_keys = [
                'embeddings_model', 'faiss_index_path', 
                'passages_file', 'qa_file'
            ]
            
            for key in required_keys:
                if key not in config:
                    self.errors.append({
                        "file": "user_assets/config.yaml",
                        "error": f"Missing required key: {key}",
                        "fix": f"Add '{key}' to your config.yaml",
                        "example": f"{key}: <appropriate_value>"
                    })
                    
            return config
            
        except yaml.YAMLError as e:
            self.errors.append({
                "file": "user_assets/config.yaml",
                "error": f"Invalid YAML format: {str(e)}",
                "fix": "Fix the YAML syntax in config.yaml",
                "example": "Use proper YAML indentation and syntax"
            })
            return None
            
    def _check_passages_jsonl(self):
        """Check passages.jsonl file."""
        passages_path = self.user_assets_path / "passages.jsonl"
        
        if not passages_path.exists():
            self.errors.append({
                "file": "user_assets/passages.jsonl",
                "error": "Passages file is missing",
                "fix": f"Create passages.jsonl at: {passages_path.absolute()}",
                "example": '''{"id":"gretil_shloka_001","text_devanagari":"त्वमेव माता च पिता त्वमेव","text_iast":"tvameva mata ca pita tvameva","work":"GRETIL:Bhagavadgita","chapter":"2","verse":"20","language":"sanskrit","source_url":"https://example.org/gretil/bg/2/20","notes":""}
{"id":"gretil_shloka_002","text_devanagari":"त्वमेव बन्धुश्च सखा त्वमेव","text_iast":"tvameva bandhusca sakha tvameva","work":"GRETIL:Bhagavadgita","chapter":"2","verse":"21","language":"sanskrit","source_url":"https://example.org/gretil/bg/2/21","notes":""}'''
            })
            return
            
        # Validate file format
        self._validate_jsonl_format(passages_path, "passages", [
            "id", "text_devanagari", "text_iast", "work", 
            "chapter", "verse", "language", "source_url", "notes"
        ])
        
    def _check_qa_pairs_jsonl(self):
        """Check qa_pairs.jsonl file."""
        qa_path = self.user_assets_path / "qa_pairs.jsonl"
        
        if not qa_path.exists():
            self.errors.append({
                "file": "user_assets/qa_pairs.jsonl",
                "error": "QA pairs file is missing",
                "fix": f"Create qa_pairs.jsonl at: {qa_path.absolute()}",
                "example": '''{"id":"qa_001","question":"What is the meaning of 'tvameva'?","answer":"You alone — tvam + eva. [gretil_shloka_001]","difficulty":"easy","related_passage_ids":["gretil_shloka_001"]}
{"id":"qa_002","question":"Translate 'mata ca pita'","answer":"Mother and father. [gretil_shloka_001]","difficulty":"easy","related_passage_ids":["gretil_shloka_001"]}'''
            })
            return
            
        # Validate file format
        self._validate_jsonl_format(qa_path, "qa_pairs", [
            "id", "question", "answer", "difficulty", "related_passage_ids"
        ])
        
    def _check_model_assets(self, config: Dict[str, Any]):
        """Check model-related assets based on config."""
        if config.get('gguf_local', False):
            model_path = config.get('model_path')
            if model_path:
                full_model_path = self.base_path / model_path
                if not full_model_path.exists():
                    self.errors.append({
                        "file": model_path,
                        "error": "Local GGUF model file not found",
                        "fix": f"Place your GGUF model at: {full_model_path.absolute()}",
                        "example": "Download a GGUF model (e.g., Mistral-7B-Instruct) and save it at the specified path, or set gguf_local: false to use hosted inference"
                    })
                    
    def _validate_jsonl_format(self, file_path: Path, file_type: str, required_fields: List[str]):
        """Validate JSONL file format and required fields."""
        try:
            line_count = 0
            with open(file_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line_count += 1
                    line = line.strip()
                    if not line:
                        continue
                        
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as e:
                        self.errors.append({
                            "file": str(file_path),
                            "error": f"Invalid JSON on line {line_num}: {str(e)}",
                            "fix": f"Fix JSON syntax on line {line_num}",
                            "example": "Each line must be valid JSON"
                        })
                        return
                        
                    # Check required fields (only for first few lines to avoid spam)
                    if line_num <= 3:
                        for field in required_fields:
                            if field not in obj:
                                self.errors.append({
                                    "file": str(file_path),
                                    "error": f"Missing required field '{field}' on line {line_num}",
                                    "fix": f"Add '{field}' field to all objects in {file_type}.jsonl",
                                    "example": f'"{field}": "appropriate_value"'
                                })
                                
            if line_count < 5:
                self.errors.append({
                    "file": str(file_path),
                    "error": f"File contains only {line_count} entries (minimum 5 recommended for testing)",
                    "fix": f"Add more entries to {file_type}.jsonl",
                    "example": f"Add at least 5 entries for basic functionality"
                })
                
        except Exception as e:
            self.errors.append({
                "file": str(file_path),
                "error": f"Cannot read file: {str(e)}",
                "fix": f"Ensure file exists and is readable",
                "example": f"Check file permissions and encoding (should be UTF-8)"
            })
            
    def _print_errors(self):
        """Print all validation errors with clear instructions."""
        print("\n" + "="*80)
        print("ERROR: Missing or invalid user assets")
        print("="*80)
        print("\nThe Sanskrit Tutor requires specific data files that you must provide.")
        print("Please fix the following issues:\n")
        
        for i, error in enumerate(self.errors, 1):
            print(f"{i}. FILE: {error['file']}")
            print(f"   ERROR: {error['error']}")
            print(f"   FIX: {error['fix']}")
            if error.get('example'):
                print(f"   EXAMPLE:")
                for line in error['example'].split('\n'):
                    print(f"   {line}")
            print()
            
        print("After fixing these issues, run the command again.")
        print("="*80)
        
    def _print_success(self):
        """Print success message with asset statistics."""
        print("\n" + "="*60)
        print("SUCCESS: All required user assets validated")
        print("="*60)
        
        # Print statistics
        passages_path = self.user_assets_path / "passages.jsonl"
        qa_path = self.user_assets_path / "qa_pairs.jsonl"
        
        if passages_path.exists():
            try:
                with open(passages_path, 'r', encoding='utf-8') as f:
                    passage_count = sum(1 for line in f if line.strip())
                size_mb = passages_path.stat().st_size / (1024 * 1024)
                print(f"✓ Loaded {passage_count} passages from user_assets/passages.jsonl — total size {size_mb:.2f} MB")
            except:
                print("✓ passages.jsonl found")
                
        if qa_path.exists():
            try:
                with open(qa_path, 'r', encoding='utf-8') as f:
                    qa_count = sum(1 for line in f if line.strip())
                print(f"✓ Loaded {qa_count} QA pairs from user_assets/qa_pairs.jsonl")
            except:
                print("✓ qa_pairs.jsonl found")
                
        print("✓ config.yaml validated")
        print("\nYou can now proceed with running the Sanskrit Tutor!")
        print("="*60)


def main():
    """Command-line interface for config validation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Validate user-supplied assets for Sanskrit Tutor"
    )
    parser.add_argument(
        "--base-path", 
        default=".", 
        help="Base path to the project directory (default: current directory)"
    )
    
    args = parser.parse_args()
    
    validator = ConfigValidator(args.base_path)
    success = validator.validate_all()
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
