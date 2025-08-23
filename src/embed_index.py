#!/usr/bin/env python3
"""
Embedding and FAISS index builder for Sanskrit Tutor RAG system.
Builds embeddings from user passages and creates searchable FAISS index.
"""

import os
import json
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import asdict
import pickle

# Try importing sentence transformers
try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    print("WARNING: sentence-transformers not available. Install with: pip install sentence-transformers")

from ingest import DataIngester, Passage


class EmbeddingIndexer:
    """Handles embedding generation and FAISS index creation."""
    
    def __init__(self, 
                 embeddings_model: str = "sentence-transformers/all-mpnet-base-v2",
                 base_path: str = "."):
        self.base_path = Path(base_path)
        self.embeddings_model_name = embeddings_model
        self.model = None
        self.dimension = None
        
    def load_embedding_model(self) -> bool:
        """
        Load the embedding model.
        
        Returns:
            True if model loaded successfully, False otherwise
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            print("ERROR: sentence-transformers package is required for embeddings.")
            print("Install with: pip install sentence-transformers")
            return False
            
        try:
            print(f"Loading embedding model: {self.embeddings_model_name}")
            self.model = SentenceTransformer(self.embeddings_model_name)
            
            # Get embedding dimension
            test_embedding = self.model.encode(["test"], show_progress_bar=False)
            self.dimension = test_embedding.shape[1]
            
            print(f"Embedding model loaded. Dimension: {self.dimension}")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to load embedding model: {str(e)}")
            print("Consider using a different model or checking your internet connection.")
            return False
    
    def load_precomputed_embeddings(self, embeddings_path: str, id_map_path: str) -> Optional[Tuple[np.ndarray, List[str]]]:
        """
        Load precomputed embeddings if available.
        
        Args:
            embeddings_path: Path to embeddings.npy file
            id_map_path: Path to id_map.json file
            
        Returns:
            Tuple of (embeddings array, passage IDs) or None if not available
        """
        embeddings_file = self.base_path / embeddings_path
        id_map_file = self.base_path / id_map_path
        
        if not (embeddings_file.exists() and id_map_file.exists()):
            return None
            
        try:
            embeddings = np.load(embeddings_file)
            with open(id_map_file, 'r', encoding='utf-8') as f:
                passage_ids = json.load(f)
                
            print(f"Loaded precomputed embeddings: {embeddings.shape[0]} passages, dim {embeddings.shape[1]}")
            return embeddings, passage_ids
            
        except Exception as e:
            print(f"WARNING: Failed to load precomputed embeddings: {str(e)}")
            return None
    
    def save_embeddings(self, embeddings: np.ndarray, passage_ids: List[str], 
                       embeddings_path: str, id_map_path: str):
        """
        Save computed embeddings for future use.
        
        Args:
            embeddings: Embedding vectors
            passage_ids: Corresponding passage IDs
            embeddings_path: Path to save embeddings.npy
            id_map_path: Path to save id_map.json
        """
        embeddings_file = self.base_path / embeddings_path
        id_map_file = self.base_path / id_map_path
        
        # Create directories if needed
        embeddings_file.parent.mkdir(parents=True, exist_ok=True)
        id_map_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            np.save(embeddings_file, embeddings)
            with open(id_map_file, 'w', encoding='utf-8') as f:
                json.dump(passage_ids, f, indent=2, ensure_ascii=False)
                
            print(f"Saved embeddings to: {embeddings_file.absolute()}")
            print(f"Saved ID mapping to: {id_map_file.absolute()}")
            
        except Exception as e:
            print(f"WARNING: Failed to save embeddings: {str(e)}")
    
    def generate_embeddings(self, passages: List[Passage], 
                           batch_size: int = 32) -> Tuple[np.ndarray, List[str]]:
        """
        Generate embeddings for passages.
        
        Args:
            passages: List of Passage objects
            batch_size: Batch size for embedding generation
            
        Returns:
            Tuple of (embeddings array, passage IDs)
        """
        if not self.model:
            raise ValueError("Embedding model not loaded. Call load_embedding_model() first.")
        
        print(f"Generating embeddings for {len(passages)} passages...")
        
        # Extract texts and IDs
        texts = [passage.get_searchable_text() for passage in passages]
        passage_ids = [passage.id for passage in passages]
        
        # Generate embeddings in batches
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        print(f"Generated embeddings: shape {embeddings.shape}")
        return embeddings, passage_ids
    
    def create_faiss_index(self, embeddings: np.ndarray, 
                          index_type: str = "FlatIP") -> faiss.Index:
        """
        Create FAISS index from embeddings.
        
        Args:
            embeddings: Embedding vectors
            index_type: Type of FAISS index ("FlatIP", "HNSW", "IVF")
            
        Returns:
            FAISS index
        """
        print(f"Creating FAISS index: type {index_type}, {embeddings.shape[0]} vectors")
        
        dimension = embeddings.shape[1]
        
        if index_type == "FlatIP":
            # Flat index with inner product (cosine similarity)
            index = faiss.IndexFlatIP(dimension)
            
        elif index_type == "HNSW":
            # Hierarchical Navigable Small World (good for medium datasets)
            index = faiss.IndexHNSWFlat(dimension, 32)  # 32 connections per node
            index.hnsw.efConstruction = 200  # Higher = better quality, slower build
            index.hnsw.efSearch = 128  # Higher = better quality, slower search
            
        elif index_type == "IVF":
            # Inverted file index (good for large datasets)
            nlist = min(100, max(1, int(embeddings.shape[0] / 10)))  # Number of clusters
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            
            # Train the index
            print("Training IVF index...")
            index.train(embeddings)
            
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        # Add embeddings to index
        print("Adding embeddings to index...")
        index.add(embeddings)
        
        print(f"FAISS index created: {index.ntotal} vectors indexed")
        return index
    
    def save_faiss_index(self, index: faiss.Index, index_path: str, 
                        passages: List[Passage], passage_ids: List[str]):
        """
        Save FAISS index and metadata.
        
        Args:
            index: FAISS index to save
            index_path: Path to save index file
            passages: List of Passage objects
            passage_ids: List of passage IDs (same order as index)
        """
        index_file = self.base_path / index_path
        metadata_file = index_file.with_suffix('.metadata.pkl')
        
        # Create directory if needed
        index_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Save FAISS index
            faiss.write_index(index, str(index_file.absolute()))
            
            # Save metadata
            metadata = {
                'passage_ids': passage_ids,
                'passages': [passage.to_dict() for passage in passages],
                'embeddings_model': self.embeddings_model_name,
                'dimension': self.dimension,
                'total_passages': len(passages)
            }
            
            with open(metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
                
            print(f"Saved FAISS index to: {index_file.absolute()}")
            print(f"Saved metadata to: {metadata_file.absolute()}")
            
        except Exception as e:
            print(f"ERROR: Failed to save FAISS index: {str(e)}")
            raise
    
    def load_faiss_index(self, index_path: str) -> Tuple[faiss.Index, Dict[str, Any]]:
        """
        Load FAISS index and metadata.
        
        Args:
            index_path: Path to index file
            
        Returns:
            Tuple of (FAISS index, metadata dict)
        """
        index_file = self.base_path / index_path
        metadata_file = index_file.with_suffix('.metadata.pkl')
        
        if not index_file.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_file.absolute()}")
        if not metadata_file.exists():
            raise FileNotFoundError(f"Index metadata not found: {metadata_file.absolute()}")
            
        try:
            # Load FAISS index
            index = faiss.read_index(str(index_file.absolute()))
            
            # Load metadata
            with open(metadata_file, 'rb') as f:
                metadata = pickle.load(f)
                
            print(f"Loaded FAISS index: {index.ntotal} vectors")
            print(f"Embeddings model: {metadata.get('embeddings_model', 'unknown')}")
            
            return index, metadata
            
        except Exception as e:
            print(f"ERROR: Failed to load FAISS index: {str(e)}")
            raise
    
    def search(self, index: faiss.Index, metadata: Dict[str, Any], 
              query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search the FAISS index with a query.
        
        Args:
            index: FAISS index
            metadata: Index metadata
            query: Search query
            k: Number of results to return
            
        Returns:
            List of search results with scores and passages
        """
        if not self.model:
            raise ValueError("Embedding model not loaded. Call load_embedding_model() first.")
            
        # Generate query embedding
        query_embedding = self.model.encode([query], show_progress_bar=False)
        
        # Search index
        scores, indices = index.search(query_embedding, k)
        
        # Format results
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0:  # Valid index
                passage_data = metadata['passages'][idx]
                results.append({
                    'passage': passage_data,
                    'score': float(score),
                    'passage_id': passage_data['id']
                })
                
        return results
    
    def build_index_from_config(self, config_path: str) -> bool:
        """
        Build complete index from configuration file.
        
        Args:
            config_path: Path to config.yaml file
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load data using ingester
            print("Loading user data...")
            ingester = DataIngester(self.base_path)
            data = ingester.load_all_data(config_path)
            
            passages = data['passages']
            config = data['config']
            
            # Load embedding model
            if not self.load_embedding_model():
                return False
            
            # Check for precomputed embeddings
            embeddings_path = "user_assets/embeddings/embeddings.npy"
            id_map_path = "user_assets/embeddings/id_map.json"
            
            precomputed = self.load_precomputed_embeddings(embeddings_path, id_map_path)
            
            if precomputed:
                embeddings, passage_ids = precomputed
                
                # Verify IDs match current passages
                current_ids = [p.id for p in passages]
                if passage_ids != current_ids:
                    print("WARNING: Precomputed embeddings don't match current passages. Regenerating...")
                    embeddings, passage_ids = self.generate_embeddings(passages)
                    self.save_embeddings(embeddings, passage_ids, embeddings_path, id_map_path)
            else:
                # Generate new embeddings
                embeddings, passage_ids = self.generate_embeddings(passages)
                self.save_embeddings(embeddings, passage_ids, embeddings_path, id_map_path)
            
            # Determine index type based on dataset size
            num_passages = len(passages)
            if num_passages < 1000:
                index_type = "FlatIP"
            elif num_passages < 10000:
                index_type = "HNSW"
            else:
                index_type = "IVF"
                
            print(f"Using {index_type} index for {num_passages} passages")
            
            # Create FAISS index
            index = self.create_faiss_index(embeddings, index_type)
            
            # Save index
            index_path = config['faiss_index_path']
            self.save_faiss_index(index, index_path, passages, passage_ids)
            
            print(f"\nIndex building completed successfully!")
            print(f"Index file: {(self.base_path / index_path).absolute()}")
            
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to build index: {str(e)}")
            return False


def main():
    """Command-line interface for index building."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Build embeddings and FAISS index for Sanskrit Tutor"
    )
    parser.add_argument(
        "--config",
        default="user_assets/config.yaml",
        help="Path to configuration file (default: user_assets/config.yaml)"
    )
    parser.add_argument(
        "--embeddings-model",
        default="sentence-transformers/all-mpnet-base-v2",
        help="Embedding model to use"
    )
    parser.add_argument(
        "--index-type",
        choices=["FlatIP", "HNSW", "IVF"],
        help="FAISS index type (auto-selected based on data size if not specified)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation (default: 32)"
    )
    parser.add_argument(
        "--test-search",
        help="Test the index with a sample query after building"
    )
    
    args = parser.parse_args()
    
    try:
        indexer = EmbeddingIndexer(args.embeddings_model)
        
        # Build index
        success = indexer.build_index_from_config(args.config)
        
        if not success:
            exit(1)
            
        # Test search if requested
        if args.test_search:
            print(f"\nTesting search with query: '{args.test_search}'")
            
            # Load the built index
            import yaml
            with open(args.config, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
                
            index, metadata = indexer.load_faiss_index(config['faiss_index_path'])
            
            results = indexer.search(index, metadata, args.test_search, k=3)
            
            print(f"Found {len(results)} results:")
            for i, result in enumerate(results, 1):
                passage = result['passage']
                print(f"{i}. [{passage['id']}] Score: {result['score']:.3f}")
                print(f"   Text: {passage['text_iast'][:100]}...")
                print(f"   Work: {passage['work']} {passage['chapter']}.{passage['verse']}")
                print()
        
        print("Index building completed successfully!")
        
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        exit(1)
    except Exception as e:
        print(f"ERROR: {str(e)}")
        exit(1)


if __name__ == "__main__":
    main()
