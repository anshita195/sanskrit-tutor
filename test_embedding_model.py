#!/usr/bin/env python3
"""
Quick test to verify which embedding model is actually being used.
"""

import yaml
from sentence_transformers import SentenceTransformer

def test_embedding_model():
    # Load config
    with open("user_assets/config.yaml", 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    model_name = config.get('embeddings_model', 'sentence-transformers/all-mpnet-base-v2')
    print(f"Config specifies: {model_name}")
    
    try:
        # Load the model
        print("Loading embedding model...")
        model = SentenceTransformer(model_name)
        
        print(f"✅ Model loaded successfully!")
        print(f"   Model name: {model_name}")
        print(f"   Dimension: {model.get_sentence_embedding_dimension()}")
        print(f"   Max sequence length: {model.get_max_seq_length()}")
        
        # Test with Sanskrit text
        test_texts = [
            "योगः कर्मसु कौशलम्",  # Devanagari
            "yogaḥ karmasu kauśalam",  # IAST  
            "Yoga is skill in action",  # English
            "योगः कर्मसु कौशलम् | yogaḥ karmasu kauśalam"  # Combined
        ]
        
        print("\n🧪 Testing with Sanskrit text...")
        embeddings = model.encode(test_texts)
        print(f"✅ Generated embeddings: {embeddings.shape}")
        
        # Test similarity
        from numpy import dot
        from numpy.linalg import norm
        
        def cosine_similarity(a, b):
            return dot(a, b) / (norm(a) * norm(b))
        
        print(f"\n📊 Similarity matrix:")
        print(f"   Devanagari vs IAST: {cosine_similarity(embeddings[0], embeddings[1]):.3f}")
        print(f"   Devanagari vs English: {cosine_similarity(embeddings[0], embeddings[2]):.3f}")
        print(f"   IAST vs English: {cosine_similarity(embeddings[1], embeddings[2]):.3f}")
        print(f"   Combined vs Devanagari: {cosine_similarity(embeddings[3], embeddings[0]):.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ ERROR: {e}")
        print(f"Install with: pip install sentence-transformers")
        return False

if __name__ == "__main__":
    test_embedding_model()
